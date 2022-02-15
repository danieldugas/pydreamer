from typing import Tuple

import torch
import torch.distributions as Distr
import torch.nn as nn
from torch import Tensor

from navrep.models.gpt import Block

from .functions import diag_normal

DEBUG = True

class TransfConfig:
    block_size = 64
    n_embd = 1536

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    n_layer = 8
    n_head = 8

class TSSMCore(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim,
                 gru_layers, gru_type, layer_norm):
        super().__init__()

        # config
        self.config = TransfConfig()
        self.config.deter_dim = deter_dim
        self.config.action_dim = action_dim
        self.config.prior_size = stoch_dim * stoch_discrete # 32 variables * 32 categories

        self.embd_pdrop = 0.1

        E = self.config.n_embd
        D = self.config.deter_dim
        S = self.config.prior_size
        self.pos_emb = nn.Parameter(torch.zeros(1, self.config.block_size, E))
        self.embed_to_deter = nn.Linear(E, D)
        self.drop = nn.Dropout(self.config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(E)
        # prior and posteriors
        self.dstate_to_prior = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Linear(D, S),
        )
        self.dstate_to_mix = nn.Linear(D, D)
        self.embedding_to_mix = nn.Linear(E, D)
        self.mix_to_post = nn.Sequential(
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Linear(D, S),
        )
        self.sampled_fullstate_to_embed = nn.Linear(
            D + S, E)

        self.cell = self

    def forward_prior(self,
                      action: Tensor,      # tensor(B, A)
                      reset: Tensor,       # tensor(B)
                      in_state: Tuple[Tensor, Tensor, Tensor, Tensor],    # [(BI,D) (BI,S)]
                      ):
        B, A = action.shape[:2]
        D = self.config.deter_dim
        S = self.config.prior_size
        context_embed, context_action, context_reset, _ = in_state
        zeroth_deter_state = torch.zeros((1, B, D), device=action.device)
        Co, B, E = context_embed.shape
        if Co > self.config.block_size:
            print("WARNING: context embedding is longer than block size")
            context_embed = context_embed[-self.config.block_size:, :, :]
            context_action = context_action[-self.config.block_size:, :, :]
            context_reset = context_reset[-self.config.block_size:, :]

        position_embeddings = self.pos_emb[:, :Co, :]  # each position maps to a (learnable) vector
        # forward the GPT model
        full_embeddings = self.drop(
            self.embed_to_deter(context_embed.moveaxis(0, 1).view(B * Co, E)).view(B, Co, D)
            + position_embeddings
        ) # B, Co, D
        x = self.blocks(full_embeddings)
        next_deterministic_states = self.ln_f(x).moveaxis(0, 1) # (Co, B, D)
        deterministic_states = torch.cat((zeroth_deter_state, # in case Co is []
                                          next_deterministic_states[:-1]), 0) # (Co, B, D)
        last_deterministic_state = deterministic_states[-1] # (B, D)
        prior = self.dstate_to_prior(last_deterministic_state)# (B, S)
        # sample from prior
        prior_distr = self.zdistr(prior)
        prior_sample = prior_distr.rsample().view(B, S)
        rec_embed = self.to_feature(last_deterministic_state, prior_sample) # actually a prediction (B, E)

        uptonow_embed = torch.cat((context_embed, rec_embed.view(1, B, E)), dim=0) # (Co+1, B, E)
        uptonow_action = torch.cat((context_action, action.view(1, B, A)), dim=0) # (Co+1, B, A)
        uptonow_reset = torch.cat((context_reset, reset.view(1, B)), dim=0) # (Co+1, B)
        out_state = (uptonow_embed, uptonow_action, uptonow_reset, last_deterministic_state)

        return prior, out_state

    def forward(self,
                embed: Tensor,       # tensor(T, B, E)
                action: Tensor,      # tensor(T, B, A)
                reset: Tensor,       # tensor(T, B)
                in_state: Tuple[Tensor, Tensor, Tensor, Tensor],    # [(BI,D) (BI,S)]
                iwae_samples: int = 1,
                do_open_loop=False,
                ):
        T, B = embed.shape[:2]
        Iw = iwae_samples

        # in state should be the whole sequence otherwise cell.forward_prior doesn't map to anything
        # out state should be the whole sequence ( truncated to max length? )
        # in state:
        # (embed Co, B, E, action (Co, B, A), reset (Co, B), h+z ((B, E), (B, S)))
        # the very first posterior needs a deterministic state from 'before' the sequence
        # no sampled state as input, because TSSM doesn't have a way to infer h from z
        context_embed, context_action, context_reset, in_deter_state = in_state
        Co, B, E = context_embed.shape
        T, B, E = embed.shape
        T, B, A = action.shape
        if T > self.config.block_size:
            raise ValueError("Cannot handle sequence longer than block size")

        S = self.config.prior_size
        D = self.config.deter_dim

        # to give the transformer the ability to use 'previous state', we treat it as a context
        # prepended =  context + sequence
        prepended_embed = torch.cat((context_embed, embed), 0)
        prepended_action = torch.cat((context_action, action), 0)
        prepended_reset = torch.cat((context_reset, reset), 0)
        P, _, _ = prepended_embed.shape
        if P > self.config.block_size:
            print("WARNING: prepended embedding is longer than block size")
            prepended_embed = prepended_embed[-self.config.block_size:, :, :]
            prepended_action = prepended_action[-self.config.block_size:, :, :]
            prepended_reset = prepended_reset[-self.config.block_size:, :]
            P, _, _ = prepended_embed.shape
            Co = P - T
            in_state = (context_embed[-Co:, :, :],
                        context_action[-Co:, :, :],
                        context_reset[-Co:, :],
                        in_deter_state)

        # Multiply batch dimension by I samples

        def expand(x):
            # (T,B,X) -> (T,BI,X)
            return x.unsqueeze(2).expand(T, B, Iw, -1).reshape(T, B * Iw, -1)

        if do_open_loop:
            actions = expand(action).unbind(0)
            reset_masks = expand(~reset.unsqueeze(2)).unbind(0)
            priors = []
            posts = []
            states = []
            post_samples = []
            deterministic_states = []

            state = in_state
            for i in range(T):
                prior, state = self.forward_prior(actions[i], reset_masks[i], state)  # open loop: post=prior
                posterior_distr = self.zdistr(prior)
                posterior_sample = posterior_distr.rsample().view(B, S)
                _, _, _, deterministic_state = state
                posts.append(prior)
                states.append(state)
                post_samples.append(posterior_sample)
                deterministic_states.append(deterministic_state)

            priors = torch.stack(posts)  # (T,BI,2S)
            posts = torch.stack(posts)          # (T,BI,2S)
            post_samples = torch.stack(post_samples)      # (T,BI,S)
            deterministic_states = torch.stack(deterministic_states)  # (T,BI,D)
            rec_embed = self.to_feature(deterministic_states, post_samples)   # (T,BI,E)
        else: # all time-steps in a single pass (parallel)
            position_embeddings = self.pos_emb[:, :P, :]  # each position maps to a (learnable) vector
            # forward the GPT model
            full_embeddings = self.drop(
                self.embed_to_deter(prepended_embed.moveaxis(0, 1).view(B * P, E)).view(B, P, D)
                + position_embeddings
            ) # B, P, D
            x = self.blocks(full_embeddings)
            # the current deterministic state is the 'next deterministic state' predicted from
            # the previous step
            prepended_next_deterministic_states = self.ln_f(x).moveaxis(0, 1) # (P, B, D)
            next_deterministic_states = prepended_next_deterministic_states[Co:, :, :] # (T, B, D)
            if Co != 0:
                in_deter_state = prepended_next_deterministic_states[:Co, :, :][-1, :, :] # (1, B, D)
            # we could also use the last context 'next deterministic state', but this works with Co = 0
            deterministic_states = torch.cat((in_deter_state.view(1, B, D),
                                              next_deterministic_states[:-1]), 0) # (T, B, D)
            posts = self.mix_to_post(
                self.dstate_to_mix(deterministic_states.moveaxis(0, 1).view(B * T, D))
                + self.embedding_to_mix(full_embeddings.view(B * T, E))
            ).view(B, T, S).moveaxis(0, 1) # (T, B, S)
            priors = self.dstate_to_prior(
                deterministic_states.moveaxis(0, 1).view(B * T, D)
            ).view(B, T, S).moveaxis(0, 1) # (T, B, S)
            # sample from posterior
            posterior_distr = self.zdistr(posts)
            post_samples = posterior_distr.rsample().view(T, B, S)
            rec_embed = self.to_feature(deterministic_states, post_samples) # (T, B, E)
            # We want rec_embed to be the "same" as embed.
            # Could do this with a loss, but loss is defined outside.
            # so we sometimes just pass the original embeddings straight through
            rec_embed = torch.where(torch.rand(embed.shape) > 0.5, embed, rec_embed)

        states = []
        for i in range(T):
            # sequence up to 'now' (-2, -1, 0), (-2, -1, 0, 1), (-2, -1, 0, 1, 2), ...
            # states[0] = context + first input, first next deterministic state
            # first deterministic state is result of Transformer(context + first input)
            uptonow_embed = prepended_embed[:Co+i+1]
            uptonow_action = prepended_action[:Co+i+1]
            uptonow_reset = prepended_action[:Co+i+1]
            states.append((uptonow_embed, uptonow_action, uptonow_reset, deterministic_states[i]))

        priors = priors.reshape(T, B, Iw, -1)
        posts = posts.reshape(T, B, Iw, -1)  # (T,BI,X) => (T,B,I,X)
        post_samples = post_samples.reshape(T, B, Iw, -1)
        rec_embed = rec_embed.reshape(T, B, Iw, -1)
        states = [(e.reshape(T, B, Iw, E),
                   a.reshape(T, B, Iw, A),
                   r.reshape(T, B, Iw),
                   z.reshape(T, B, Iw, S))
                  for (e, a, r, z) in states]
        out_state = states[-1]

        return (
            priors,                      # tensor(T,B,I,2S)
            posts,                       # tensor(T,B,I,2S)
            post_samples,                     # tensor(T,B,I,S)
            rec_embed,                    # tensor(T,B,I,D+S)
            states,
            out_state,
        )

    def init_state(self, batch_size):
        B = batch_size
        E = self.config.n_embd
        A = self.config.action_dim
        D = self.config.deter_dim
        uptonow_embed = torch.zeros((0, B, E))
        uptonow_action = torch.zeros((0, B, A))
        uptonow_reset = torch.zeros((0, B))
        last_deterministic_state = torch.zeros((B, D))
        out_state = (uptonow_embed, uptonow_action, uptonow_reset, last_deterministic_state)
        return out_state

    def states_to_deter(self, states):
        deterministic_states = []
        for _, _, _, deterministic_state in states:
            deterministic_states.append(deterministic_state)
        deter = torch.stack(deterministic_states) # (T, B, D)
        T, B, D = deter.shape
        return deter

    def to_feature(self, h: Tensor, z: Tensor) -> Tensor: # either (P, B, E/S) or (B, E/S)
        hz = torch.cat((h, z), -1) # P, B, E+S
        DS = hz.shape[-1]
        E = self.config.n_embd
        rec_embed = self.sampled_fullstate_to_embed(hz.view(-1, DS)).view(h.shape[:-1] + (E,))
        return rec_embed # (P, B, E) or (B, E)

    def feature_replace_z(self, features: Tensor, z: Tensor):
        raise NotImplementedError

    def zdistr(self, pp: Tensor) -> Distr.Distribution:
        # pp = post or prior
        if self.stoch_discrete:
            logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_discrete))
            # NOTE: .float() needed to force float32 on AMP
            distr = Distr.OneHotCategoricalStraightThrough(logits=logits.float())
            # This makes d.entropy() and d.kl() sum over stoch_dim
            distr = Distr.independent.Independent(distr, 1)
            return distr
        else:
            return diag_normal(pp)
