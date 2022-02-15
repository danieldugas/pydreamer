from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import rnn
from .functions import *
from .common import *

DEBUG = True

class TSSMCore(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        self.cell = self

    def forward_prior(self,
                      action: Tensor,      # tensor(B, A)
                      reset: Tensor,       # tensor(B)
                      in_state: Tuple[Tensor, Tensor, Tensor, Tensor],    # [(BI,D) (BI,S)]
                      ):
        B, A = action.shape[:2]
        context_embed, context_action, context_reset, _ = in_state
        zeroth_deter_state = torch.zeros((1, B, D), device=action.device)
        Co, B, E = context_embed.shape
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
        rec_embed = self.to_feature(last_deterministic_state, prior_sample) # (B, E)

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
        I = iwae_samples

        # in state should be the whole sequence otherwise cell.forward_prior doesn't map to anything
        # out state should be the whole sequence ( truncated to max length? )
        # in state:
        # (embed Co, B, E, action (Co, B, A), reset (Co, B), h+z ((B, E), (B, S)))
        # the very first posterior needs a deterministic state from 'before' the sequence
        # no sampled state as input, because TSSM doesn't have a way to infer h from z
        context_embed, context_action, context_reset, in_deter_state = in_state
        Co, B, E = context_embed.shape

        # to give the transformer the ability to use 'previous state', we treat it as a context
        # prepended =  context + sequence
        prepended_embed = torch.cat((context_embed, embed), 0)
        prepended_action = torch.cat((context_action, action), 0)
        prepended_reset = torch.cat((context_reset, reset), 0)
        P, _, _ = prepended_embed.shape

        # Multiply batch dimension by I samples

        def expand(x):
            # (T,B,X) -> (T,BI,X)
            return x.unsqueeze(2).expand(T, B, I, -1).reshape(T, B * I, -1)

        if do_open_loop:
            embeds = expand(embed).unbind(0)     # (T,B,...) => List[(BI,...)]
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
            rec_embed = self.to_feature(deterministic_states, post_samples)   # (T,BI,D+S)
        else: # all time-steps in a single pass (parallel)
            position_embeddings = self.pos_emb[:, :P, :]  # each position maps to a (learnable) vector
            # forward the GPT model
            full_embeddings = self.drop(
                self.embed_to_deter(prepended_embed.moveaxis(0, 1).view(B * P, E)).view(B, P, D)
                + position_embeddings
            ) # B, P, D
            x = self.blocks(full_embeddings)
            # the current deterministic state is the 'next deterministic state' predicted from the previous step
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
            ).view(B, T, self.prior_size).moveaxis(0, 1) # (T, B, S)
            priors = self.dstate_to_prior(
                deterministic_states.moveaxis(0, 1).view(B * T, D)
            ).view(B, T, self.prior_size).moveaxis(0, 1) # (T, B, S)
            # sample from posterior
            posterior_distr = self.zdistr(posts)
            post_samples = posterior_distr.rsample().view(T, B, S)
            rec_embed = self.to_feature(deterministic_states, post_samples) # (T, B, E)

        states = []
        for i in range(T):
            # sequence up to 'now' (-2, -1, 0), (-2, -1, 0, 1), (-2, -1, 0, 1, 2), ...
            # states[0] = context + first input, first next deterministic state
            # first deterministic state is result of Transformer(context + first input)
            uptonow_embed = prepended_embed[:Co+i+1]
            uptonow_action = prepended_action[:Co+i+1]
            uptonow_reset = prepended_action[:Co+i+1]
            states.append((uptonow_embed, uptonow_action, uptonow_reset, deterministic_states[i]))

        priors = priors.reshape(T, B, I, -1)
        posts = posts.reshape(T, B, I, -1)  # (T,BI,X) => (T,B,I,X)
        post_samples = post_samples.reshape(T, B, I, -1)
        rec_embed = rec_embed.reshape(T, B, I, -1)
        states = [(e.reshape(T, B, I, E), a.reshape(T, B, I, A), r.reshape(T, B, I), z.reshape(T, B, I, S))
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
        ES = hz.shape[-1]
        rec_embed = self.sampled_fullstate_to_embed(hz.view(-1, ES)).view(h.shape) # E = D
        return rec_embed # (P, B, E) or (B, E)

    def feature_replace_z(self, features: Tensor, z: Tensor):
        raise NotImplementedError

    def zdistr(self, pp: Tensor) -> D.Distribution:
        # pp = post or prior
        if self.stoch_discrete:
            logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_discrete))
            distr = D.OneHotCategoricalStraightThrough(logits=logits.float())  # NOTE: .float() needed to force float32 on AMP
            distr = D.independent.Independent(distr, 1)  # This makes d.entropy() and d.kl() sum over stoch_dim
            return distr
        else:
            return diag_normal(pp)
