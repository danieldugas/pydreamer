from typing import Tuple

import torch
import torch.distributions as Distr
import torch.nn as nn
from torch import Tensor

from navrep.models.gpt import Block

from pydreamer.models.functions import diag_normal

DEBUG = True

class TransfConfig:
    block_size = 32

    n_embd = 2048 # the transformer embedding size (not same as embed_dim, the encoder output)
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
        self.config.embed_dim = embed_dim
        self.config.deter_dim = deter_dim
        self.config.action_dim = action_dim
        self.config.prior_size = stoch_dim * stoch_discrete # 32 variables * 32 categories
        self.config.stoch_discrete = stoch_discrete
        self.config.stoch_dim = stoch_dim
        self.config.n_embd = deter_dim

        self.embd_pdrop = 0.1

        E = self.config.embed_dim
        D = self.config.deter_dim
        S = self.config.prior_size
        self.pos_emb = nn.Parameter(torch.zeros(1, self.config.block_size, D))
        self.embed_to_deter = nn.Linear(E, D)
        self.drop = nn.Dropout(self.config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(D)
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

    def forward_prior(self,
                      action: Tensor,      # tensor(B, A)
                      reset: Tensor,       # tensor(B)
                      in_state: Tuple[Tensor, Tensor, Tensor, Tensor],    # [(T,BI,D) (BI,S)]
                      ):
        B, A = action.shape[:2]
        D = self.config.deter_dim
        S = self.config.prior_size
        context_embed, context_action, context_reset, _ = in_state
        if reset is None:
            reset = torch.zeros((B,), device=action.device)
        zeroth_deter_state = torch.zeros((1, B, D), device=action.device)
        Co, B, E = context_embed.shape
        if Co > self.config.block_size:
#             print("WARNING: context embedding is longer than block size")
            context_embed = context_embed[-self.config.block_size:, :, :]
            context_action = context_action[-self.config.block_size:, :, :]
            context_reset = context_reset[-self.config.block_size:, :]
            Co, B, E = context_embed.shape

        position_embeddings = self.pos_emb[:, :Co, :]  # each position maps to a (learnable) vector
        # forward the GPT model
        full_embeddings = self.drop(
            self.embed_to_deter(context_embed.moveaxis(0, 1).reshape(B * Co, E)).reshape(B, Co, D)
            + position_embeddings
        ) # B, Co, D
        x = self.blocks(full_embeddings.float())
        next_deterministic_states = self.ln_f(x).moveaxis(0, 1) # (Co, B, D)
        deterministic_states = torch.cat((zeroth_deter_state, # in case Co is []
                                          next_deterministic_states[:-1]), 0) # (Co, B, D)
        last_deterministic_state = deterministic_states[-1] # (B, D)
        B, D = last_deterministic_state.shape
        prior = self.dstate_to_prior(last_deterministic_state)# (B, S)
        # sample from prior
        prior_distr = self.zdistr(prior)
        prior_sample = prior_distr.rsample().view(B, S)
        rec_embed = self.hz_to_feature(last_deterministic_state, prior_sample) # actually a prediction (B, E)

        uptonow_embed = torch.cat((context_embed, rec_embed.view(1, B, E)), dim=0) # (Co+1, B, E)
        uptonow_action = torch.cat((context_action, action.view(1, B, A)), dim=0) # (Co+1, B, A)
        uptonow_reset = torch.cat((context_reset, reset.view(1, B, 1)), dim=0) # (Co+1, B, 1)
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
        prepended_reset = torch.cat((context_reset, reset.view(T, B, 1)), 0)
        P, _, _ = prepended_embed.shape
        if P > self.config.block_size:
            print("WARNING: prepended embedding is longer than block size")
            prepended_embed = prepended_embed[-self.config.block_size:, :, :]
            prepended_action = prepended_action[-self.config.block_size:, :, :]
            prepended_reset = prepended_reset[-self.config.block_size:, :, :]
            P, _, _ = prepended_embed.shape
            Co = P - T
            in_state = (context_embed[-Co:, :, :],
                        context_action[-Co:, :, :],
                        context_reset[-Co:, :, :],
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
            rec_embed = self.hz_to_feature(deterministic_states, post_samples)   # (T,BI,E)
        else: # all time-steps in a single pass (parallel)
            position_embeddings = self.pos_emb[:, :P, :]  # each position maps to a (learnable) vector
            # forward the GPT model
            full_embeddings = self.drop(
                self.embed_to_deter(prepended_embed.moveaxis(0, 1).reshape(B * P, E)).reshape(B, P, D)
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
                self.dstate_to_mix(deterministic_states.moveaxis(0, 1).reshape(B * T, D))
                + self.embedding_to_mix(embed.view(B * T, E))
            ).view(B, T, S).moveaxis(0, 1) # (T, B, S)
            priors = self.dstate_to_prior(
                deterministic_states.moveaxis(0, 1).reshape(B * T, D)
            ).view(B, T, S).moveaxis(0, 1) # (T, B, S)
            # sample from posterior
            posterior_distr = self.zdistr(posts)
            post_samples = posterior_distr.rsample().view(T, B, S)
            rec_embed = self.hz_to_feature(deterministic_states, post_samples) # (T, B, E)
            # We want rec_embed to be the "same" as embed.
            # Could do this with a loss, but loss is defined outside.
            # so we sometimes just pass the original embeddings straight through
            rec_embed = torch.where(torch.rand(embed.shape, device=action.device) > 0.5, embed, rec_embed)

        states = []
        for i in range(T):
            # sequence up to 'now' (-2, -1, 0), (-2, -1, 0, 1), (-2, -1, 0, 1, 2), ...
            # states[0] = context + first input, first next deterministic state
            # first deterministic state is result of Transformer(context + first input)
            uptonow_embed = prepended_embed[:Co+i+1]
            uptonow_action = prepended_action[:Co+i+1]
            uptonow_reset = prepended_reset[:Co+i+1]
            states.append((uptonow_embed, uptonow_action, uptonow_reset, deterministic_states[i]))

        priors = priors.reshape(T, B, Iw, -1)
        posts = posts.reshape(T, B, Iw, -1)  # (T,BI,X) => (T,B,I,X)
        post_samples = post_samples.reshape(T, B, Iw, -1)
        rec_embed = rec_embed.reshape(T, B, Iw, -1)
        states = [(e.reshape(-1, B, Iw, E), # -1 because each state in the sequence has a different ctx
                   a.reshape(-1, B, Iw, A),
                   r.reshape(-1, B, Iw, 1),
                   z.reshape(B, Iw, D))
                  for (e, a, r, z) in states]
        for (e, a, r, z) in states:
            assert e.shape[:3] == a.shape[:3] and a.shape[:3] == r.shape[:3]
            assert e.shape[1:3] == z.shape[:2]

        out_state = self.last_state(states) # ((T, B, E), (T, B, A), (T, B, 1), (B, D))

        return (
            priors,                      # tensor(T,B,I,2S)
            posts,                       # tensor(T,B,I,2S)
            post_samples,                     # tensor(T,B,I,S)
            rec_embed,                    # tensor(T,B,I,D+S)
            states,
            out_state,
        )

    def init_state(self, batch_size):
        device = next(self.parameters()).device
        B = batch_size
        E = self.config.embed_dim
        A = self.config.action_dim
        D = self.config.deter_dim
        uptonow_embed = torch.zeros((0, B, E), device=device)
        uptonow_action = torch.zeros((0, B, A), device=device)
        uptonow_reset = torch.zeros((0, B, 1), device=device)
        last_deterministic_state = torch.zeros((B, D), device=device)
        out_state = (uptonow_embed, uptonow_action, uptonow_reset, last_deterministic_state)
        return out_state

    def states_to_deter(self, states):
        deterministic_states = []
        for _, _, _, deterministic_state in states:
            deterministic_states.append(deterministic_state)
        deter = torch.stack(deterministic_states) # (T, B, I, D)
        T, B, Iw, D = deter.shape
        return deter

    def state_to_feature(self, state: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        # unlike RSSM, here we have to recompute the posterior from the state (context sequence)
        D = self.config.deter_dim
        S = self.config.prior_size
        context_embed, _, _, deterministic_state = state
        T, B, E = context_embed.shape
        B, D = deterministic_state.shape
        last_context_embed = context_embed[-1]
        post = self.mix_to_post(
            self.dstate_to_mix(deterministic_state)
            + self.embedding_to_mix(last_context_embed)
        ).view(B, S)
        # sample from posterior
        posterior_distr = self.zdistr(post)
        post_sample = posterior_distr.rsample().view(B, S)
        rec_embed = self.hz_to_feature(deterministic_state, post_sample) # (B, E)
        return rec_embed

    def hz_to_feature(self, h: Tensor, z: Tensor) -> Tensor: # either (P, B, E/S) or (B, E/S)
        hz = torch.cat((h, z), -1) # P, B, E+S
        DS = hz.shape[-1]
        E = self.config.embed_dim
        rec_embed = self.sampled_fullstate_to_embed(hz.view(-1, DS)).view(h.shape[:-1] + (E,))
        return rec_embed # (P, B, E) or (B, E)

    def first_state(self, states: list) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (e, a, r, z) = states[0]
        return (e[:,:,0,:], a[:,:,0,:], r[:,:,0,:], z[:,0,:]) # remove I dim

    def last_state(self, states: list) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (e, a, r, z) = states[-1]
        return (e[:,:,0,:], a[:,:,0,:], r[:,:,0,:], z[:,0,:]) # remove I dim

    def feature_replace_z(self, features: Tensor, z: Tensor):
        raise NotImplementedError

    def zdistr(self, pp: Tensor) -> Distr.Distribution:
        # pp = post or prior
        if self.config.stoch_discrete:
            logits = pp.reshape(pp.shape[:-1] + (self.config.stoch_dim, self.config.stoch_discrete))
            # NOTE: .float() needed to force float32 on AMP
            distr = Distr.OneHotCategoricalStraightThrough(logits=logits.float())
            # This makes d.entropy() and d.kl() sum over stoch_dim
            distr = Distr.independent.Independent(distr, 1)
            return distr
        else:
            return diag_normal(pp)


if __name__ == '__main__':
    import argparse

    import torch.nn as nn

    from pydreamer import tools
    from pydreamer.models.dreamer import Dreamer

    tools.configure_logging(prefix='[TRAIN]')
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = tools.read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        parser.add_argument(f'--{key}', type=type(value) if value is not None else str, default=value)
    conf = parser.parse_args(remaining)

    device = torch.device(conf.device)
    conf.tssm = True
    conf.batch_size = 40
    if conf.model == 'dreamer':
        model = Dreamer(conf)
    else:
        assert False, conf.model
    model.to(device)
    print(model)

    obs = {}
    obs["action"] = torch.rand(conf.batch_length, conf.batch_size, conf.action_dim, device=device)
    obs["reset"] = torch.zeros(conf.batch_length, conf.batch_size, device=device) > 0
    obs["terminal"] = torch.zeros(conf.batch_length, conf.batch_size, device=device)
    obs["image"] = torch.rand(conf.batch_length, conf.batch_size,
                              conf.image_channels, conf.image_size, conf.image_size, device=device)
    obs["vecobs"] = torch.rand(conf.batch_length, conf.batch_size, conf.vecobs_size, device=device)
    obs["reward"] = torch.rand(conf.batch_length, conf.batch_size, device=device)
    state = model.init_state(conf.batch_size * conf.iwae_samples)
    losses, new_state, loss_metrics, tensors, dream_tensors = \
        model.training_step(obs,
                            state,
                            iwae_samples=conf.iwae_samples,
                            imag_horizon=conf.imag_horizon,
                            do_image_pred=False,
                            do_dream_tensors=False)
    model.training_step(obs,
                        state,
                        iwae_samples=conf.iwae_samples,
                        imag_horizon=conf.imag_horizon,
                        do_image_pred=False,
                        do_dream_tensors=True)
    model.training_step(obs,  # observation will be ignored in forward pass because of imagine=True
                        state,
                        iwae_samples=conf.iwae_samples,
                        imag_horizon=conf.imag_horizon,
                        do_open_loop=True,
                        do_image_pred=True)
