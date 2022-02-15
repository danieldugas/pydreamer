from typing import Any, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ..tools import *
from .a2c import *
from .common import *
from .functions import *
from .encoders import *
from .decoders import *
from .rnn import *
from .rssm import *
from .tssm import TSSMCore
from .probes import *


class Dreamer(nn.Module):

    def __init__(self, conf):
        super().__init__()

        # World model

        self.wm = WorldModel(conf)
        state_dim = self.wm.features_dim

        # Actor critic

        self.ac = ActorCritic(in_dim=state_dim,
                              out_actions=conf.action_dim,
                              layer_norm=conf.layer_norm,
                              gamma=conf.gamma,
                              lambda_gae=conf.lambda_gae,
                              entropy_weight=conf.entropy,
                              target_interval=conf.target_interval,
                              actor_grad=conf.actor_grad,
                              actor_dist=conf.actor_dist,
                              )

        # Map probe

        if conf.map_model == 'direct':
            map_model = MapProbeHead(state_dim + 4, conf)
        elif conf.map_model == 'none':
            map_model = NoProbeHead()
        else:
            raise NotImplementedError(f'Unknown map_model={conf.map_model}')
        self.map_model = map_model

    def init_optimizers(self, lr, lr_actor=None, lr_critic=None, eps=1e-5):
        optimizer_wm = torch.optim.AdamW(self.wm.parameters(), lr=lr, eps=eps)
        optimizer_map = torch.optim.AdamW(self.map_model.parameters(), lr=lr, eps=eps)
        optimizer_actor = torch.optim.AdamW(self.ac.actor.parameters(), lr=lr_actor or lr, eps=eps)
        optimizer_critic = torch.optim.AdamW(self.ac.critic.parameters(), lr=lr_critic or lr, eps=eps)
        return optimizer_wm, optimizer_map, optimizer_actor, optimizer_critic

    def grad_clip(self, grad_clip, grad_clip_ac=None):
        grad_metrics = {
            'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), grad_clip),
            'grad_norm_map': nn.utils.clip_grad_norm_(self.map_model.parameters(), grad_clip),
            'grad_norm_actor': nn.utils.clip_grad_norm_(self.ac.actor.parameters(), grad_clip_ac or grad_clip),
            'grad_norm_critic': nn.utils.clip_grad_norm_(self.ac.critic.parameters(), grad_clip_ac or grad_clip),
        }
        return grad_metrics

    def init_state(self, batch_size: int):
        return self.wm.init_state(batch_size)

    def forward(self,
                obs: Dict[str, Tensor],
                in_state: Any,
                ) -> Tuple[D.Distribution, Any, Dict]:
        assert 'action' in obs, 'Observation should contain previous action'
        act_shape = obs['action'].shape
        assert len(act_shape) == 3 and act_shape[0] == 1, f'Expected shape (1,B,A), got {act_shape}'

        # Forward (world model)

        features, out_state = self.wm.forward(obs, in_state)

        # Forward (actor critic)

        feature = features[:, :, 0]  # (T=1,B,I=1,F) => (1,B,F)
        action_distr = self.ac.forward_actor(feature)  # (1,B,A)
        value = self.ac.forward_value(feature)  # (1,B)

        metrics = dict(policy_value=value.detach().mean())
        return action_distr, out_state, metrics

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      imag_horizon: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      do_dream_tensors=False,
                      ):
        assert 'action' in obs, '`action` required in observation'
        assert 'reward' in obs, '`reward` required in observation'
        assert 'reset' in obs, '`reset` required in observation'
        assert 'terminal' in obs, '`terminal` required in observation'
        T, B = obs['action'].shape[:2]
        I, H = iwae_samples, imag_horizon

        # World model

        loss_model, features, states, out_state, metrics, tensors = \
            self.wm.training_step(obs,
                                  in_state,
                                  iwae_samples=iwae_samples,
                                  do_open_loop=do_open_loop,
                                  do_image_pred=do_image_pred)

        # Map probe

        loss_map, metrics_map, tensors_map = self.map_model.training_step(features.detach(), obs)
        metrics.update(**metrics_map)
        tensors.update(**tensors_map)

        # Policy

        # reshuffles states so that each iteration / example seeds a new dream sequence
        if self.wm.is_tssm:
            # we can't reshuffle across the sequence dimension, because states have different shapes
            # depending on context history. We have to iterate
            features_dream = []
            actions_dream = []
            rewards_dream = []
            terminals_dream = []
            for in_state_dream in states:
                (e, a, r, z) = in_state_dream
                T = e.shape[0]
                in_state_dream = (e[:,:,0,:].detach(),
                                  a[:,:,0,:].detach(),
                                  r[:,:,0,:].detach(),
                                  z[:,0,:].detach())
                feat, act, rew, term = \
                    self.dream(in_state_dream, H, self.ac.actor_grad == 'dynamics')  # (H+1,BI,D)
                features_dream.append(feat)
                actions_dream.append(act)
                rewards_dream.append(rew)
                terminals_dream.append(term)
            features_dream = torch.cat(features_dream, dim=1)  # (H+1,TBI,D)
            actions_dream = torch.cat(actions_dream, dim=1)  # (H+1,TBI,A)
            # we can't easily concat distributions, so we just recompute them
            self.wm.requires_grad_(False)  # Prevent dynamics gradiens from affecting world model
            rewards_dream = self.wm.decoder.reward.forward(features_dream)      # (H+1,TBI)
            terminals_dream = self.wm.decoder.terminal.forward(features_dream)  # (H+1,TBI)
            self.wm.requires_grad_(True)
        else:
            in_state_dream: StateB = map_structure(states, lambda x: flatten_batch(x.detach())[0])  # type: ignore  # (T,B,I) => (TBI)
            # Note features_dream includes the starting "real" features at features_dream[0]
            features_dream, actions_dream, rewards_dream, terminals_dream = \
                self.dream(in_state_dream, H, self.ac.actor_grad == 'dynamics')  # (H+1,TBI,D)
        (loss_actor, loss_critic), metrics_ac, tensors_ac = \
            self.ac.training_step(features_dream, actions_dream, rewards_dream, terminals_dream)
        metrics.update(**metrics_ac)
        tensors.update(policy_value=unflatten_batch(tensors_ac['value'][0], (T, B, I)).mean(-1))

        # Dream for a log sample.

        dream_tensors = {}
        if do_dream_tensors:
            with torch.no_grad():  # careful not to invoke modules first time under no_grad (https://github.com/pytorch/pytorch/issues/60164)
                # The reason we don't just take real features_dream is because it's really big (H*T*B*I),
                # and here for inspection purposes we only dream from first step, so it's (H*B).
                # Oh, and we set here H=T-1, so we get (T,B), and the dreamed experience aligns with actual.
                if self.wm.is_tssm:
                    first_state = self.wm.core.first_state(states)
                    # in_state_dream = first_state
                    in_state_dream: StateB = map_structure(first_state, lambda x: x.detach())
                else:
                    in_state_dream: StateB = map_structure(states, lambda x: x.detach()[0, :, 0])  # type: ignore  # (T,B,I) => (B)
                features_dream, actions_dream, rewards_dream, terminals_dream = self.dream(in_state_dream, T - 1)  # H = T-1
                image_dream = self.wm.decoder.image.forward(features_dream)
                _, _, tensors_ac = self.ac.training_step(features_dream, actions_dream, rewards_dream, terminals_dream, log_only=True)
                # The tensors are intentionally named same as in tensors, so the logged npz looks the same for dreamed or not
                dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]),  # first action is real from previous step
                                     reward_pred=rewards_dream.mean,
                                     terminal_pred=terminals_dream.mean,
                                     image_pred=image_dream,
                                     **tensors_ac)
                assert dream_tensors['action_pred'].shape == obs['action'].shape
                assert dream_tensors['image_pred'].shape == obs['image'].shape

        return (loss_model, loss_map, loss_actor, loss_critic), out_state, metrics, tensors, dream_tensors

    def dream(self, in_state: StateB, imag_horizon: int, dynamics_gradients=False):
        features = []
        actions = []
        state = in_state
        self.wm.requires_grad_(False)  # Prevent dynamics gradiens from affecting world model

        for i in range(imag_horizon):
            feature = self.wm.core.state_to_feature(state)
            action_dist = self.ac.forward_actor(feature)
            if dynamics_gradients:
                action = action_dist.rsample()
            else:
                action = action_dist.sample()
            features.append(feature)
            actions.append(action)
            # When using dynamics gradients, this causes gradients in RSSM, which we don't want.
            # This is handled in backprop - the optimizer_model will ignore gradients from loss_actor.
            _, state = self.wm.core.forward_prior(action, None, state)

        feature = self.wm.core.state_to_feature(state)
        features.append(feature)
        features = torch.stack(features)  # (H+1,TBI,D)
        actions = torch.stack(actions)  # (H,TBI,A)

        rewards = self.wm.decoder.reward.forward(features)      # (H+1,TBI)
        terminals = self.wm.decoder.terminal.forward(features)  # (H+1,TBI)

        self.wm.requires_grad_(True)
        return features, actions, rewards, terminals

    def __str__(self):
        # Short representation
        s = []
        s.append(f'Model: {param_count(self)} parameters')
        for submodel in (self.wm.encoder, self.wm.decoder, self.wm.core, self.ac, self.map_model):
            if submodel is not None:
                s.append(f'  {type(submodel).__name__:<15}: {param_count(submodel)} parameters')
        return '\n'.join(s)

    def __repr__(self):
        # Long representation
        return super().__repr__()


class WorldModel(nn.Module):

    def __init__(self, conf):
        super().__init__()

        self.deter_dim = conf.deter_dim
        self.stoch_dim = conf.stoch_dim
        self.stoch_discrete = conf.stoch_discrete
        self.kl_weight = conf.kl_weight
        self.kl_balance = None if conf.kl_balance == 0.5 else conf.kl_balance
        self.is_tssm = conf.tssm

        # Encoder
        self.encoder = MultiEncoder(conf)

        self.features_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        if conf.tssm:
            self.features_dim = self.encoder.out_dim

        # Decoders
        self.decoder = MultiDecoder(self.features_dim, conf)

        # RSSM
        Core = RSSMCore
        if conf.tssm:
            Core = TSSMCore
        self.core = Core(embed_dim=self.encoder.out_dim,
                         action_dim=conf.action_dim,
                         deter_dim=conf.deter_dim,
                         stoch_dim=conf.stoch_dim,
                         stoch_discrete=conf.stoch_discrete,
                         hidden_dim=conf.hidden_dim,
                         gru_layers=conf.gru_layers,
                         gru_type=conf.gru_type,
                         layer_norm=conf.layer_norm)

        # Init
        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size: int) -> Tuple[Any, Any]:
        return self.core.init_state(batch_size)

    def forward(self,
                obs: Dict[str, Tensor],
                in_state: Any
                ):
        loss, features, states, out_state, metrics, tensors = \
            self.training_step(obs, in_state, forward_only=True)
        return features, out_state

    def training_step(self,
                      obs: Dict[str, Tensor],
                      in_state: Any,
                      iwae_samples: int = 1,
                      do_open_loop=False,
                      do_image_pred=False,
                      forward_only=False
                      ):

        # Encoder

        embed = self.encoder(obs)

        # RSSM

        prior, post, post_samples, features, states, out_state = \
            self.core.forward(embed,
                              obs['action'],
                              obs['reset'],
                              in_state,
                              iwae_samples=iwae_samples,
                              do_open_loop=do_open_loop)

        if forward_only:
            return torch.tensor(0.0), features, states, out_state, {}, {}

        # Decoder

        loss_reconstr, metrics, tensors = self.decoder.training_step(features, obs)

        # KL loss

        d = self.core.zdistr
        dprior = d(prior)
        dpost = d(post)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)  # (T,B,I)
        if iwae_samples == 1:
            # Analytic KL loss, standard for VAE
            if not self.kl_balance:
                loss_kl = loss_kl_exact
            else:
                loss_kl_postgrad = D.kl.kl_divergence(dpost, d(prior.detach()))
                loss_kl_priograd = D.kl.kl_divergence(d(post.detach()), dprior)
                loss_kl = (1 - self.kl_balance) * loss_kl_postgrad + self.kl_balance * loss_kl_priograd
        else:
            # Sampled KL loss, for IWAE
            z = post_samples.reshape(dpost.batch_shape + dpost.event_shape)
            loss_kl = dpost.log_prob(z) - dprior.log_prob(z)

        # Total loss

        assert loss_kl.shape == loss_reconstr.shape
        loss_model_tbi = self.kl_weight * loss_kl + loss_reconstr
        loss_model = -logavgexp(-loss_model_tbi, dim=2)

        # Metrics

        with torch.no_grad():
            loss_kl = -logavgexp(-loss_kl_exact, dim=2)  # Log exact KL loss even when using IWAE, it avoids random negative values
            entropy_prior = dprior.entropy().mean(dim=2)
            entropy_post = dpost.entropy().mean(dim=2)
            tensors.update(loss_kl=loss_kl.detach(),
                           entropy_prior=entropy_prior,
                           entropy_post=entropy_post)
            metrics.update(loss_model=loss_model.mean(),
                           loss_kl=loss_kl.mean(),
                           entropy_prior=entropy_prior.mean(),
                           entropy_post=entropy_post.mean())

        # Predictions

        if do_image_pred:
            with torch.no_grad():
                prior_samples = self.core.zdistr(prior).sample().reshape(post_samples.shape)
                deter_states = self.core.states_to_deter(states)
                features_prior = self.core.hz_to_feature(deter_states, prior_samples)
                # Decode from prior
                _, mets, tens = self.decoder.training_step(features_prior, obs, extra_metrics=True)
                metrics_logprob = {k.replace('loss_', 'logprob_'): v for k, v in mets.items() if k.startswith('loss_')}
                tensors_logprob = {k.replace('loss_', 'logprob_'): v for k, v in tens.items() if k.startswith('loss_')}
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                metrics.update(**metrics_logprob)   # logprob_image, ...
                tensors.update(**tensors_logprob)  # logprob_image, ...
                tensors.update(**tensors_pred)  # image_pred, ...

        return loss_model.mean(), features, states, out_state, metrics, tensors
