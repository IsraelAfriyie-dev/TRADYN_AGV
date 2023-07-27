"""
CEM planning algorithm

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import numpy as np
import torch
from colorednoise import powerlaw_psd_gaussian
from numpy.random import normal
from torch import nn
from torch.distributions.distribution import Distribution
from tqdm import tqdm

__all__ = [
    "Belief",
    "InitialCemState",
    "AbstractTransitionModel",
    "AbstractReturnModel",
    "CemTransitionModel",
    "CEM",
]


class Belief(Distribution):
    @property
    def device(self):
        return self.mean.device


class InitialCemState:
    def __init__(self, state):
        self.state = state

    @property
    def device(self):
        return self.state.device

    @property
    def batch_shape(self):
        return self.state.shape[:-1]

    def expand(self, target_shape):
        target_shape = torch.Size(target_shape)
        return InitialCemState(
            self.state[..., None, :].expand(*target_shape, self.state.shape[-1])
        )


class AbstractTransitionModel:
    def __init__(self, action_size):
        self._action_size = action_size

    @property
    def action_size(self):
        return self._action_size

    def multi_step(self, initial_state: "InitialCemState", actions: torch.Tensor):
        """
        Multi-step forward prediction

        Parameters
        ----------
        initial_state: `InitialCemState` with batch shape <bs>
            Initial state
        actions: `torch.Tensor` [T x <bs> x action_size]
            Actions to apply. actions[0] is applied to initial_state

        Returns
        -------
        predicted_states: `Belief` or `torch.Tensor` with batch shape [(T+1) x <bs>]
            Predictions passed to return model
        """
        raise NotImplementedError


class AbstractReturnModel(nn.Module):
    def __init__(self):
        super(AbstractReturnModel, self).__init__()

    def forward(self, predicted_states, actions: torch.Tensor) -> torch.Tensor:
        """
        Return for applying action sequence leading to state beliefs

        Parameters
        ----------
        predicted_states: `Belief` or `torch.Tensor` with batch shape [(T+1) x <bs>]
            Predicted states including initial state
        actions: `torch.Tensor`, [T x <bs> x action_size]
            Applied actions (first action is applied to first state belief)

        Returns
        -------
        returns: torch.Tensor [<bs>, 1]
        """
        raise NotImplementedError


class CemTransitionModel(AbstractTransitionModel):
    def __init__(self, transition_model, context_latent, local_ctx_fcn):
        """
        Transition model for CEM

        Parameters
        ----------
        transition_model : `GruTransitionModel`
        context_latent : `torch.Tensor`, [B x context_dim]
        local_ctx_fcn: Context lookup function, see documentation of `GruTransitionModel`
        """
        self.transition_model = transition_model
        self.context_latent = context_latent
        self.local_ctx_fcn = local_ctx_fcn
        super(CemTransitionModel, self).__init__(transition_model.action_dim)

    def multi_step(self, initial_cem_state, action_sequence):
        """
        Multi-step prediction

        Parameters
        ----------
        initial_cem_state : `InitialCemState`, shape [B x n_candidates x state_dim]
        action_sequence : `torch.Tensor`, [T x B x n_candidates x action_dim]

        Returns
        -------
        prediction: `torch.Tensor`,
            [(T+1) x B x n_candidates x state_dim]
        """
        ctx_mean = self.context_latent.mean
        state = initial_cem_state.state
        ctx_mean = ctx_mean[:, None, :].expand(
            ctx_mean.shape[0], state.shape[1], ctx_mean.shape[-1]
        )
        prediction = self.transition_model.forward_multi_step(
            state, action_sequence, ctx_mean, self.local_ctx_fcn, return_mean_only=True
        )
        return prediction


class CEM(nn.Module):
    def __init__(
        self,
        transition_model,
        return_model,
        planning_horizon,
        action_space,
        optimisation_iters=10,
        candidates=1000,
        top_candidates=100,
        initial_std_factor=1,
        colorednoise_beta=0,
        clip_actions=True,
        return_all_actions=False,
        return_mean=True,
        verbose=False,
    ):
        """
        Cross-entropy-method planning algorithm

        Parameters
        ----------
        transition_model: `AbstractTransitionModel`
            Transition model
        return_model: `AbstractReturnModel`
            Return model
        planning_horizon: int
            Planning horizon
        action_space: `gym.spaces.Box`
            Action space
        optimisation_iters: int
            Number of CEM iterations
        candidates: int
            Number of candidates per iteration
        top_candidates: int
            Number of best candidates to refit belief per iteration
        initial_std_factor: float
            Factor on the initial std (std = factor * action_space.high)
        colorednoise_beta: float
            Beta parameter for colored noise
        clip_actions: bool
            Clip actions to action_space range
        return_all_actions: bool, default False
            Return all actions instead of the first one to apply
        return_mean: bool, default True
            If True, return mean of best actions; if False,
            return best action sequence
        verbose: bool, default False
            Be verbose
        """
        super().__init__()
        self.transition_model, self.return_model = transition_model, return_model
        self.planning_horizon = planning_horizon
        self.action_space = action_space
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates
        self.initial_std_factor = initial_std_factor
        self.colorednoise_beta = colorednoise_beta
        self.clip_actions = clip_actions
        self.return_all_actions = return_all_actions
        self.return_mean = return_mean
        self.verbose = verbose
        self.rng = np.random.RandomState(42)

    def forward(self, initial_state_belief):
        """
        Compute optimal action for current state

        Parameters
        ----------
        initial_state_belief: Belief with batch_shape <bs>
            Distribution of initial state
        """
        action_size = self.transition_model.action_size
        device = initial_state_belief.device
        expanded_state_belief = initial_state_belief.expand(
            initial_state_belief.batch_shape + torch.Size([self.candidates])
        )
        # expanded_state_belief: <bs> x n_candidates x <variable_dim>
        action_range = [
            torch.Tensor(self.action_space.low).to(device),
            torch.Tensor(self.action_space.high).to(device),
        ]
        assert all(-action_range[0] == action_range[1])
        action_belief_shape = [
            self.planning_horizon,
            *initial_state_belief.batch_shape,
            1,
            action_size,
        ]
        action_sample_shape = [
            self.planning_horizon,
            *initial_state_belief.batch_shape,
            self.candidates,
            action_size,
        ]
        action_mean, action_std_dev = (
            torch.zeros(*action_belief_shape, device=device),
            self.initial_std_factor
            * torch.ones(*action_belief_shape, device=device)
            * action_range[1],
        )

        iterable = list(range(self.optimisation_iters))
        if self.verbose:
            iterable = tqdm(iterable)
        for _ in iterable:
            # Evaluate J action sequences from the current belief
            # (over entire sequence at once, batched over particles)
            # Sample actions [T x <bs> x n_candidates x action_dim]
            if self.colorednoise_beta > 0 and self.planning_horizon > 1:
                # 'powerlaw_psd_gaussian' correlates samples on the last dimension
                # -> put time as last dim
                random_samples = powerlaw_psd_gaussian(
                    self.colorednoise_beta,
                    action_sample_shape[1:] + [action_sample_shape[0]],
                    random_state=self.rng,
                )
                random_samples = np.moveaxis(random_samples, -1, 0).astype(np.float32)
            else:
                random_samples = self.rng.randn(*action_sample_shape).astype(np.float32)

            random_samples = torch.from_numpy(random_samples).to(device)

            actions = action_mean + action_std_dev * random_samples
            if self.clip_actions:
                actions = torch.where(
                    actions < action_range[0], action_range[0], actions
                )
                actions = torch.where(
                    actions > action_range[1], action_range[1], actions
                )
            # actions: [T x <bs> x n_candidates x action_dim]

            # Sample next states
            next_state_belief = self.transition_model.multi_step(
                expanded_state_belief, actions
            )
            # next_state_belief: batch_shape [(T+1) x <bs> x n_candidates]

            # Calculate returns
            returns = self.return_model.forward(next_state_belief, actions).unsqueeze(0)
            # returns: [1 x <bs> x n_candidates x 1]
            # Re-fit action belief to the K best action sequences
            # If the best action sequence should be returned, sort
            _, topk = returns.topk(
                self.top_candidates, dim=-2, largest=True, sorted=not self.return_mean
            )
            topk = topk.expand(*actions.shape[:-2], topk.shape[-2], actions.shape[-1])
            best_actions = torch.gather(actions, dim=-2, index=topk)
            # best_actions = [T x <bs> x top_candidates x action_size]

            # Update belief with new means and standard deviations
            action_mean = best_actions.mean(
                dim=-2, keepdim=True
            )  # Mean of all candidates
            # action_mean = T x <bs> x 1 x action_size
            action_std_dev = best_actions.std(dim=-2, unbiased=False, keepdim=True)

        planner_info = {}
        if self.return_mean:
            action_sequence = action_mean[..., 0, :]
        else:
            action_sequence = best_actions[..., 0, :]

        if self.return_all_actions:
            return action_sequence, planner_info
        else:
            return action_sequence[0], planner_info
