"""
Common structures for parametrized and contextual envs

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""

import gym
import numpy as np

__all__ = [
    "ParametrizedEnvNotInitialized",
    "ParametrizedEnvAlreadyInitialized",
    "ParametrizedEnvWrapper",
    "SampleActionMixin",
    "SizeProperties",
]


class ParametrizedEnvNotInitialized(Exception):
    pass


class ParametrizedEnvAlreadyInitialized(Exception):
    pass


class ParametrizedEnvWrapper(gym.Env):
    def __init__(self, context_space):
        self._context_space = context_space
        self.env = None
        self.context = None
        self._last_seed = None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def reward_range(self):
        return self.env.reward_range

    @property
    def metadata(self):
        return self.env.metadata

    @property
    def context_dim(self):
        return self._context_space.shape[0]

    @property
    def context_space(self):
        return self._context_space

    @property
    def unwrapped(self):
        return self.env

    def initialize_context(self, seed, context=None):
        if self.context is not None:
            raise ParametrizedEnvAlreadyInitialized
        if context is not None:
            assert seed is None
        if context is None:
            self._context_space.seed(seed)
            context = self._context_space.sample()
        self.env = self._construct_env(context)
        self.context = context
        self._last_seed = seed

    def _construct_env(self, context):
        raise NotImplementedError

    def release_context(self):
        self.context = None

    def _assert_initialized(self):
        if self.context is None:
            raise ParametrizedEnvNotInitialized

    def seed(self, seed=None):
        self._assert_initialized()
        return self.env.seed(seed)

    def reset(self, **kwargs):
        self._assert_initialized()
        return self.env.reset(**kwargs)

    def step(self, action):
        self._assert_initialized()
        return self.env.step(action)

    def render(self, mode="human"):
        self._assert_initialized()
        return self.env.render(mode)

    def close(self):
        self._assert_initialized()
        return self.env.close()

    def is_transition_informative(self, x, u, x_next):
        raise NotImplementedError


class SampleActionMixin:
    def sample_action(self, *shape):
        if shape:
            action = np.stack(
                [self.action_space.sample() for _ in range(np.prod(np.array(shape)))]
            )
            action = action.reshape(*shape, *self.action_space.shape)
        else:
            action = self.action_space.sample()
        return action


class SizeProperties:
    def __init__(self, state_dim, action_dim):
        self._state_dim = state_dim
        self._action_dim = action_dim

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim
