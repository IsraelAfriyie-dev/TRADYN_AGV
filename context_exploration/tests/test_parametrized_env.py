"""
Tests for parametrized and contextual environments

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import unittest

import numpy as np

from context_exploration.data.envs import ContextUnicycleLandscapeEnv
from context_exploration.data.envs.common import (
    ParametrizedEnvAlreadyInitialized,
    ParametrizedEnvNotInitialized,
)


class TestParametrizedEnv(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def run_seeded_rollout(self, env, seed):
        env.seed(seed)
        env.action_space.seed(seed)
        env.reset()
        done = False
        observations = []
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            observations.append(obs)
        return observations

    def assert_observation_similarity(self, observations_1, observations_2):
        np.testing.assert_allclose(observations_1, observations_2)

    def assert_reproducibility_reseed(self, env_class, env_kwargs, seed):
        env = env_class(**env_kwargs)
        env.initialize_context(42)
        obs_1 = self.run_seeded_rollout(env, seed)
        obs_2 = self.run_seeded_rollout(env, seed)
        self.assert_observation_similarity(obs_1, obs_2)
        env.close()

    def assert_reproducibility_reconstruct(self, env_class, env_kwargs, seed):
        env_1 = env_class(**env_kwargs)
        env_1.initialize_context(42)
        env_2 = env_class(**env_kwargs)
        env_2.initialize_context(42)
        obs_1 = self.run_seeded_rollout(env_1, seed)
        obs_2 = self.run_seeded_rollout(env_2, seed)
        self.assert_observation_similarity(obs_1, obs_2)
        env_1.close()
        env_2.close()

    def assert_constant_context_reset(self, env_class, env_kwargs, seed):
        env = env_class(**env_kwargs)
        env.initialize_context(42)
        env.seed(seed)
        obs_a = env.reset()
        context_a = env.context
        obs_b = env.reset()
        context_b = env.context
        np.testing.assert_allclose(context_a, context_b)
        self.assertFalse(np.allclose(obs_a, obs_b))
        env.close()

    def assert_constant_obs_reset(self, env_class, env_kwargs, seed):
        env = env_class(**env_kwargs)
        env.initialize_context(42)
        env.seed(seed)
        obs_a = env.reset()
        env.seed(seed)
        obs_b = env.reset()
        np.testing.assert_allclose(obs_a, obs_b)
        env.close()

    def assert_different_context_different_dynamics(self, env_class, env_kwargs, seed):
        env_1 = env_class(**env_kwargs)
        env_1.initialize_context(42)
        env_1.seed(seed)
        env_1.action_space.seed(seed)
        obs_1 = env_1.reset()
        env_2 = env_class(**env_kwargs)
        env_2.initialize_context(43)
        env_2.seed(seed)
        env_2.action_space.seed(seed)
        obs_2 = env_2.reset()
        # First observation may actually be different for different contexts
        # E.g., for unicycle, context contains landscape
        # np.testing.assert_allclose(obs_1, obs_2)
        action = env_1.action_space.sample()
        obs_1_next, _, _, _ = env_1.step(action)
        obs_2_next, _, _, _ = env_2.step(action)
        self.assertFalse(np.allclose(obs_1_next, obs_2_next))
        env_1.close()
        env_2.close()

    def assert_fail_reset_unseeded(self, env_class, env_kwargs, seed):
        env = env_class(**env_kwargs)
        self.assertRaises(ParametrizedEnvNotInitialized, env.reset)
        env.close()

    def assert_different_context_reinit(self, env_class, env_kwargs, seed):
        env = env_class(**env_kwargs)
        env.initialize_context(42)
        context_a = env.context
        env.release_context()
        env.initialize_context(43)
        context_b = env.context
        env.release_context()
        if context_a.shape != context_b.shape:
            pass
        else:
            self.assertFalse(np.allclose(context_a, context_b))
        env.close()

    def assert_fail_reinit_no_release(self, env_class, env_kwargs, seed):
        env = env_class(**env_kwargs)
        env.initialize_context(42)
        self.assertRaises(ParametrizedEnvAlreadyInitialized, env.initialize_context, 42)
        env.close()

    def assert_pass_single_env(self, env_class, env_kwargs, seed):
        for test in [
            self.assert_reproducibility_reseed,
            self.assert_reproducibility_reconstruct,
            self.assert_constant_context_reset,
            self.assert_constant_obs_reset,
            self.assert_fail_reset_unseeded,
            self.assert_different_context_reinit,
            self.assert_fail_reinit_no_release,
            self.assert_different_context_different_dynamics,
        ]:
            with self.subTest(msg=f"{env_class.__name__} - {test.__name__}"):
                test(env_class, env_kwargs, seed)

    def test_unicycle(self):
        env_class = ContextUnicycleLandscapeEnv
        env_kwargs_list = [
            dict(is_vary_robot=False, is_vary_terrain=True, reset_split="train"),
            dict(is_vary_robot=False, is_vary_terrain=False, reset_split="train"),
            dict(is_vary_robot=True, is_vary_terrain=True, reset_split="train"),
            dict(is_vary_robot=True, is_vary_terrain=False, reset_split="train"),
        ]
        seed = 42
        with self.subTest(msg=f"{env_class.__name__}"):
            for env_kwargs in env_kwargs_list:
                self.assert_pass_single_env(env_class, env_kwargs, seed)
