"""
Sample (correlated) action sequences

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
from copy import deepcopy

import numpy as np
from colorednoise import powerlaw_psd_gaussian

__all__ = [
    "RandomSampleExcitationIterator",
    "RandomSampleExcitationController",
    "GaussianCorrelatedExcitationIterator",
    "GaussianCorrelatedExcitationController",
]


class RandomSampleExcitationIterator:
    def __init__(self, env, seed):
        assert seed is not None
        self._env = env
        self._action_space = deepcopy(env.action_space)
        self._action_space.seed(seed)

    def __next__(self):
        return self._action_space.sample()


class RandomSampleExcitationController:
    def __init__(self, env):
        super(RandomSampleExcitationController, self).__init__()
        self._env = env

    def get_iterator(self, excitation_seed):
        return RandomSampleExcitationIterator(env=self._env, seed=excitation_seed)


class GaussianCorrelatedExcitationIterator:
    def __init__(self, env, mean, std, beta, seed):
        assert seed is not None
        self._env = env
        self._mean = mean
        self._std = std
        self._beta = beta
        self._action_space = deepcopy(env.action_space)
        self._rng = np.random.RandomState(seed)
        max_samples = 1000
        self._idx = 0
        self._noise = powerlaw_psd_gaussian(
            beta, (self._action_space.shape[0], max_samples), random_state=self._rng
        )

    def _clip(self, val):
        return np.clip(val, self._action_space.low, self._action_space.high)

    def __next__(self):
        action = self._mean + self._std * self._noise[:, self._idx]
        action = self._clip(action)
        self._idx += 1
        return action


class GaussianCorrelatedExcitationController:
    def __init__(self, env, mean, std, beta):
        super(GaussianCorrelatedExcitationController, self).__init__()
        self._env = env
        self._mean = mean
        self._std = std
        self._beta = beta

    def get_iterator(self, excitation_seed):
        return GaussianCorrelatedExcitationIterator(
            env=self._env,
            mean=self._mean,
            std=self._std,
            beta=self._beta,
            seed=excitation_seed,
        )


class MockEnv:
    pass


if __name__ == "__main__":
    from gym.spaces import Box

    mock_env = MockEnv()
    mock_env.action_space = Box(low=-np.ones(1), high=np.ones(1))

    controller = GaussianCorrelatedExcitationController(
        mock_env, mean=np.zeros(1), std=0.5 * np.ones(1), beta=1.0
    )

    n_action_trajs = 10
    n_steps = 100

    all_actions = np.zeros((n_action_trajs, n_steps))

    for traj_idx in range(n_action_trajs):
        it = controller.get_iterator(traj_idx)

        actions = np.zeros(n_steps)
        for step in range(n_steps):
            all_actions[traj_idx, step] = next(it)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)

    for traj_idx in range(n_action_trajs):
        ax.plot(all_actions[traj_idx])

    plt.show()
