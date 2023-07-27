"""
Common structures for unicycle evaluation

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import numpy as np
import torch

from context_exploration.data.envs import make_env
from context_exploration.model.context_encoder import ContextSet


class VectorizedExcitationIterator:
    def __init__(self, iterator_list):
        self._iterator_list = iterator_list

    def __next__(self):
        actions = [next(it) for it in self._iterator_list]
        return np.stack(actions)


class DummyVecEnv:
    def __init__(self, env_name, kwarg_updates, n_instances):
        self.instances = [
            make_env(env_name, **kwarg_updates) for _ in range(n_instances)
        ]

    def _zip(self, other):
        assert len(other) == len(self.instances)
        for env, o in zip(self.instances, other):
            yield env, o

    @property
    def action_space(self):
        return self.instances[0].action_space

    def reset(self, init_robot_state):
        obs_arr = []
        for env, i_init_robot_state in self._zip(init_robot_state):
            obs = env.reset(init_robot_state=i_init_robot_state)
            obs_arr.append(obs)
        return np.stack(obs_arr)

    def seed(self, seed):
        for env, i_seed in self._zip(seed):
            env.seed(i_seed)

    def initialize_context(self, seed=None, context=None):
        if context is not None:
            assert seed is None
            for env, c in self._zip(context):
                env.initialize_context(seed=None, context=c)
        else:
            assert seed is not None
            for env, i_seed in self._zip(seed):
                env.initialize_context(i_seed)

    def release_context(self):
        for env in self.instances:
            env.release_context()

    def step(self, action):
        obs_arr = []
        for env, i_action in self._zip(action):
            obs, _, _, _ = env.step(i_action)
            obs_arr.append(obs)
        return np.stack(obs_arr)

    def get_excitation_iterator(self, seed):
        iterator_list = []
        for env, i_seed in self._zip(seed):
            iterator = env.excitation_controller.get_iterator(i_seed)
            iterator_list.append(iterator)
        return VectorizedExcitationIterator(iterator_list)

    def query_terrain_state(self, robot_obs):
        terrain_states = []
        for env, i_robot_obs in self._zip(robot_obs):
            terrain_states.append(env.query_terrain_state(i_robot_obs))

        terrain_states = torch.stack(terrain_states)
        return terrain_states


def collect_random_traj(
    vec_env, env_context, env_seed, action_seed, initial_state, n_transitions
):
    vec_env.initialize_context(**env_context)
    vec_env.seed(env_seed)
    obs = vec_env.reset(init_robot_state=initial_state)
    action_iterator = vec_env.get_excitation_iterator(action_seed)
    obs_list = [
        obs,
    ]
    action_list = []
    for _ in range(n_transitions):
        action = next(action_iterator)
        action_list.append(action)
        obs = vec_env.step(action)
        obs_list.append(obs)
    vec_env.release_context()
    obs_arr = np.stack(obs_list)
    action_arr = np.stack(action_list)
    return obs_arr, action_arr


def calibrate_batch(
    vec_env,
    context_encoder,
    env_context,
    env_seed,
    action_seed,
    initial_state,
    n_calib_transitions,
    return_obs_arr=False,
):
    device = context_encoder.device
    obs_arr, action_arr = collect_random_traj(
        vec_env, env_context, env_seed, action_seed, initial_state, n_calib_transitions
    )
    context_set = (
        ContextSet.from_array(
            obs_arr[:-1],
            action_arr,
            obs_arr[1:],
        )
        .as_torch()
        .to(device)
    )
    context_latent = context_encoder.forward_tensor(
        context_set.x, context_set.u, context_set.x_next
    )
    if return_obs_arr:
        return context_latent, obs_arr, action_arr
    else:
        return context_latent


def generate_eval_setting(evaluation_idx):
    state_low = np.array([0.1, 0.1, 0, -np.pi])
    state_high = np.array([0.9, 0.9, 0, np.pi])
    rng = np.random.RandomState(evaluation_idx)
    context_seed = rng.randint(0, int(1e8))
    env_seed = rng.randint(0, int(1e8))
    action_seed = env_seed
    initial_state = state_low + rng.rand(*state_low.shape) * (state_high - state_low)
    target_state = state_low + rng.rand(*state_low.shape) * (state_high - state_low)
    return {
        "evaluation_idx": evaluation_idx,
        "context_seed": context_seed,
        "env_seed": env_seed,
        "action_seed": action_seed,
        "initial_state": initial_state,
        "target_state": target_state,
    }


def generate_eval_setting_batch(evaluation_idx_arr):
    setting_list = []
    for evaluation_idx in evaluation_idx_arr:
        setting_list.append(generate_eval_setting(evaluation_idx))

    keys = setting_list[0].keys()
    setting_batch = {key: np.stack([s[key] for s in setting_list]) for key in keys}
    return setting_batch
