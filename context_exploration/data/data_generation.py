"""
Generate training data script

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import os
from pathlib import Path

import gym
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from context_exploration.data.envs import ENVS, make_env
from context_exploration.data.envs.unicycle_landscape import Landscape
from context_exploration.data.wrappers import CaptureWrapper

# This file is intended to be run as a script
__all__ = []


class ConcatEnv(gym.Env):
    """
    To keep rollouts with the same context seed together,
    we simply concatenate their actions and observations.
    """

    def __init__(self, env_1, env_2):
        self.env_1 = env_1
        self.env_2 = env_2

    def reset(self, **kwargs):
        obs_1 = self.env_1.reset()
        obs_2 = self.env_2.reset()
        obs = np.concatenate((obs_1, obs_2), axis=-1)
        return obs

    def step(self, action):
        action_1, action_2 = np.split(action, 2)
        obs_1, reward_1, done_1, _ = self.env_1.step(action_1)
        obs_2, reward_2, done_2, _ = self.env_2.step(action_2)
        if done_1 and done_2:
            done = True
        elif not done_1 and not done_2:
            done = False
        else:
            raise ValueError
        reward = np.array([reward_1, reward_2])
        obs = np.concatenate((obs_1, obs_2), axis=-1)
        info = {}
        return obs, reward, done, info

    def close(self):
        self.env_1.close()
        self.env_2.close()


def generate_rollout(env_name: str, seed: int, experience_dir: Path):
    env_1 = make_env(env_name, reset_split="train")
    env_2 = make_env(env_name, reset_split="train")

    # same context for both environments
    env_1.initialize_context(seed)
    env_2.initialize_context(seed)

    # but different initial state seed
    env_seed_gen = np.random.RandomState(seed)
    env_1_seed = env_seed_gen.randint(0, int(1e8))
    env_2_seed = env_seed_gen.randint(0, int(1e8))
    env_1.seed(env_1_seed)
    env_2.seed(env_2_seed)

    concat_env = ConcatEnv(env_1, env_2)

    capture_env = CaptureWrapper(
        concat_env,
        rollout_uuid=f"seed={seed}",
        save_renderings=False,
        process_rendering_fcn=None,
    )

    capture_env.reset()

    # We have to call this after reset. Before reset(),
    # the action space is a MockSpace due to unset context parameters.
    # We excite the environments with different actions.
    action_iterator_1 = env_1.excitation_controller.get_iterator(env_1_seed)
    action_iterator_2 = env_2.excitation_controller.get_iterator(env_2_seed)

    done = False
    step = 0
    while not done:
        action_1 = next(action_iterator_1)
        action_2 = next(action_iterator_2)
        action = np.concatenate((action_1, action_2), axis=-1)
        obs, reward, done, _ = capture_env.step(action)
        step += 1

    rollout_data = capture_env.close()
    return rollout_data


def generate_single_env(env_name, experience_basedir):
    if "unicycle" in env_name:
        n_env_samples = 10_000 + 5_000
    else:
        raise ValueError

    experience_dir = experience_basedir.joinpath(env_name)
    os.makedirs(experience_dir, exist_ok=True)
    container_file = experience_dir.joinpath(f"np_container_0-{n_env_samples - 1}.npz")
    if os.path.isfile(container_file):
        print(f"Experience for {env_name} already exists")
        return

    jobs = []
    for seed in tqdm(range(n_env_samples)):
        job = delayed(generate_rollout)(env_name, seed, experience_dir)
        jobs.append(job)
    rollout_data_list = Parallel(n_jobs=16)(tqdm(jobs))

    rollout_np = {"action": [], "reward": [], "done": [], "observation": []}
    for item in rollout_data_list:
        for k in rollout_np.keys():
            rollout_np[k].append(item["rollout"][k])

    for k, v in rollout_np.items():
        rollout_np[k] = np.stack(v)

    try:
        np.savez(container_file, **rollout_np)
    except Exception:
        print("Writing file failed - writing to /tmp!")
        experience_dir = Path("/tmp").joinpath(env_name)
        os.makedirs(experience_dir, exist_ok=True)
        container_file = experience_dir.joinpath(
            f"np_container_0-{n_env_samples - 1}.npz"
        )
        np.savez(container_file, **rollout_np)


def generate_data():
    # Cache landscapes
    Landscape.cache_landscapes()

    experience_basedir = (
        Path(__file__).resolve().parent.parent.parent.joinpath("data", "experience")
    )
    print(f"Target directory: {experience_basedir}")
    os.makedirs(experience_basedir, exist_ok=True)

    for env_name in ENVS.keys():
        generate_single_env(env_name, experience_basedir)


if __name__ == "__main__":
    generate_data()
