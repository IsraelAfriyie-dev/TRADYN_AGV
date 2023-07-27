"""
Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""

from context_exploration.data.envs.unicycle_landscape import ContextUnicycleLandscapeEnv

ENVS = {
    "unicycle_robotvary_terrainpatches": [
        ContextUnicycleLandscapeEnv,
        dict(
            is_vary_robot=True,
            is_vary_terrain=True,
            param_range="default",
        ),
    ]
}


def make_env(env_name, **kwarg_updates):
    kwargs = ENVS[env_name][1]
    for key, val in kwarg_updates.items():
        kwargs[key] = val
    return ENVS[env_name][0](**kwargs)
