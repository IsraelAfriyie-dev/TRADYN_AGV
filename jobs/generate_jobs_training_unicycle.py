"""
Copyright 2021 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import itertools

ENV_NAMES = [
    "unicycle_robotvary_terrainpatches",
]
USE_ENV_LOCAL_CONTEXT = [True, False]
SEED = [1, 2, 3, 4, 5]

commands_init = []
commands_restart = []
for (
    env_name,
    use_env_local_context,
    seed,
) in itertools.product(ENV_NAMES, USE_ENV_LOCAL_CONTEXT, SEED):

    run_id = f"model_env={env_name}_uselocalctx={use_env_local_context}_seed={seed}"

    command_init = (
        f"python -m context_exploration.train_model "
        f"--id={run_id} "
        f"with unicycle_cfg "
        f"env_name={env_name} "
        f"use_env_local_context={use_env_local_context} "
        f"seed={seed}\n"
    )
    commands_init.append(command_init)

    command_restart = (
        f"python -m context_exploration.train_model restart_base={run_id}\n"
    )
    commands_restart.append(command_restart)

# write commands to job file
with open(f"jobs_training_unicycle_init.txt", "w") as handle:
    handle.writelines(commands_init)

with open(f"jobs_training_unicycle_restart.txt", "w") as handle:
    handle.writelines(commands_restart)
