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
USELOCALCTX = [True, False]
CALIBRATE_ROBOT = [True, False]
SEED = [1, 2, 3, 4, 5]
EVAL_SEED = [1, 2, 3, 4, 5]

commands = []
for (
    environment_name,
    uselocalctx,
    calibrate_robot,
    seed,
    eval_seed,
) in itertools.product(
    ENV_NAMES, USELOCALCTX, CALIBRATE_ROBOT, SEED, EVAL_SEED
):

    run_id = (
        f"model_env={environment_name}_"
        f"uselocalctx={uselocalctx}_"
        f"seed={seed}"
    )

    command = (
        f"python -m context_exploration.evaluation.unicycle.unicycle_planning "
        f"{run_id} {calibrate_robot} {eval_seed} \n"
    )
    commands.append(command)

# write commands to job file
with open(f"jobs_unicycle_eval_planning.txt", "w") as handle:
    handle.writelines(commands)
