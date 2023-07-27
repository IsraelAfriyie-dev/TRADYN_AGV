"""
Unicycle planning evaluation

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from context_exploration.cem.cem import (
    CEM,
    AbstractReturnModel,
    CemTransitionModel,
    InitialCemState,
)
from context_exploration.evaluation.unicycle.common import (
    DummyVecEnv,
    calibrate_batch,
    generate_eval_setting_batch,
)
from context_exploration.model.loader import get_run_directory, load_model


class UnicycleCemReturnModel(AbstractReturnModel):
    def __init__(self, target_state, dist_cost, term_cost, ctrl_cost):
        """
        Return model for CEM
        """
        super(UnicycleCemReturnModel, self).__init__()
        # target_state is [x, y, v, theta]
        self.target_state = target_state
        self.dist_cost = dist_cost
        self.term_cost = term_cost
        self.ctrl_cost = ctrl_cost

    def forward(self, predicted_states, actions):
        """
        Compute 'return' for unicycle navigation task

        Parameters
        ----------
        predicted_states: torch.Tensor,
            [(T+1) x B x n_candidates x state_dim]
            Includes initial state
        actions: torch.Tensor,
            [T x B x n_candidates x state_dim]

        Returns
        -------
        return_: torch.Tensor, [B x n_candidates x 1]
        """
        # predicted_states is [x, y, v_norm, cos(t), sin(t), f1, f2, f3]
        abserr_x = torch.abs(
            (predicted_states[..., 0] - self.target_state[..., None, 0])
        )
        abserr_y = torch.abs(
            (predicted_states[..., 1] - self.target_state[..., None, 1])
        )
        cost_u = actions[..., 0] ** 2  # steering is for free

        dist = abserr_x ** 2 + abserr_y ** 2
        # Terminal cost
        dist_term = dist[-1]
        # Normalize distances
        dist_term = dist_term - dist_term.mean(dim=-1, keepdims=True)
        dist_term = dist_term / dist_term.std(dim=-1, keepdims=True)

        assert self.dist_cost == 0

        total_cost = (self.ctrl_cost * cost_u.sum(0) + self.term_cost * dist_term)[
            ..., None
        ]
        return_ = -1 * total_cost
        return return_


def plan_batch(
    vec_env,
    context,
    env_seed,
    transition_model,
    context_latent,
    initial_robot_state,
    target_robot_state,
    planning_parameters,
):
    """
    Solve a batch of planning problems
    """
    device = transition_model.device
    if transition_model.local_context_dim > 0:
        local_ctx_fcn = vec_env.query_terrain_state
    else:
        local_ctx_fcn = None

    target_robot_state = torch.Tensor(target_robot_state).to(device)
    cem_return_model = UnicycleCemReturnModel(
        target_robot_state,
        dist_cost=planning_parameters["dist_cost"],
        term_cost=planning_parameters["term_cost"],
        ctrl_cost=planning_parameters["ctrl_cost"],
    )

    vec_env.initialize_context(**context)
    vec_env.seed(env_seed)

    cem_transition_model = CemTransitionModel(
        transition_model, context_latent, local_ctx_fcn
    )

    cem = CEM(
        cem_transition_model,
        cem_return_model,
        planning_horizon=planning_parameters["planning_horizon"],
        action_space=vec_env.action_space,
        initial_std_factor=planning_parameters["initial_std_factor"],
        colorednoise_beta=planning_parameters["colorednoise_beta"],
        candidates=1000,
        top_candidates=50,
        optimisation_iters=20,
    )

    obs = vec_env.reset(init_robot_state=initial_robot_state)
    obs_arr = [
        obs,
    ]
    action_arr = []
    with torch.no_grad():
        for step in tqdm(range(planning_parameters["n_steps"])):
            initial_state = torch.Tensor(obs).to(device)
            if transition_model.local_context_dim > 0:
                initial_state = initial_state[
                    ..., : -transition_model.local_context_dim
                ]
            initial_cem_state = InitialCemState(initial_state)
            cem.planning_horizon = min(
                planning_parameters["planning_horizon"],
                planning_parameters["n_steps"] - step,
            )
            best_action, _ = cem(initial_cem_state)
            best_action = best_action.cpu().numpy()
            action_arr.append(best_action)
            obs = vec_env.step(best_action)
            obs_arr.append(obs)

    obs_arr = np.stack(obs_arr)
    action_arr = np.stack(action_arr)
    vec_env.release_context()
    return obs_arr, action_arr


def evaluate_plan_batch(target_state, obs_arr, action_arr):
    """
    Evaluate batch of planning tasks

    Parameters
    ----------
    target_state: np.ndarray, shape [B x target_dim]
    obs_arr: np.ndarray, shape [(T+1) x B x obs_dim]
    action_arr: np.ndarray, shape [(T+1) x B x action_dim]

    Returns
    -------
    batch_evals: List[Dict]
        thr_energy_all:
            Sum of squared throttle controls (full trajectory)
        success:
            Any state is closer than 0.02 (eucl. distance) to target
        n_steps:
            Number of steps taken until 'success'
        thr_energy_success
            Sum of squared throttle controls (until close to goal)
    """
    obs_xy = obs_arr[..., :2]
    target_xy = target_state[..., :2]
    distance = np.sqrt(((obs_xy - target_xy[None, :, :]) ** 2).sum(axis=-1))
    batch_eval = []
    dist_thres = 0.05
    for batch_idx in range(distance.shape[1]):
        dist_item = distance[:, batch_idx]
        ctrl_item = action_arr[:, batch_idx, :]
        success = np.any(dist_item <= dist_thres)
        thr_energy = ctrl_item[:, 0] ** 2
        thr_energy_all = thr_energy.sum()
        if success:
            n_steps = np.min(np.where(dist_item <= dist_thres)[0])
            thr_energy_success = thr_energy[:n_steps].sum()
        else:
            n_steps = None
            thr_energy_success = None
        batch_eval.append(
            {
                # Closest seen distance to target
                "closest_dist": np.min(dist_item),
                # Array of distances to target
                "dist": dist_item,
                # Any distance is closer than threshold to target
                "success": success,
                # Number of steps until success
                "n_steps": n_steps,
                # Throttle energy for all controls
                "thr_energy_all": thr_energy_all,
                # Throttle energy of controls until success
                "thr_energy_success": thr_energy_success,
            }
        )
    return batch_eval


def run_batch_planning_evaluations(
    run_id, calibrate_robot, planning_parameters, evaluation_seed, reset_split
):
    """
    We compare the following settings:

    Model trained on (given by run_id):
        Patchy terrain (always)
        Varying robot / Fixed robot
        With context / without context
        Terrain-lookup / Terrain-as-observation
    Model evaluated with:
        Calibrated robot
        Uncalibrated robot (empty context set)
    """

    step = "100000_best"
    device = "cuda"
    kwarg_updates = {
        "reset_split": reset_split,
        "max_duration": planning_parameters["n_steps"],
    }
    n_settings_per_evaluation = 30
    n_calib_transitions = 10
    env_name, transition_model, context_encoder = load_model(run_id, step, device)
    vec_env = DummyVecEnv(env_name, kwarg_updates, n_settings_per_evaluation)

    eval_idx_array = np.arange(
        evaluation_seed * n_settings_per_evaluation,
        (evaluation_seed + 1) * n_settings_per_evaluation,
    )

    eval_setting_batch = generate_eval_setting_batch(eval_idx_array)

    if calibrate_robot:
        env_context = {"seed": eval_setting_batch["context_seed"]}
        context_latent = calibrate_batch(
            vec_env,
            context_encoder,
            env_context,
            eval_setting_batch["env_seed"],
            eval_setting_batch["action_seed"],
            eval_setting_batch["initial_state"],
            n_calib_transitions,
        )
    else:
        context_latent = context_encoder.empty_set_context(
            batch_dim=n_settings_per_evaluation
        )

    context = {"seed": eval_setting_batch["context_seed"]}

    obs_arr, action_arr = plan_batch(
        vec_env,
        context,
        eval_setting_batch["env_seed"],
        transition_model,
        context_latent,
        eval_setting_batch["initial_state"],
        eval_setting_batch["target_state"],
        planning_parameters,
    )

    batch_eval = evaluate_plan_batch(
        eval_setting_batch["target_state"], obs_arr, action_arr
    )
    for idx, item in enumerate(batch_eval):
        item["run_id"] = run_id
        item["step"] = step
        item["reset_split"] = reset_split
        item["calibrate_robot"] = calibrate_robot
        item["n_calib_transitions"] = n_calib_transitions
        item["obs_arr"] = obs_arr[:, idx, :]
        item["action_arr"] = action_arr[:, idx, :]
        for key in eval_setting_batch.keys():
            item[key] = eval_setting_batch[key][idx]
        item["planning_parameters"] = planning_parameters

    df = pd.DataFrame(batch_eval)
    return df


def run_single_planning_evaluation(
    planning_parameters,
    base_eval_setting,
    manual_context=None,
    manual_target=None,
    display=True,
):
    run_id = "model_env=unicycle_robotvary_terrainpatches_uselocalctx=True_seed=1"
    step = "100000_best"
    device = "cuda"
    reset_split = "test"
    calibrate_robot = True

    kwarg_updates = {
        "reset_split": reset_split,
        "max_duration": planning_parameters["n_steps"],
    }
    env_name, transition_model, context_encoder = load_model(run_id, step, device)

    eval_setting_batch = generate_eval_setting_batch([base_eval_setting])

    n_instances = 1
    n_calib_transitions = 10

    vec_env = DummyVecEnv(env_name, kwarg_updates, n_instances)

    if manual_context is None:
        env_context = {"seed": eval_setting_batch["context_seed"]}
    else:
        env_context = {"seed": None, "context": manual_context[None, :]}

    env_context_single = {
        k: (None if v is None else v[0]) for k, v in env_context.items()
    }

    if calibrate_robot:
        context_latent = calibrate_batch(
            vec_env,
            context_encoder,
            env_context,
            eval_setting_batch["env_seed"],
            eval_setting_batch["action_seed"],
            eval_setting_batch["initial_state"],
            n_calib_transitions,
        )
    else:
        context_latent = context_encoder.empty_set_context(batch_dim=1)

    if manual_target is not None:
        eval_setting_batch["target_state"] = manual_target[None, :]

    obs_arr, action_arr = plan_batch(
        vec_env,
        env_context,
        eval_setting_batch["env_seed"],
        transition_model,
        context_latent,
        eval_setting_batch["initial_state"],
        eval_setting_batch["target_state"],
        planning_parameters,
    )

    batch_eval = evaluate_plan_batch(
        eval_setting_batch["target_state"], obs_arr, action_arr
    )
    vec_env.initialize_context(**env_context)
    for idx, item in enumerate(batch_eval):
        item["run_id"] = run_id
        item["step"] = step
        item["reset_split"] = reset_split
        item["calibrate_robot"] = calibrate_robot
        item["context"] = vec_env.instances[idx].context
        item["obs_arr"] = obs_arr[:, idx, :]
        item["action_arr"] = action_arr[:, idx, :]
        for key in eval_setting_batch.keys():
            item[key] = eval_setting_batch[key][idx]
        item["planning_parameters"] = planning_parameters
    vec_env.release_context()

    if display:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=1, ncols=1)
        env = vec_env.instances[0]
        env.initialize_context(**env_context_single)
        env.plot_friction(ax)
        for instance_idx in range(1):
            ax.plot(obs_arr[:, instance_idx, 0], obs_arr[:, instance_idx, 1])
            ax.scatter(obs_arr[:, instance_idx, 0], obs_arr[:, instance_idx, 1])
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    return batch_eval


def get_eval_filename(run_id, calibrate_robot, evaluation_seed):
    run_dir = get_run_directory(run_id)
    candidate_file = run_dir.joinpath(
        "eval_plan_calib{}_seed{}.pkl".format(calibrate_robot, evaluation_seed)
    )
    return candidate_file


def generate_teaser_rollouts(planning_parameters):
    base_eval_seed = 112
    contexts = {
        "heavy": np.array([4, 750, 3 * np.pi / 16, 65]),
        "light": np.array([1, 750, 3 * np.pi / 16, 65]),
    }
    targets = {
        "right": np.array([0.75, 0.14384459, 0, 0]),
        "left": np.array([0.5, 0.3, 0, 0]),
    }

    item_list = []
    for context_name, context in contexts.items():
        for target_name, target in targets.items():
            item = run_single_planning_evaluation(
                planning_parameters,
                base_eval_setting=base_eval_seed,
                manual_context=context,
                manual_target=target,
                display=False,
            )[0]
            item["evaluation_idx"] = context_name + "_" + target_name
            item_list.append(item)
    df = pd.DataFrame(item_list)
    df.to_pickle("unicycle_teaser_data.pkl")


def main(argv):
    """
    Calling schemes:

    unicycle_planning.py (no arguments)
        Run a single experiment and show plot
    unicycle_planning.py teaser
        Generate rollouts for teaser figure in paper
    unicycle_planning.py <RUN_ID> <CALIBRATE?> <EVAL_IDX>
        Evaluate a particular run (for batch evaluation)
    """

    n_steps = 50
    term_cost = 1
    ctrl_cost = 0.5

    planning_parameters = {
        "n_steps": n_steps,
        "dist_cost": 0,
        "term_cost": term_cost,
        "ctrl_cost": ctrl_cost,
        "planning_horizon": 30,
        "initial_std_factor": 0.5,
        "colorednoise_beta": 0.5,
    }

    if len(argv) == 1:
        print(run_single_planning_evaluation(planning_parameters, base_eval_setting=5))
    elif len(argv) == 2 and argv[1] == "teaser":
        generate_teaser_rollouts(planning_parameters)
    elif len(argv) == 4:
        run_id = str(argv[1])
        calibrate_robot = {"True": True, "False": False}[argv[2]]
        evaluation_seed = int(argv[3])
        reset_split = "test"
        df = run_batch_planning_evaluations(
            run_id=run_id,
            calibrate_robot=calibrate_robot,
            planning_parameters=planning_parameters,
            evaluation_seed=evaluation_seed,
            reset_split=reset_split,
        )
        eval_filename = get_eval_filename(run_id, calibrate_robot, evaluation_seed)
        df.to_pickle(eval_filename)

    else:
        raise ValueError


if __name__ == "__main__":
    main(sys.argv)
