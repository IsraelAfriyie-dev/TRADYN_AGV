"""
Unicycle prediction evaluation

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

from context_exploration.evaluation.unicycle.common import (
    DummyVecEnv,
    calibrate_batch,
    collect_random_traj,
    generate_eval_setting_batch,
)
from context_exploration.model.loader import get_run_directory, load_model


def predict_batch(
    vec_env,
    context_seed,
    env_seed,
    action_seed,
    transition_model,
    context_latent,
    initial_state,
    n_transitions,
):
    """
    Compute prediction statistics for batch
    """
    if transition_model.local_context_dim > 0:
        local_ctx_fcn = vec_env.query_terrain_state
    else:
        local_ctx_fcn = None

    # Actual trajectory
    env_context = {"seed": context_seed}
    gt_obs_arr, action_arr = collect_random_traj(
        vec_env, env_context, env_seed, action_seed, initial_state, n_transitions
    )
    if transition_model.local_context_dim > 0:
        gt_obs_arr = gt_obs_arr[..., : -transition_model.local_context_dim]

    # Prediction
    x = torch.from_numpy(gt_obs_arr[0]).to(transition_model.device).float()
    action_arr = torch.from_numpy(action_arr).to(transition_model.device).float()
    pred_arr = transition_model.forward_multi_step(
        x,
        action_arr,
        context_latent.mean,
        local_ctx_fcn,
        return_mean_only=True,
        probe_dict=None,
    )
    pred_arr = pred_arr.cpu().detach().numpy()
    return gt_obs_arr, action_arr, pred_arr


def evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr):
    """
    Evaluate batch of prediction tasks

    Parameters
    ----------
    gt_obs_arr: np.ndarray, shape [(T+1) x B x target_dim]
    action_arr: np.ndarray, shape [T x B x obs_dim]
    pred_arr: np.ndarray, shape [(T+1) x B x action_dim]

    Returns
    -------
    batch_evals: List[Dict]
        l2_pos: Euclidean distance between positions
        l2_vel: Euclidean distance between velocities
        abs_angle: Absolute angular error
    """
    # x, y, v, cos(th), sin(th), <terrain>
    l2_norm = lambda x: np.sqrt((x ** 2).sum(axis=-1))

    batch_eval = []
    for batch_idx in range(gt_obs_arr.shape[1]):
        l2_pos = l2_norm(gt_obs_arr[:, batch_idx, :2] - pred_arr[:, batch_idx, :2])
        l2_vel = l2_norm(gt_obs_arr[:, batch_idx, 2:3] - pred_arr[:, batch_idx, 2:3])
        gt_theta = np.arctan2(gt_obs_arr[:, batch_idx, 4], gt_obs_arr[:, batch_idx, 3])
        pred_theta = np.arctan2(pred_arr[:, batch_idx, 4], pred_arr[:, batch_idx, 3])
        diff_theta = pred_theta - gt_theta
        # Normalize angle difference
        # https://stackoverflow.com/a/2007279
        angle_diff = np.arctan2(np.sin(diff_theta), np.cos(diff_theta))
        batch_eval.append(
            {
                "l2_pos": l2_pos,
                "l2_vel": l2_vel,
                "angle_diff": angle_diff,
            }
        )
    return batch_eval


def run_batch_prediction_evaluations(run_id, calibrate_robot, evaluation_seed):
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
    reset_split = "test"
    n_pred_transitions = 150
    kwarg_updates = {
        "reset_split": reset_split,
        "max_duration": n_pred_transitions,
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

    context_seed = eval_setting_batch["context_seed"]
    env_seed = eval_setting_batch["env_seed"]
    calib_action_seed = env_seed
    pred_action_seed = calib_action_seed + 1
    initial_state = eval_setting_batch["initial_state"]

    if calibrate_robot:
        env_context = {"seed": context_seed}
        context_latent, calib_obs, calib_act = calibrate_batch(
            vec_env,
            context_encoder,
            env_context,
            env_seed,
            calib_action_seed,
            initial_state,
            n_calib_transitions,
            return_obs_arr=True,
        )
    else:
        context_latent = context_encoder.empty_set_context(
            batch_dim=n_settings_per_evaluation
        )

    gt_obs_arr, action_arr, pred_arr = predict_batch(
        vec_env,
        context_seed,
        env_seed,
        pred_action_seed,
        transition_model,
        context_latent,
        initial_state,
        n_pred_transitions,
    )

    batch_eval = evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr)
    for idx, item in enumerate(batch_eval):
        item["run_id"] = run_id
        item["step"] = step
        item["reset_split"] = reset_split
        item["calibrate_robot"] = calibrate_robot
        item["n_calib_transitions"] = n_calib_transitions
        item["obs_arr"] = gt_obs_arr[:, idx, :]
        item["action_arr"] = action_arr[:, idx, :]
        item["pred_arr"] = pred_arr[:, idx, :]
        item["context_seed"] = context_seed[idx]
        item["env_seed"] = env_seed[idx]
        item["pred_action_seed"] = pred_action_seed[idx]
        item["initial_state"] = initial_state[idx]
        if calibrate_robot:
            item["calib_obs"] = calib_obs[:, idx, :]
            item["calib_act"] = calib_act[:, idx, :]

    df = pd.DataFrame(batch_eval)
    return df


def run_single_prediction_evaluation():
    run_id = "model_env=unicycle_robotvary_terrainpatches_uselocalctx=True_seed=1"
    step = "100000_best"
    device = "cuda"
    n_pred_transitions = 150
    calibrate_robot = True

    kwarg_updates = {
        "reset_split": "train",
        "max_duration": n_pred_transitions,
    }
    env_name, transition_model, context_encoder = load_model(run_id, step, device)

    context_seed = np.array([2215104])
    env_seed = np.array([60969750])
    calib_action_seed = env_seed
    pred_action_seed = calib_action_seed + 1

    initial_state = np.array([0.5, 0.5, 0, 1.83])[None, :]
    n_instances = 1
    n_calib_transitions = 50

    vec_env = DummyVecEnv(env_name, kwarg_updates, n_instances)
    env_context = {"seed": context_seed}
    if calibrate_robot:
        context_latent, calib_obs, calib_act = calibrate_batch(
            vec_env,
            context_encoder,
            env_context,
            env_seed,
            calib_action_seed,
            initial_state,
            n_calib_transitions,
            return_obs_arr=True,
        )
    else:
        context_latent = context_encoder.empty_set_context(batch_dim=1)
        calib_obs, calib_act = None, None

    gt_obs_arr, action_arr, pred_arr = predict_batch(
        vec_env,
        context_seed,
        env_seed,
        pred_action_seed,
        transition_model,
        context_latent,
        initial_state,
        n_pred_transitions,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)
    env = vec_env.instances[0]
    env.initialize_context(context_seed[0])
    env.plot_landscape(ax)
    for instance_idx in range(1):
        ax.plot(gt_obs_arr[:, instance_idx, 0], gt_obs_arr[:, instance_idx, 1])
        ax.scatter(gt_obs_arr[:, instance_idx, 0], gt_obs_arr[:, instance_idx, 1])
        ax.plot(pred_arr[:, instance_idx, 0], pred_arr[:, instance_idx, 1])
        ax.scatter(pred_arr[:, instance_idx, 0], pred_arr[:, instance_idx, 1])
        if calib_obs is not None:
            ax.plot(calib_obs[:, instance_idx, 0], calib_obs[:, instance_idx, 1])
            ax.scatter(calib_obs[:, instance_idx, 0], calib_obs[:, instance_idx, 1])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


def get_eval_filename(run_id, calibrate_robot, evaluation_seed):
    run_dir = get_run_directory(run_id)
    candidate_file = run_dir.joinpath(
        "eval_pred_calib{}_seed{}.pkl".format(calibrate_robot, evaluation_seed)
    )
    return candidate_file


def main(argv):
    """
    Calling schemes:

    unicycle_prediction.py (no arguments)
        Run a single experiment and show plot
    unicycle_planning.py <RUN_ID> <CALIBRATE?> <EVAL_IDX>
        Evaluate a particular run (for evaluation on the cluster)
    """
    if len(argv) == 1:
        run_single_prediction_evaluation()
    else:
        run_id = str(argv[1])
        calibrate_robot = {"True": True, "False": False}[argv[2]]
        evaluation_seed = int(argv[3])
        df = run_batch_prediction_evaluations(
            run_id=run_id,
            calibrate_robot=calibrate_robot,
            evaluation_seed=evaluation_seed,
        )
        eval_filename = get_eval_filename(run_id, calibrate_robot, evaluation_seed)
        df.to_pickle(eval_filename)


if __name__ == "__main__":
    main(sys.argv)
