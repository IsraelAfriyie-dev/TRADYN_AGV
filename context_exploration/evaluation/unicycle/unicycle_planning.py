# Step 2: Modifications to unicycle_planning.py
# These are the EXACT changes to make

# 1. ADD this import at the top (around line 18, after the other imports)
# Add this import at line ~18 in unicycle_planning.py
from .FailureRecognizer import FailureRecognizer, FailureType

# 2. MODIFY the plan_batch function 
# FIND this function (around line 66) and REPLACE the execution loop

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
    Solve a batch of planning problems - ENHANCED with failure recognition
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
    obs_arr = [obs]
    action_arr = []
    
    # ADD: Initialize failure recognizer for each batch item
    failure_recognizers = [FailureRecognizer() for _ in range(obs.shape[0])]
    failure_detected = [False] * obs.shape[0]
    failure_details = [[] for _ in range(obs.shape[0])]

    with torch.no_grad():
        for step in tqdm(range(planning_parameters["n_steps"])):
            # EXISTING CEM planning code (keep unchanged)
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
            
            # EXISTING step execution (keep unchanged)
            obs = vec_env.step(best_action)
            obs_arr.append(obs)
            
            # ADD: Failure recognition for each batch item
            for batch_idx in range(obs.shape[0]):
                if not failure_detected[batch_idx]:  # Only check if not already failed
                    
                    # Extract robot state from observation
                    robot_state = type('RobotState', (), {
                        'position': obs[batch_idx, :2],  # x, y position
                        'velocity': obs[batch_idx, 2:4] if obs.shape[1] > 2 else np.array([0, 0]),
                        'heading': np.arctan2(obs[batch_idx, 4], obs[batch_idx, 3]) if obs.shape[1] > 4 else 0.0
                    })()
                    
                    # Extract action for this batch item
                    action_state = type('Action', (), {
                        'throttle': best_action[batch_idx]
                    })()
                    
                    # Goal position for this batch item
                    goal_position = target_robot_state[batch_idx, :2].cpu().numpy()
                    
                    # Get terrain context (simplified)
                    terrain_context = type('TerrainContext', (), {
                        'friction': np.array([1.0]),  # Placeholder - you may need to extract this properly
                        'elevation': np.array([0.0])   # Placeholder
                    })()
                    
                    # Check for failure
                    is_failure, failure_type, details = failure_recognizers[batch_idx].check_for_failure(
                        robot_state, action_state, goal_position, terrain_context
                    )
                    
                    if is_failure:
                        failure_detected[batch_idx] = True
                        failure_info = {
                            'step': step,
                            'failure_type': failure_type.value,
                            'details': details,
                            'position': robot_state.position.copy(),
                            'goal_distance': np.linalg.norm(robot_state.position - goal_position)
                        }
                        failure_details[batch_idx].append(failure_info)
                        
                        print(f"Failure detected in batch {batch_idx} at step {step}: {failure_type.value}")
                        print(f"  Details: {details}")
                        
                        # For now, we just log the failure
                        # In Step 3, we'll add recovery attempts

    obs_arr = np.stack(obs_arr)
    action_arr = np.stack(action_arr)
    vec_env.release_context()
    
    # ADD: Return failure information along with original results
    return obs_arr, action_arr, failure_details

# 3. MODIFY the evaluate_plan_batch function to handle failure information
# FIND this function (around line 122) and ADD failure metrics

def evaluate_plan_batch(target_state, obs_arr, action_arr, failure_details=None):
    """
    Evaluate batch of planning tasks - ENHANCED with failure analysis
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
        
        # ADD: Include failure information in evaluation
        batch_item = {
            "closest_dist": np.min(dist_item),
            "dist": dist_item,
            "success": success,
            "n_steps": n_steps,
            "thr_energy_all": thr_energy_all,
            "thr_energy_success": thr_energy_success,
        }
        
        # ADD: Failure analysis
        if failure_details and batch_idx < len(failure_details):
            batch_item["failures_detected"] = len(failure_details[batch_idx])
            batch_item["failure_types"] = [f['failure_type'] for f in failure_details[batch_idx]]
            batch_item["first_failure_step"] = failure_details[batch_idx][0]['step'] if failure_details[batch_idx] else None
            batch_item["failure_details"] = failure_details[batch_idx]
        else:
            batch_item["failures_detected"] = 0
            batch_item["failure_types"] = []
            batch_item["first_failure_step"] = None
            batch_item["failure_details"] = []
            
        batch_eval.append(batch_item)
    
    return batch_eval

# 4. UPDATE the calling functions to handle the new return values
# FIND run_batch_planning_evaluations function (around line 160) and MODIFY this part:

def run_batch_planning_evaluations(
    run_id, calibrate_robot, planning_parameters, evaluation_seed, reset_split
):
    # ... existing code stays the same until the plan_batch call ...
    
    # CHANGE this line (around line 200):
    # OLD: obs_arr, action_arr = plan_batch(...)
    # NEW: 
    obs_arr, action_arr, failure_details = plan_batch(
        vec_env,
        context,
        eval_setting_batch["env_seed"],
        transition_model,
        context_latent,
        eval_setting_batch["initial_state"],
        eval_setting_batch["target_state"],
        planning_parameters,
    )

    # CHANGE this line:
    # OLD: batch_eval = evaluate_plan_batch(eval_setting_batch["target_state"], obs_arr, action_arr)
    # NEW:
    batch_eval = evaluate_plan_batch(
        eval_setting_batch["target_state"], obs_arr, action_arr, failure_details
    )
    
    # ... rest of the function stays the same ...

# 5. ALSO UPDATE run_single_planning_evaluation function (around line 220)
def run_single_planning_evaluation(
    planning_parameters,
    base_eval_setting,
    manual_context=None,
    manual_target=None,
    display=True,
):
    # ... existing code stays the same until the plan_batch call ...
    
    # CHANGE this line (around line 270):
    # OLD: obs_arr, action_arr = plan_batch(...)
    # NEW:
    obs_arr, action_arr, failure_details = plan_batch(
        vec_env,
        env_context,
        eval_setting_batch["env_seed"],
        transition_model,
        context_latent,
        eval_setting_batch["initial_state"],
        eval_setting_batch["target_state"],
        planning_parameters,
    )

    # CHANGE this line:
    # OLD: batch_eval = evaluate_plan_batch(eval_setting_batch["target_state"], obs_arr, action_arr)
    # NEW:
    batch_eval = evaluate_plan_batch(
        eval_setting_batch["target_state"], obs_arr, action_arr, failure_details
    )
    
    # ... rest of the function stays the same ...
