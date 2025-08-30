# Step 2B: Modifications to unicycle_prediction.py
# These are the EXACT changes to make

# 1. ADD this import at the top (around line 16, after other imports)
from .FailureRecognizer import FailureRecognizer, FailureType

# 2. MODIFY the predict_batch function 
# FIND this function (around line 25) and ENHANCE it

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
    Compute prediction statistics for batch - ENHANCED with failure recognition
    """
    if transition_model.local_context_dim > 0:
        local_ctx_fcn = vec_env.query_terrain_state
    else:
        local_ctx_fcn = None

    # Actual trajectory (ground truth)
    env_context = {"seed": context_seed}
    gt_obs_arr, action_arr = collect_random_traj(
        vec_env, env_context, env_seed, action_seed, initial_state, n_transitions
    )
    if transition_model.local_context_dim > 0:
        gt_obs_arr = gt_obs_arr[..., : -transition_model.local_context_dim]

    # Model prediction
    x = torch.from_numpy(gt_obs_arr[0]).to(transition_model.device).float()
    action_arr_torch = torch.from_numpy(action_arr).to(transition_model.device).float()
    pred_arr = transition_model.forward_multi_step(
        x,
        action_arr_torch,
        context_latent.mean,
        local_ctx_fcn,
        return_mean_only=True,
        probe_dict=None,
    )
    pred_arr = pred_arr.cpu().detach().numpy()
    
    # ADD: Prediction failure analysis
    # Initialize failure recognizers for prediction analysis
    prediction_failure_recognizers = [FailureRecognizer() for _ in range(gt_obs_arr.shape[1])]
    prediction_failures = [[] for _ in range(gt_obs_arr.shape[1])]
    
    # Analyze prediction failures step by step
    for step in range(1, gt_obs_arr.shape[0]):  # Start from step 1 (after initial state)
        for batch_idx in range(gt_obs_arr.shape[1]):
            
            # Extract ground truth robot state
            gt_robot_state = type('RobotState', (), {
                'position': gt_obs_arr[step, batch_idx, :2],
                'velocity': gt_obs_arr[step, batch_idx, 2:4] if gt_obs_arr.shape[2] > 2 else np.array([0, 0]),
                'heading': np.arctan2(gt_obs_arr[step, batch_idx, 4], gt_obs_arr[step, batch_idx, 3]) if gt_obs_arr.shape[2] > 4 else 0.0
            })()
            
            # Extract predicted robot state  
            pred_robot_state = type('RobotState', (), {
                'position': pred_arr[step, batch_idx, :2],
                'velocity': pred_arr[step, batch_idx, 2:4] if pred_arr.shape[2] > 2 else np.array([0, 0]),
                'heading': np.arctan2(pred_arr[step, batch_idx, 4], pred_arr[step, batch_idx, 3]) if pred_arr.shape[2] > 4 else 0.0
            })()
            
            # Action taken
            action_state = type('Action', (), {
                'throttle': action_arr[step-1, batch_idx]  # step-1 because action_arr is shorter
            })()
            
            # Dummy goal (for failure detection - prediction doesn't have explicit goals)
            dummy_goal = gt_obs_arr[-1, batch_idx, :2]  # Use final position as "goal"
            
            # Simple terrain context
            terrain_context = type('TerrainContext', (), {
                'friction': np.array([1.0]),
                'elevation': np.array([0.0])
            })()
            
            # Check if ground truth trajectory shows failure
            is_gt_failure, gt_failure_type, gt_details = prediction_failure_recognizers[batch_idx].check_for_failure(
                gt_robot_state, action_state, dummy_goal, terrain_context
            )
            
            # Check prediction quality - is the prediction diverging from reality?
            position_error = np.linalg.norm(gt_robot_state.position - pred_robot_state.position)
            velocity_error = np.linalg.norm(gt_robot_state.velocity - pred_robot_state.velocity)
            
            # Detect prediction failure (large divergence from ground truth)
            is_pred_failure = (position_error > 0.1) or (velocity_error > 0.5)  # Thresholds to tune
            
            if is_gt_failure or is_pred_failure:
                failure_info = {
                    'step': step,
                    'gt_failure': is_gt_failure,
                    'gt_failure_type': gt_failure_type.value if gt_failure_type else None,
                    'gt_failure_details': gt_details,
                    'prediction_failure': is_pred_failure,
                    'position_error': position_error,
                    'velocity_error': velocity_error,
                    'gt_position': gt_robot_state.position.copy(),
                    'pred_position': pred_robot_state.position.copy()
                }
                prediction_failures[batch_idx].append(failure_info)
    
    return gt_obs_arr, action_arr, pred_arr, prediction_failures

# 3. MODIFY the evaluate_predict_batch function
# FIND this function (around line 60) and ADD failure metrics

def evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr, prediction_failures=None):
    """
    Evaluate batch of prediction tasks - ENHANCED with failure analysis
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
        angle_diff = np.arctan2(np.sin(diff_theta), np.cos(diff_theta))
        
        # EXISTING metrics
        batch_item = {
            "l2_pos": l2_pos,
            "l2_vel": l2_vel,
            "angle_diff": angle_diff,
        }
        
        # ADD: Prediction failure analysis
        if prediction_failures and batch_idx < len(prediction_failures):
            batch_failures = prediction_failures[batch_idx]
            
            # Count different types of failures
            gt_failures = [f for f in batch_failures if f['gt_failure']]
            pred_failures = [f for f in batch_failures if f['prediction_failure']]
            
            batch_item.update({
                "gt_failures_detected": len(gt_failures),
                "prediction_failures_detected": len(pred_failures),
                "total_failures": len(batch_failures),
                "failure_types": [f['gt_failure_type'] for f in gt_failures if f['gt_failure_type']],
                "max_position_error": max([f['position_error'] for f in batch_failures], default=0),
                "max_velocity_error": max([f['velocity_error'] for f in batch_failures], default=0),
                "first_failure_step": batch_failures[0]['step'] if batch_failures else None,
                "failure_details": batch_failures
            })
        else:
            batch_item.update({
                "gt_failures_detected": 0,
                "prediction_failures_detected": 0,
                "total_failures": 0,
                "failure_types": [],
                "max_position_error": 0,
                "max_velocity_error": 0,
                "first_failure_step": None,
                "failure_details": []
            })
            
        batch_eval.append(batch_item)
    
    return batch_eval

# 4. UPDATE the calling functions to handle new return values
# FIND run_batch_prediction_evaluations function (around line 97) and MODIFY:

def run_batch_prediction_evaluations(run_id, calibrate_robot, evaluation_seed):
    # ... existing code stays the same until the predict_batch call ...
    
    # CHANGE this line (around line 140):
    # OLD: gt_obs_arr, action_arr, pred_arr = predict_batch(...)
    # NEW:
    gt_obs_arr, action_arr, pred_arr, prediction_failures = predict_batch(
        vec_env,
        context_seed,
        env_seed,
        pred_action_seed,
        transition_model,
        context_latent,
        initial_state,
        n_pred_transitions,
    )

    # CHANGE this line:
    # OLD: batch_eval = evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr)
    # NEW:
    batch_eval = evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr, prediction_failures)
    
    # ... rest of the function stays the same ...

# 5. ALSO UPDATE run_single_prediction_evaluation function (around line 170)
def run_single_prediction_evaluation():
    # ... existing code stays the same until the predict_batch call ...
    
    # CHANGE this line (around line 200):
    # OLD: gt_obs_arr, action_arr, pred_arr = predict_batch(...)
    # NEW:
    gt_obs_arr, action_arr, pred_arr, prediction_failures = predict_batch(
        vec_env,
        context_seed,
        env_seed,
        pred_action_seed,
        transition_model,
        context_latent,
        initial_state,
        n_pred_transitions,
    )
    
    # ADD: Print failure information for single evaluation
    if prediction_failures:
        print(f"Prediction failures detected: {len(prediction_failures[0])} failures")
        for failure in prediction_failures[0][:3]:  # Show first 3 failures
            print(f"  Step {failure['step']}: GT failure={failure['gt_failure']}, "
                  f"Pred failure={failure['prediction_failure']}, "
                  f"Pos error={failure['position_error']:.4f}")
    
    # The rest of the visualization code stays the same...
