"""
Enhanced unicycle prediction module with integrated failure recognition system.
This module extends the original prediction capabilities to detect and analyze
both ground truth failures and prediction divergences.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import pickle
import os
import yaml

# ADD: Import the FailureRecognizer
from .FailureRecognizer import FailureRecognizer, FailureType

# Original imports (update these based on your actual imports)
from .data_utils import collect_random_traj
from .models import load_transition_model


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
    
    Returns:
        gt_obs_arr: Ground truth observations
        action_arr: Actions taken
        pred_arr: Model predictions
        prediction_failures: List of failure analysis per batch
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


def evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr, prediction_failures=None):
    """
    Evaluate batch of prediction tasks - ENHANCED with failure analysis
    
    Returns:
        batch_eval: List of evaluation metrics including failure analysis
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


def run_batch_prediction_evaluations(run_id, calibrate_robot, evaluation_seed):
    """
    Run batch prediction evaluations with failure analysis
    """
    # Load configuration and models
    config_path = f"configs/{run_id}.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize environment
    vec_env = create_vec_env(config['env_params'])
    
    # Load transition model
    transition_model = load_transition_model(run_id, config)
    
    # Set seeds
    np.random.seed(evaluation_seed)
    torch.manual_seed(evaluation_seed)
    
    # Generate test contexts
    n_test_contexts = config.get('n_test_contexts', 10)
    n_pred_transitions = config.get('n_pred_transitions', 100)
    
    all_results = []
    
    for context_idx in range(n_test_contexts):
        context_seed = np.random.randint(0, 10000)
        env_seed = np.random.randint(0, 10000)
        pred_action_seed = np.random.randint(0, 10000)
        
        # Get context latent
        context_latent = transition_model.encode_context(context_seed)
        
        # Get initial state
        vec_env.reset(seed=env_seed)
        initial_state = vec_env.get_state()
        
        # MODIFIED: Get predictions with failure analysis
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
        
        # MODIFIED: Evaluate with failure analysis
        batch_eval = evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr, prediction_failures)
        
        # Store results
        results = {
            'context_idx': context_idx,
            'context_seed': context_seed,
            'batch_eval': batch_eval,
            'failure_summary': {
                'total_gt_failures': sum(item['gt_failures_detected'] for item in batch_eval),
                'total_pred_failures': sum(item['prediction_failures_detected'] for item in batch_eval),
                'unique_failure_types': list(set(
                    ft for item in batch_eval for ft in item['failure_types']
                ))
            }
        }
        all_results.append(results)
        
        # Print summary
        print(f"Context {context_idx}: GT failures={results['failure_summary']['total_gt_failures']}, "
              f"Pred failures={results['failure_summary']['total_pred_failures']}")
    
    return all_results


def run_single_prediction_evaluation(
    run_id, 
    context_seed=1234, 
    env_seed=5678, 
    pred_action_seed=9012,
    n_pred_transitions=100,
    visualize=True
):
    """
    Run single prediction evaluation with failure analysis and visualization
    """
    # Load configuration and models
    config_path = f"configs/{run_id}.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize environment
    vec_env = create_vec_env(config['env_params'])
    
    # Load transition model
    transition_model = load_transition_model(run_id, config)
    
    # Get context latent
    context_latent = transition_model.encode_context(context_seed)
    
    # Get initial state
    vec_env.reset(seed=env_seed)
    initial_state = vec_env.get_state()
    
    # MODIFIED: Get predictions with failure analysis
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
        print(f"\nPrediction failures detected: {len(prediction_failures[0])} failures")
        for failure in prediction_failures[0][:3]:  # Show first 3 failures
            print(f"  Step {failure['step']}: GT failure={failure['gt_failure']}, "
                  f"Pred failure={failure['prediction_failure']}, "
                  f"Pos error={failure['position_error']:.4f}")
    
    # Visualization
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot trajectories
        ax = axes[0, 0]
        ax.plot(gt_obs_arr[:, 0, 0], gt_obs_arr[:, 0, 1], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(pred_arr[:, 0, 0], pred_arr[:, 0, 1], 'r--', label='Prediction', linewidth=2)
        
        # Mark failure points
        if prediction_failures and prediction_failures[0]:
            for failure in prediction_failures[0]:
                if failure['gt_failure']:
                    ax.plot(failure['gt_position'][0], failure['gt_position'][1], 
                           'ko', markersize=8, label='GT Failure' if failure == prediction_failures[0][0] else "")
                if failure['prediction_failure']:
                    ax.plot(failure['pred_position'][0], failure['pred_position'][1], 
                           'rx', markersize=8, label='Pred Divergence' if failure == prediction_failures[0][0] else "")
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Trajectory Comparison with Failure Points')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot position error over time
        ax = axes[0, 1]
        pos_errors = np.linalg.norm(gt_obs_arr[:, 0, :2] - pred_arr[:, 0, :2], axis=1)
        ax.plot(pos_errors, 'b-', linewidth=2)
        ax.axhline(y=0.1, color='r', linestyle='--', label='Failure Threshold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Position Error')
        ax.set_title('Prediction Error Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot velocity comparison
        ax = axes[1, 0]
        ax.plot(gt_obs_arr[:, 0, 2], 'b-', label='GT Velocity', linewidth=2)
        ax.plot(pred_arr[:, 0, 2], 'r--', label='Pred Velocity', linewidth=2)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Velocity')
        ax.set_title('Velocity Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot failure timeline
        ax = axes[1, 1]
        if prediction_failures and prediction_failures[0]:
            failure_steps = [f['step'] for f in prediction_failures[0]]
            failure_types = [f['gt_failure_type'] if f['gt_failure'] else 'Prediction' 
                           for f in prediction_failures[0]]
            
            # Create a timeline visualization
            for i, (step, ftype) in enumerate(zip(failure_steps, failure_types)):
                color = 'red' if ftype != 'Prediction' else 'orange'
                ax.barh(i, 1, left=step, height=0.8, color=color, alpha=0.7, 
                       label=ftype if i == 0 else "")
                ax.text(step + 0.5, i, f"{ftype[:10]}", va='center', fontsize=8)
        
        ax.set_xlim(0, n_pred_transitions)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Failure Events')
        ax.set_title('Failure Timeline')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
    
    return gt_obs_arr, pred_arr, prediction_failures


def create_vec_env(env_params):
    """
    Create vectorized environment
    Note: Implement this based on your specific environment setup
    """
    # Placeholder - replace with your actual environment creation
    pass


def load_transition_model(run_id, config):
    """
    Load trained transition model
    Note: Implement this based on your model loading logic
    """
    # Placeholder - replace with your actual model loading
    pass


if __name__ == "__main__":
    # Example usage
    run_id = "example_run"
    
    # Run batch evaluation with failure analysis
    print("Running batch prediction evaluations with failure analysis...")
    batch_results = run_batch_prediction_evaluations(
        run_id=run_id,
        calibrate_robot=False,
        evaluation_seed=42
    )
    
    # Run single evaluation with visualization
    print("\nRunning single prediction evaluation with visualization...")
    gt, pred, failures = run_single_prediction_evaluation(
        run_id=run_id,
        context_seed=1234,
        env_seed=5678,
        pred_action_seed=9012,
        n_pred_transitions=100,
        visualize=True
    )
    
    print("\nEvaluation complete!")
