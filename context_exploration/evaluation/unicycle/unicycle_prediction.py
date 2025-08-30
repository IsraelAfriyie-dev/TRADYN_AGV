# Step 2B: Exact Modifications to unicycle_prediction.py
# Integration with TRADYN AGV Failure Recognition System
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from datetime import datetime
import pickle
import json

# 1. ADD this import at the top (around line 16, after other imports)
from failure_recognizer import (
    FailureLearningSystem, 
    TerrainFailureDetector, 
    FailureType,
    TerrainContext,
    RobotContext,
    create_terrain_context_from_data,
    create_robot_context_from_state
)
import numpy as np
from datetime import datetime

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
    enable_failure_recognition=True,  # ADD: Optional parameter
):
    """
    Compute prediction statistics for batch - ENHANCED with failure recognition
    
    This enhanced version detects both ground truth trajectory failures AND
    prediction quality failures (when model predictions diverge from reality).
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
    
    # ADD: Enhanced prediction failure analysis
    prediction_failures = [[] for _ in range(gt_obs_arr.shape[1])]
    
    if enable_failure_recognition:
        # Initialize failure detection systems for each batch item
        failure_detectors = []
        terrain_contexts = []
        robot_contexts = []
        
        for batch_idx in range(gt_obs_arr.shape[1]):
            # Create failure detector with prediction-specific configuration
            config = {
                'prediction_error_threshold': 0.1,  # Position error threshold
                'velocity_error_threshold': 0.2,    # Velocity error threshold
                'min_progress_rate': 0.05,          # For ground truth failure detection
                'stuck_time_threshold': 3.0,       # For ground truth failure detection
            }
            detector = TerrainFailureDetector(config)
            
            # Create terrain context (you may need to extract this from environment)
            terrain_data = {
                'slope_angle': env_context.get('slope_angle', 0.0),
                'friction_coefficient': env_context.get('friction_coefficient', 0.7),
                'terrain_roughness': env_context.get('roughness', 0.1),
                'obstacle_density': env_context.get('obstacle_density', 0.0),
                'terrain_type': env_context.get('terrain_type', 'unknown')
            }
            terrain_context = create_terrain_context_from_data(terrain_data)
            
            # Create robot context
            robot_data = {
                'mass': 20.0,  # Default values - extract from actual system if available
                'max_velocity': 2.0,
                'max_acceleration': 1.0,
                'wheel_radius': 0.1,
                'battery_level': 1.0
            }
            robot_context = create_robot_context_from_state(robot_data)
            
            failure_detectors.append(detector)
            terrain_contexts.append(terrain_context)
            robot_contexts.append(robot_context)
        
        # Analyze prediction failures step by step
        for step in range(1, gt_obs_arr.shape[0]):  # Start from step 1 (after initial state)
            current_time = step * 0.1  # Assume 0.1 second time steps
            
            for batch_idx in range(gt_obs_arr.shape[1]):
                detector = failure_detectors[batch_idx]
                terrain_context = terrain_contexts[batch_idx]
                robot_context = robot_contexts[batch_idx]
                
                # Extract ground truth state
                gt_position = gt_obs_arr[step, batch_idx, :2]
                gt_velocity = gt_obs_arr[step, batch_idx, 2:4] if gt_obs_arr.shape[2] > 2 else np.array([0.0, 0.0])
                
                # Extract predicted state
                pred_position = pred_arr[step, batch_idx, :2]
                pred_velocity = pred_arr[step, batch_idx, 2:4] if pred_arr.shape[2] > 2 else np.array([0.0, 0.0])
                
                # Update ground truth failure detector state
                energy_consumed = step * 0.01  # Simple energy model
                battery_level = max(0.0, 1.0 - energy_consumed)
                
                detector.update_state(current_time, gt_position, energy_consumed, gt_velocity)
                
                # 1. Check for ground truth trajectory failures
                dummy_goal = gt_obs_arr[-1, batch_idx, :2]  # Use final position as goal
                gt_failures = detector.check_all_failures(
                    dummy_goal, battery_level, terrain_context
                )
                
                # 2. Check for prediction quality failures
                position_error = np.linalg.norm(pred_position - gt_position)
                velocity_error = np.linalg.norm(pred_velocity - gt_velocity)
                
                # Detect significant prediction divergence
                prediction_quality_failure = None
                if position_error > detector.thresholds['prediction_error_threshold']:
                    prediction_quality_failure = {
                        'failure_type': FailureType.DYNAMICS_MISMATCH,
                        'subtype': 'position_prediction_error',
                        'severity': min(1.0, position_error / detector.thresholds['prediction_error_threshold']),
                        'position_error': position_error,
                        'step': step
                    }
                elif velocity_error > 0.2:  # Velocity error threshold
                    prediction_quality_failure = {
                        'failure_type': FailureType.DYNAMICS_MISMATCH,
                        'subtype': 'velocity_prediction_error', 
                        'severity': min(1.0, velocity_error / 0.2),
                        'velocity_error': velocity_error,
                        'step': step
                    }
                
                # 3. Check for cumulative prediction drift
                if step > 5:  # Need some history
                    # Calculate cumulative drift from initial prediction
                    initial_pred_pos = pred_arr[0, batch_idx, :2]
                    current_gt_pos = gt_obs_arr[step, batch_idx, :2]
                    cumulative_drift = np.linalg.norm(pred_position - (initial_pred_pos + (current_gt_pos - gt_obs_arr[0, batch_idx, :2])))
                    
                    if cumulative_drift > 0.3:  # Cumulative drift threshold
                        prediction_quality_failure = {
                            'failure_type': FailureType.CONTEXT_ERROR,
                            'subtype': 'cumulative_prediction_drift',
                            'severity': min(1.0, cumulative_drift / 0.5),
                            'cumulative_drift': cumulative_drift,
                            'step': step
                        }
                
                # Record failures
                if gt_failures or prediction_quality_failure:
                    failure_info = {
                        'step': step,
                        'time': current_time,
                        
                        # Ground truth failure information
                        'gt_failure': len(gt_failures) > 0,
                        'gt_failure_type': gt_failures[0]['failure_type'].value if gt_failures else None,
                        'gt_failure_details': gt_failures[0] if gt_failures else None,
                        
                        # Prediction quality failure information
                        'prediction_failure': prediction_quality_failure is not None,
                        'pred_failure_type': prediction_quality_failure['failure_type'].value if prediction_quality_failure else None,
                        'pred_failure_subtype': prediction_quality_failure.get('subtype') if prediction_quality_failure else None,
                        'pred_failure_severity': prediction_quality_failure.get('severity', 0) if prediction_quality_failure else 0,
                        
                        # Detailed error metrics
                        'position_error': position_error,
                        'velocity_error': velocity_error,
                        'gt_position': gt_position.copy(),
                        'pred_position': pred_position.copy(),
                        'gt_velocity': gt_velocity.copy(),
                        'pred_velocity': pred_velocity.copy(),
                        
                        # Context information
                        'terrain_type': terrain_context.terrain_type,
                        'slope_angle': terrain_context.slope_angle,
                    }
                    
                    prediction_failures[batch_idx].append(failure_info)
                    
                    # Print significant failures (optional - can be controlled by verbosity setting)
                    if position_error > 0.2 or (gt_failures and len(prediction_failures[batch_idx]) == 1):
                        print(f"Prediction analysis - Batch {batch_idx}, Step {step}:")
                        if gt_failures:
                            print(f"  Ground truth failure: {gt_failures[0]['failure_type'].value}")
                        if prediction_quality_failure:
                            print(f"  Prediction failure: {prediction_quality_failure['subtype']} "
                                  f"(severity: {prediction_quality_failure['severity']:.3f})")
                        print(f"  Position error: {position_error:.4f}, Velocity error: {velocity_error:.4f}")
    
    return gt_obs_arr, action_arr, pred_arr, prediction_failures


# 3. MODIFY the evaluate_predict_batch function
# FIND this function (around line 60) and ADD failure metrics
def evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr, prediction_failures=None):
    """
    Evaluate batch of prediction tasks - ENHANCED with comprehensive failure analysis
    
    Now includes both traditional prediction metrics AND failure detection metrics.
    """
    # x, y, v, cos(th), sin(th), <terrain>
    l2_norm = lambda x: np.sqrt((x ** 2).sum(axis=-1))

    batch_eval = []
    for batch_idx in range(gt_obs_arr.shape[1]):
        # EXISTING traditional metrics
        l2_pos = l2_norm(gt_obs_arr[:, batch_idx, :2] - pred_arr[:, batch_idx, :2])
        l2_vel = l2_norm(gt_obs_arr[:, batch_idx, 2:3] - pred_arr[:, batch_idx, 2:3])
        gt_theta = np.arctan2(gt_obs_arr[:, batch_idx, 4], gt_obs_arr[:, batch_idx, 3])
        pred_theta = np.arctan2(pred_arr[:, batch_idx, 4], pred_arr[:, batch_idx, 3])
        diff_theta = pred_theta - gt_theta
        # Normalize angle difference
        angle_diff = np.arctan2(np.sin(diff_theta), np.cos(diff_theta))
        
        batch_item = {
            "l2_pos": l2_pos,
            "l2_vel": l2_vel,
            "angle_diff": angle_diff,
        }
        
        # ADD: Enhanced prediction failure analysis
        if prediction_failures and batch_idx < len(prediction_failures):
            batch_failures = prediction_failures[batch_idx]
            
            # Separate ground truth failures from prediction quality failures
            gt_failures = [f for f in batch_failures if f['gt_failure']]
            pred_failures = [f for f in batch_failures if f['prediction_failure']]
            
            # Calculate failure statistics
            position_errors = [f['position_error'] for f in batch_failures]
            velocity_errors = [f['velocity_error'] for f in batch_failures]
            
            batch_item.update({
                # Basic failure counts
                "gt_failures_detected": len(gt_failures),
                "prediction_failures_detected": len(pred_failures),
                "total_failures": len(batch_failures),
                
                # Failure type analysis
                "gt_failure_types": [f['gt_failure_type'] for f in gt_failures if f['gt_failure_type']],
                "pred_failure_types": [f['pred_failure_subtype'] for f in pred_failures if f['pred_failure_subtype']],
                
                # Error statistics
                "max_position_error": max(position_errors) if position_errors else 0.0,
                "avg_position_error": np.mean(position_errors) if position_errors else 0.0,
                "max_velocity_error": max(velocity_errors) if velocity_errors else 0.0,
                "avg_velocity_error": np.mean(velocity_errors) if velocity_errors else 0.0,
                
                # Failure timing
                "first_failure_step": batch_failures[0]['step'] if batch_failures else None,
                "last_failure_step": batch_failures[-1]['step'] if batch_failures else None,
                "failure_rate": len(batch_failures) / gt_obs_arr.shape[0],  # failures per step
                
                # Prediction quality assessment
                "prediction_quality": (
                    "poor" if any(f['position_error'] > 0.2 for f in batch_failures) else
                    "moderate" if any(f['position_error'] > 0.1 for f in batch_failures) else
                    "good"
                ),
                
                # Terrain context (from first failure)
                "terrain_context": {
                    "terrain_type": batch_failures[0].get('terrain_type', 'unknown'),
                    "slope_angle": batch_failures[0].get('slope_angle', 0.0)
                } if batch_failures else None,
                
                # Severity analysis
                "max_failure_severity": max(
                    [f.get('pred_failure_severity', 0) for f in pred_failures], 
                    default=0.0
                ),
                
                # Detailed failure information
                "failure_details": batch_failures
            })
        else:
            # No failures detected
            batch_item.update({
                "gt_failures_detected": 0,
                "prediction_failures_detected": 0,
                "total_failures": 0,
                "gt_failure_types": [],
                "pred_failure_types": [],
                "max_position_error": 0.0,
                "avg_position_error": 0.0,
                "max_velocity_error": 0.0,
                "avg_velocity_error": 0.0,
                "first_failure_step": None,
                "last_failure_step": None,
                "failure_rate": 0.0,
                "prediction_quality": "good",
                "terrain_context": None,
                "max_failure_severity": 0.0,
                "failure_details": []
            })
            
        batch_eval.append(batch_item)
    
    return batch_eval


# 4. UPDATE the calling functions to handle new return values
# FIND run_batch_prediction_evaluations function (around line 97) and MODIFY:
def run_batch_prediction_evaluations(run_id, calibrate_robot, evaluation_seed, 
                                    enable_failure_recognition=True):
    """Enhanced batch prediction evaluations with failure recognition."""
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
        enable_failure_recognition=enable_failure_recognition
    )

    # CHANGE this line:
    # OLD: batch_eval = evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr)
    # NEW:
    batch_eval = evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr, prediction_failures)
    
    # ... rest of the function stays the same ...


# 5. ALSO UPDATE run_single_prediction_evaluation function (around line 170)
def run_single_prediction_evaluation(run_id=None, display=True, enable_failure_recognition=True):
    """Enhanced single prediction evaluation with failure recognition and detailed reporting."""
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
        enable_failure_recognition=enable_failure_recognition
    )
    
    # CHANGE this line:
    # OLD: batch_eval = evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr)
    # NEW:
    batch_eval = evaluate_predict_batch(gt_obs_arr, action_arr, pred_arr, prediction_failures)
    
    # ADD: Enhanced failure reporting for single evaluation
    if display and enable_failure_recognition and prediction_failures:
        print("\n" + "="*60)
        print("PREDICTION FAILURE ANALYSIS")
        print("="*60)
        
        for batch_idx, batch_failures in enumerate(prediction_failures):
            if batch_failures:
                print(f"\nBatch {batch_idx}: {len(batch_failures)} failures detected")
                
                # Summary statistics
                gt_failures = [f for f in batch_failures if f['gt_failure']]
                pred_failures = [f for f in batch_failures if f['prediction_failure']]
                
                print(f"  Ground truth failures: {len(gt_failures)}")
                print(f"  Prediction quality failures: {len(pred_failures)}")
                
                if batch_failures:
                    avg_pos_error = np.mean([f['position_error'] for f in batch_failures])
                    max_pos_error = max([f['position_error'] for f in batch_failures])
                    print(f"  Average position error: {avg_pos_error:.4f}")
                    print(f"  Maximum position error: {max_pos_error:.4f}")
                    
                    # Show worst failures
                    worst_failures = sorted(batch_failures, key=lambda x: x['position_error'], reverse=True)[:3]
                    print(f"  Top {min(3, len(worst_failures))} worst failures:")
                    for i, failure in enumerate(worst_failures):
                        print(f"    {i+1}. Step {failure['step']}: {failure.get('pred_failure_subtype', 'unknown')} "
                              f"(pos error: {failure['position_error']:.4f})")
                
                # Terrain context
                if batch_failures and batch_failures[0].get('terrain_type'):
                    terrain = batch_failures[0]['terrain_type']
                    slope = batch_failures[0].get('slope_angle', 0)
                    print(f"  Terrain: {terrain} (slope: {slope:.1f}°)")
            else:
                print(f"\nBatch {batch_idx}: No failures detected")
        
        # Overall summary
        total_failures = sum(len(failures) for failures in prediction_failures)
        if total_failures > 0:
            print(f"\nOVERALL SUMMARY:")
            print(f"  Total failures across all batches: {total_failures}")
            
            # Most common failure types
            all_pred_failures = []
            for batch_failures in prediction_failures:
                all_pred_failures.extend([f for f in batch_failures if f['prediction_failure']])
            
            if all_pred_failures:
                failure_subtypes = [f.get('pred_failure_subtype', 'unknown') for f in all_pred_failures]
                from collections import Counter
                failure_counts = Counter(failure_subtypes)
                print(f"  Most common prediction failure types:")
                for ftype, count in failure_counts.most_common(3):
                    print(f"    {ftype}: {count} occurrences")
    
    # The rest of the visualization code stays the same...
    return gt_obs_arr, action_arr, pred_arr, batch_eval, prediction_failures


# 6. ADD helper function for prediction failure pattern analysis
def analyze_prediction_failure_patterns(batch_results, print_summary=True):
    """
    Analyze prediction failure patterns across batch results.
    
    This helps identify systematic issues with the dynamics model's predictions
    and suggests areas for improvement.
    """
    if not batch_results:
        return {}
    
    all_failures = []
    prediction_quality_counts = {"good": 0, "moderate": 0, "poor": 0}
    terrain_performance = {}
    total_evaluations = len(batch_results)
    
    # Collect all failure information
    for result in batch_results:
        prediction_quality_counts[result.get("prediction_quality", "good")] += 1
        
        # Collect terrain-specific performance
        terrain_context = result.get("terrain_context")
        if terrain_context:
            terrain_type = terrain_context["terrain_type"]
            if terrain_type not in terrain_performance:
                terrain_performance[terrain_type] = {
                    "evaluations": 0,
                    "failures": 0,
                    "avg_errors": [],
                    "slope_angles": []
                }
            
            terrain_performance[terrain_type]["evaluations"] += 1
            terrain_performance[terrain_type]["failures"] += result.get("total_failures", 0)
            terrain_performance[terrain_type]["avg_errors"].append(result.get("avg_position_error", 0))
            terrain_performance[terrain_type]["slope_angles"].append(terrain_context.get("slope_angle", 0))
        
        # Collect individual failures
        for failure in result.get("failure_details", []):
            all_failures.append(failure)
    
    # Analyze failure patterns
    failure_type_counts = {}
    prediction_errors = []
    early_failures = 0
    
    for failure in all_failures:
        # Count failure types
        if failure.get('prediction_failure'):
            ftype = failure.get('pred_failure_subtype', 'unknown')
            failure_type_counts[ftype] = failure_type_counts.get(ftype, 0) + 1
        
        prediction_errors.append(failure.get('position_error', 0))
        
        # Count early failures (within first 10 steps)
        if failure.get('step', 0) < 10:
            early_failures += 1
    
    # Calculate terrain performance statistics
    for terrain_type, perf in terrain_performance.items():
        if perf["avg_errors"]:
            perf["mean_error"] = np.mean(perf["avg_errors"])
            perf["failure_rate"] = perf["failures"] / perf["evaluations"]
            perf["mean_slope"] = np.mean(perf["slope_angles"])
    
    analysis = {
        'total_evaluations': total_evaluations,
        'total_failures': len(all_failures),
        'prediction_quality_distribution': prediction_quality_counts,
        'common_failure_types': failure_type_counts,
        'terrain_performance': terrain_performance,
        'avg_prediction_error': np.mean(prediction_errors) if prediction_errors else 0,
        'max_prediction_error': max(prediction_errors) if prediction_errors else 0,
        'early_failure_rate': early_failures / len(all_failures) if all_failures else 0,
        'overall_failure_rate': len(all_failures) / total_evaluations if total_evaluations > 0 else 0
    }
    
    if print_summary:
        print(f"\nPREDICTION FAILURE PATTERN ANALYSIS:")
        print(f"  Total evaluations: {analysis['total_evaluations']}")
        print(f"  Total failures detected: {analysis['total_failures']}")
        print(f"  Overall failure rate: {analysis['overall_failure_rate']:.1%}")
        print(f"  Average prediction error: {analysis['avg_prediction_error']:.4f}")
        
        print(f"\nPrediction Quality Distribution:")
        for quality, count in prediction_quality_counts.items():
            percentage = count / total_evaluations * 100
            print(f"    {quality}: {count} ({percentage:.1f}%)")
        
        if failure_type_counts:
            print(f"\nMost Common Prediction Failure Types:")
            sorted_failures = sorted(failure_type_counts.items(), key=lambda x: x[1], reverse=True)
            for ftype, count in sorted_failures[:5]:
                print(f"    {ftype}: {count} occurrences")
        
        if terrain_performance:
            print(f"\nPerformance by Terrain Type:")
            for terrain, perf in terrain_performance.items():
                print(f"    {terrain}: {perf['failure_rate']:.1%} failure rate, "
                      f"avg error: {perf.get('mean_error', 0):.4f}")
        
        if analysis['early_failure_rate'] > 0.3:
            print(f"  ⚠️  High early failure rate ({analysis['early_failure_rate']:.1%}) - "
                  f"model may have systematic issues")
    
    return analysis


# Example usage configuration
if __name__ == "__main__":
    # Example of running prediction evaluation with failure recognition
    print("TRADYN Prediction Evaluation with Failure Recognition")
    print("=" * 60)
    
    # Run single evaluation with failure analysis
    results = run_single_prediction_evaluation(
        run_id="test_with_failures",
        display=True,
        enable_failure_recognition=True
    )
    
    # The results now include failure information that can be used for:
    # 1. Identifying when model predictions are unreliable
    # 2. Understanding which terrain types cause prediction failures
    # 3. Learning which contexts require model improvements
    # 4. Transferring knowledge about prediction reliability between similar terrains
