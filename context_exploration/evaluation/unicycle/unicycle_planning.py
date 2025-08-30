# Modifications for unicycle_planning.py
# Integration with TRADYN AGV Failure Recognition System

# 1. ADD these imports at the top (around line 18, after the other imports)
from .FailureRecognizer import (
    FailureLearningSystem, 
    TerrainContext, 
    RobotContext, 
    FailureType,
    create_terrain_context_from_data,
    create_robot_context_from_state
)
import numpy as np
from datetime import datetime

# 2. MODIFY the plan_batch function 
# FIND this function (around line 66) and REPLACE with enhanced version

def plan_batch(
    vec_env,
    context,
    env_seed,
    transition_model,
    context_latent,
    initial_robot_state,
    target_robot_state,
    planning_parameters,
    enable_failure_recognition=True,  # ADD: Optional parameter
):
    """
    Solve a batch of planning problems - ENHANCED with failure recognition and learning
    
    This enhanced version detects navigation failures during planning execution,
    learns from past failures on similar terrain, and provides failure insights
    for improving future planning attempts.
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
    
    # ADD: Initialize failure recognition system for each batch item
    failure_systems = []
    failure_contexts = []
    robot_contexts = []
    navigation_ids = []
    
    if enable_failure_recognition:
        for batch_idx in range(obs.shape[0]):
            # Create failure recognition system
            config = {
                'min_progress_rate': planning_parameters.get('min_progress_rate', 0.05),
                'stuck_time_threshold': planning_parameters.get('stuck_time_threshold', 5.0),
                'max_energy_rate': planning_parameters.get('max_energy_rate', 0.5),
            }
            failure_system = FailureLearningSystem(config)
            
            # Extract terrain context from environment context
            terrain_data = {
                'slope_angle': context.get('slope_angle', 0.0),
                'friction_coefficient': context.get('friction_coefficient', 0.7),
                'terrain_roughness': context.get('roughness', 0.1),
                'obstacle_density': context.get('obstacle_density', 0.0),
                'terrain_type': context.get('terrain_type', 'unknown')
            }
            terrain_context = create_terrain_context_from_data(terrain_data)
            
            # Extract robot context
            robot_data = {
                'mass': context.get('robot_mass', 20.0),
                'max_velocity': planning_parameters.get('max_velocity', 2.0),
                'max_acceleration': planning_parameters.get('max_acceleration', 1.0),
                'wheel_radius': context.get('wheel_radius', 0.1),
                'battery_level': 1.0  # Start with full battery
            }
            robot_context = create_robot_context_from_state(robot_data)
            
            # Start navigation monitoring
            nav_id = f"planning_batch_{env_seed}_{batch_idx}_{datetime.now().strftime('%H%M%S')}"
            start_pos = obs[batch_idx, :2]
            goal_pos = target_robot_state[batch_idx, :2].cpu().numpy()
            
            # Get insights from similar terrain experiences
            similar_experiences, lessons = failure_system.start_navigation(
                nav_id, start_pos, goal_pos, terrain_context, robot_context
            )
            
            if similar_experiences:
                print(f"Batch {batch_idx}: Found {len(similar_experiences)} similar terrain experiences")
                # Apply lessons to planning parameters (optional enhancement)
                terrain_insights = failure_system.get_terrain_insights(terrain_context)
                if terrain_insights['common_failures']:
                    print(f"  Common failures on this terrain: {terrain_insights['common_failures'][:2]}")
            
            failure_systems.append(failure_system)
            failure_contexts.append(terrain_context)
            robot_contexts.append(robot_context)
            navigation_ids.append(nav_id)
    else:
        failure_systems = [None] * obs.shape[0]
        failure_contexts = [None] * obs.shape[0]
        robot_contexts = [None] * obs.shape[0]
        navigation_ids = [None] * obs.shape[0]

    # Planning execution with failure monitoring
    failure_details = [[] for _ in range(obs.shape[0])]
    total_energy_consumed = [0.0] * obs.shape[0]
    start_time = 0.0
    
    with torch.no_grad():
        for step in tqdm(range(planning_parameters["n_steps"])):
            current_time = step * 0.1  # Assume 0.1 second time steps
            
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
            
            # ADD: Failure recognition and monitoring
            if enable_failure_recognition:
                for batch_idx in range(obs.shape[0]):
                    failure_system = failure_systems[batch_idx]
                    terrain_context = failure_contexts[batch_idx]
                    robot_context = robot_contexts[batch_idx]
                    
                    if failure_system is None:
                        continue
                        
                    # Extract current state information
                    current_position = obs[batch_idx, :2]
                    current_velocity = obs[batch_idx, 2:4] if obs.shape[1] > 2 else np.array([0.0, 0.0])
                    goal_position = target_robot_state[batch_idx, :2].cpu().numpy()
                    
                    # Estimate energy consumption from throttle action
                    throttle = best_action[batch_idx, 0] if best_action.shape[1] > 0 else 0.0
                    energy_step = abs(throttle) * 0.01  # Simple energy model
                    total_energy_consumed[batch_idx] += energy_step
                    
                    # Update battery level
                    battery_level = max(0.0, 1.0 - total_energy_consumed[batch_idx])
                    robot_context.battery_level = battery_level
                    
                    # Update failure monitoring system
                    failure_system.update_navigation_state(
                        current_time, current_position, total_energy_consumed[batch_idx], current_velocity, battery_level
                    )
                    
                    # Check for failures
                    detected_failures = failure_system.check_for_failures(
                        goal_position, battery_level, terrain_context
                    )
                    
                    # Process detected failures
                    for failure_info in detected_failures:
                        failure_event = failure_system.record_failure(
                            failure_info, terrain_context, robot_context,
                            obs_arr[0][batch_idx, :2],  # start position
                            goal_position,  # goal position
                            current_position,  # failure position
                            [],  # planned trajectory (could be enhanced)
                            [obs_step[batch_idx, :2] for obs_step in obs_arr],  # actual trajectory so far
                            total_energy_consumed[batch_idx],
                            current_time
                        )
                        
                        # Add to failure details for this batch
                        failure_details[batch_idx].append({
                            'step': step,
                            'time': current_time,
                            'failure_type': failure_info['failure_type'].value,
                            'severity': failure_info.get('severity', 0.5),
                            'position': current_position.copy(),
                            'goal_distance': np.linalg.norm(current_position - goal_position),
                            'energy_consumed': total_energy_consumed[batch_idx],
                            'battery_level': battery_level,
                            'failure_details': failure_info
                        })
                        
                        print(f"Planning Failure - Batch {batch_idx}, Step {step}: {failure_info['failure_type'].value}")
                        print(f"  Severity: {failure_info.get('severity', 0.5):.3f}")
                        print(f"  Distance to goal: {np.linalg.norm(current_position - goal_position):.3f}")
                        
                        # Get recovery suggestions
                        strategies = failure_system.suggest_recovery_strategies(
                            failure_event.failure_type, terrain_context
                        )
                        if strategies:
                            print(f"  Suggested strategies: {strategies[:2]}")  # Show top 2
                            
                            # For now, just log the suggestions
                            # In future versions, could modify planning parameters or retry

    # Finalize failure monitoring
    if enable_failure_recognition:
        for batch_idx, failure_system in enumerate(failure_systems):
            if failure_system:
                # Determine if navigation was successful
                final_position = obs_arr[-1][batch_idx, :2]
                goal_position = target_robot_state[batch_idx, :2].cpu().numpy()
                success = np.linalg.norm(final_position - goal_position) <= 0.05
                
                failure_system.finish_navigation(success)
                
                if not success and not failure_details[batch_idx]:
                    # Navigation failed but no specific failure detected - record as general failure
                    failure_details[batch_idx].append({
                        'step': planning_parameters["n_steps"] - 1,
                        'time': (planning_parameters["n_steps"] - 1) * 0.1,
                        'failure_type': 'goal_not_reached',
                        'severity': 0.7,
                        'position': final_position.copy(),
                        'goal_distance': np.linalg.norm(final_position - goal_position),
                        'energy_consumed': total_energy_consumed[batch_idx],
                        'battery_level': robot_contexts[batch_idx].battery_level,
                        'failure_details': {'reason': 'Failed to reach goal within time limit'}
                    })

    obs_arr = np.stack(obs_arr)
    action_arr = np.stack(action_arr)
    vec_env.release_context()
    
    # Return results including failure information
    return obs_arr, action_arr, failure_details


# 3. MODIFY the evaluate_plan_batch function to handle failure information
def evaluate_plan_batch(target_state, obs_arr, action_arr, failure_details=None):
    """
    Evaluate batch of planning tasks - ENHANCED with comprehensive failure analysis
    
    Now includes failure metrics alongside traditional success/energy metrics.
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
        
        # Original evaluation metrics
        batch_item = {
            "closest_dist": np.min(dist_item),
            "dist": dist_item,
            "success": success,
            "n_steps": n_steps,
            "thr_energy_all": thr_energy_all,
            "thr_energy_success": thr_energy_success,
        }
        
        # ADD: Enhanced failure analysis
        if failure_details and batch_idx < len(failure_details):
            batch_failures = failure_details[batch_idx]
            
            # Basic failure counts
            batch_item["failures_detected"] = len(batch_failures)
            batch_item["failure_types"] = [f['failure_type'] for f in batch_failures]
            batch_item["first_failure_step"] = batch_failures[0]['step'] if batch_failures else None
            
            # Failure severity analysis
            if batch_failures:
                severities = [f.get('severity', 0.5) for f in batch_failures]
                batch_item["max_failure_severity"] = max(severities)
                batch_item["avg_failure_severity"] = sum(severities) / len(severities)
                
                # Energy at failure
                energy_at_failures = [f.get('energy_consumed', 0) for f in batch_failures]
                batch_item["energy_at_first_failure"] = energy_at_failures[0] if energy_at_failures else 0
                
                # Progress at failure
                goal_distances = [f.get('goal_distance', float('inf')) for f in batch_failures]
                batch_item["distance_at_first_failure"] = goal_distances[0] if goal_distances else float('inf')
                
                # Categorize dominant failure type
                failure_type_counts = {}
                for f in batch_failures:
                    ftype = f['failure_type']
                    failure_type_counts[ftype] = failure_type_counts.get(ftype, 0) + 1
                
                dominant_failure = max(failure_type_counts.items(), key=lambda x: x[1])
                batch_item["dominant_failure_type"] = dominant_failure[0]
                batch_item["dominant_failure_count"] = dominant_failure[1]
                
            batch_item["failure_details"] = batch_failures
        else:
            # No failures detected
            batch_item["failures_detected"] = 0
            batch_item["failure_types"] = []
            batch_item["first_failure_step"] = None
            batch_item["max_failure_severity"] = 0.0
            batch_item["avg_failure_severity"] = 0.0
            batch_item["energy_at_first_failure"] = 0.0
            batch_item["distance_at_first_failure"] = 0.0
            batch_item["dominant_failure_type"] = None
            batch_item["dominant_failure_count"] = 0
            batch_item["failure_details"] = []
            
        batch_eval.append(batch_item)
    
    return batch_eval


# 4. UPDATE the calling functions to handle the new return values
def run_batch_planning_evaluations(
    run_id, calibrate_robot, planning_parameters, evaluation_seed, reset_split
):
    """
    Enhanced batch planning evaluations with failure recognition
    """
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
        enable_failure_recognition=planning_parameters.get('enable_failure_recognition', True)
    )

    # CHANGE this line:
    # OLD: batch_eval = evaluate_plan_batch(eval_setting_batch["target_state"], obs_arr, action_arr)
    # NEW:
    batch_eval = evaluate_plan_batch(
        eval_setting_batch["target_state"], obs_arr, action_arr, failure_details
    )
    
    # ... rest of the function stays the same ...


def run_single_planning_evaluation(
    planning_parameters,
    base_eval_setting,
    manual_context=None,
    manual_target=None,
    display=True,
):
    """
    Enhanced single planning evaluation with failure recognition
    """
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
        enable_failure_recognition=planning_parameters.get('enable_failure_recognition', True)
    )

    # CHANGE this line:
    # OLD: batch_eval = evaluate_plan_batch(eval_setting_batch["target_state"], obs_arr, action_arr)
    # NEW:
    batch_eval = evaluate_plan_batch(
        eval_setting_batch["target_state"], obs_arr, action_arr, failure_details
    )
    
    # ADD: Print failure summary for single evaluations (optional)
    if display and failure_details:
        print("\n" + "="*50)
        print("FAILURE ANALYSIS SUMMARY")
        print("="*50)
        for batch_idx, batch_failures in enumerate(failure_details):
            if batch_failures:
                print(f"\nBatch {batch_idx}: {len(batch_failures)} failures detected")
                for i, failure in enumerate(batch_failures):
                    print(f"  {i+1}. Step {failure['step']}: {failure['failure_type']} "
                          f"(severity: {failure['severity']:.3f})")
            else:
                print(f"\nBatch {batch_idx}: No failures detected")
    
    # ... rest of the function stays the same ...


# 5. ADD helper function for failure analysis (optional)
def analyze_failure_patterns(batch_results, print_summary=True):
    """
    Analyze failure patterns across batch results for insights
    """
    all_failures = []
    success_count = 0
    total_count = len(batch_results)
    
    # Collect all failures
    for result in batch_results:
        if result.get('success', False):
            success_count += 1
        
        for failure in result.get('failure_details', []):
            all_failures.append(failure)
    
    if not all_failures:
        if print_summary:
            print(f"No failures detected across {total_count} planning attempts")
        return {}
    
    # Analyze failure patterns
    failure_types = {}
    severity_by_type = {}
    step_distribution = []
    
    for failure in all_failures:
        ftype = failure['failure_type']
        failure_types[ftype] = failure_types.get(ftype, 0) + 1
        
        if ftype not in severity_by_type:
            severity_by_type[ftype] = []
        severity_by_type[ftype].append(failure.get('severity', 0.5))
        
        step_distribution.append(failure['step'])
    
    analysis = {
        'total_failures': len(all_failures),
        'success_rate': success_count / total_count,
        'failure_types': failure_types,
        'avg_severity_by_type': {
            ftype: np.mean(severities) 
            for ftype, severities in severity_by_type.items()
        },
        'avg_failure_step': np.mean(step_distribution) if step_distribution else 0,
        'early_failure_rate': sum(1 for step in step_distribution if step < 10) / len(step_distribution) if step_distribution else 0
    }
    
    if print_summary:
        print(f"\nFAILURE PATTERN ANALYSIS:")
        print(f"  Total planning attempts: {total_count}")
        print(f"  Success rate: {analysis['success_rate']:.1%}")
        print(f"  Total failures detected: {analysis['total_failures']}")
        print(f"  Most common failure types:")
        
        sorted_failures = sorted(failure_types.items(), key=lambda x: x[1], reverse=True)
        for ftype, count in sorted_failures[:5]:
            avg_severity = analysis['avg_severity_by_type'][ftype]
            print(f"    {ftype}: {count} occurrences (avg severity: {avg_severity:.3f})")
        
        if analysis['early_failure_rate'] > 0:
            print(f"  Early failure rate (< 10 steps): {analysis['early_failure_rate']:.1%}")
    
    return analysis

# Example usage in evaluation scripts:
# if __name__ == "__main__":
#     # Run planning evaluation with failure recognition enabled
#     planning_parameters = {
#         # ... existing parameters ...
#         'enable_failure_recognition': True,
#         'min_progress_rate': 0.05,  # Minimum progress rate before considering failure
#         'stuck_time_threshold': 5.0,  # Time threshold for stuck detection
#         'max_energy_rate': 0.5,  # Maximum energy consumption rate
#     }
#     
#     results = run_batch_planning_evaluations(...)
#     failure_analysis = analyze_failure_patterns(results)
