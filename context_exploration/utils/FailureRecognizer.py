# Step 1: Failure Recognition System
# Add this as: context_exploration/failure_recognition.py

import numpy as np
from collections import deque
from enum import Enum
from typing import Tuple, Optional, List

class FailureType(Enum):
    STUCK = "stuck"
    HIGH_ENERGY = "high_energy"
    POOR_PROGRESS = "poor_progress"
    OSCILLATION = "oscillation"
    TERRAIN_MISMATCH = "terrain_mismatch"

class FailureRecognizer:
    """
    Recognizes non-fatal navigation failures during terrain traversal.
    
    This system monitors the vehicle's performance in real-time and identifies
    when the current navigation strategy is failing, allowing for adaptive
    re-planning before complete mission failure.
    """
    
    def __init__(self, 
                 stuck_velocity_threshold=0.1,      # m/s - below this is considered stuck
                 stuck_time_window=5,               # seconds to check for being stuck
                 energy_threshold_multiplier=2.5,   # multiplier for excessive energy
                 progress_time_window=10,           # seconds to check progress
                 oscillation_threshold=0.5):        # radians for oscillation detection
        
        # Thresholds for different failure types
        self.stuck_velocity_threshold = stuck_velocity_threshold
        self.stuck_time_window = stuck_time_window
        self.energy_threshold_multiplier = energy_threshold_multiplier
        self.progress_time_window = progress_time_window
        self.oscillation_threshold = oscillation_threshold
        
        # Historical data for analysis
        self.velocity_history = deque(maxlen=int(stuck_time_window * 10))  # Assume 10Hz
        self.energy_history = deque(maxlen=int(progress_time_window * 10))
        self.position_history = deque(maxlen=int(progress_time_window * 10))
        self.heading_history = deque(maxlen=20)  # For oscillation detection
        
        # Goal tracking
        self.initial_goal_distance = None
        self.best_goal_distance = float('inf')
        
    def update_state(self, robot_state, action, goal_position):
        """Update the recognizer with current robot state and action"""
        
        # Update histories
        velocity_magnitude = np.linalg.norm(robot_state.velocity)
        self.velocity_history.append(velocity_magnitude)
        
        # Calculate energy from action (throttle command)
        if hasattr(action, 'throttle'):
            energy = np.linalg.norm(action.throttle) ** 2
        else:
            energy = np.linalg.norm(action) ** 2  # Assume action is throttle
        self.energy_history.append(energy)
        
        # Update position and goal distance
        current_position = robot_state.position
        self.position_history.append(current_position)
        
        goal_distance = np.linalg.norm(current_position - goal_position)
        if self.initial_goal_distance is None:
            self.initial_goal_distance = goal_distance
        self.best_goal_distance = min(self.best_goal_distance, goal_distance)
        
        # Update heading history
        if hasattr(robot_state, 'heading'):
            self.heading_history.append(robot_state.heading)
    
    def check_for_failure(self, robot_state, action, goal_position, 
                         terrain_context) -> Tuple[bool, Optional[FailureType], dict]:
        """
        Main failure detection method.
        
        Returns:
            (is_failure, failure_type, failure_details)
        """
        
        # Update state first
        self.update_state(robot_state, action, goal_position)
        
        # Check each failure type
        failure_checks = [
            self._check_stuck_failure(),
            self._check_energy_failure(terrain_context),
            self._check_progress_failure(goal_position),
            self._check_oscillation_failure(),
            self._check_terrain_mismatch_failure(robot_state, terrain_context)
        ]
        
        # Return first detected failure
        for is_failure, failure_type, details in failure_checks:
            if is_failure:
                return True, failure_type, details
                
        return False, None, {}
    
    def _check_stuck_failure(self) -> Tuple[bool, Optional[FailureType], dict]:
        """Detect if vehicle is stuck (very low velocity for extended time)"""
        
        if len(self.velocity_history) < self.stuck_time_window * 5:  # Need enough history
            return False, None, {}
        
        recent_velocities = list(self.velocity_history)[-int(self.stuck_time_window * 5):]
        avg_velocity = np.mean(recent_velocities)
        max_velocity = np.max(recent_velocities)
        
        is_stuck = (avg_velocity < self.stuck_velocity_threshold and 
                   max_velocity < self.stuck_velocity_threshold * 2)
        
        details = {
            'avg_velocity': avg_velocity,
            'velocity_threshold': self.stuck_velocity_threshold,
            'time_window': self.stuck_time_window
        }
        
        return is_stuck, FailureType.STUCK if is_stuck else None, details
    
    def _check_energy_failure(self, terrain_context) -> Tuple[bool, Optional[FailureType], dict]:
        """Detect excessive energy consumption for the terrain type"""
        
        if len(self.energy_history) < 10:  # Need some history
            return False, None, {}
        
        recent_energy = list(self.energy_history)[-10:]
        avg_energy = np.mean(recent_energy)
        
        # Estimate expected energy based on terrain characteristics
        expected_energy = self._estimate_expected_energy(terrain_context)
        energy_threshold = expected_energy * self.energy_threshold_multiplier
        
        is_high_energy = avg_energy > energy_threshold
        
        details = {
            'actual_energy': avg_energy,
            'expected_energy': expected_energy,
            'threshold': energy_threshold,
            'multiplier': self.energy_threshold_multiplier
        }
        
        return is_high_energy, FailureType.HIGH_ENERGY if is_high_energy else None, details
    
    def _check_progress_failure(self, goal_position) -> Tuple[bool, Optional[FailureType], dict]:
        """Detect lack of progress toward goal"""
        
        if len(self.position_history) < self.progress_time_window * 5:
            return False, None, {}
        
        # Check if we've made reasonable progress
        current_position = self.position_history[-1]
        past_position = self.position_history[-int(self.progress_time_window * 5)]
        
        current_goal_distance = np.linalg.norm(current_position - goal_position)
        past_goal_distance = np.linalg.norm(past_position - goal_position)
        
        # We should have made some progress
        progress_made = past_goal_distance - current_goal_distance
        expected_minimum_progress = 0.1  # meters - should make at least this much progress
        
        is_poor_progress = progress_made < expected_minimum_progress
        
        details = {
            'progress_made': progress_made,
            'expected_minimum': expected_minimum_progress,
            'current_distance': current_goal_distance,
            'best_distance': self.best_goal_distance
        }
        
        return is_poor_progress, FailureType.POOR_PROGRESS if is_poor_progress else None, details
    
    def _check_oscillation_failure(self) -> Tuple[bool, Optional[FailureType], dict]:
        """Detect harmful oscillatory behavior in heading"""
        
        if len(self.heading_history) < 10:
            return False, None, {}
        
        recent_headings = list(self.heading_history)[-10:]
        heading_changes = np.diff(recent_headings)
        
        # Handle angle wraparound
        heading_changes = np.array([self._normalize_angle(change) for change in heading_changes])
        
        # Check for rapid, large heading changes (oscillation)
        heading_variance = np.var(heading_changes)
        is_oscillating = heading_variance > self.oscillation_threshold
        
        details = {
            'heading_variance': heading_variance,
            'threshold': self.oscillation_threshold,
            'recent_changes': heading_changes.tolist()
        }
        
        return is_oscillating, FailureType.OSCILLATION if is_oscillating else None, details
    
    def _check_terrain_mismatch_failure(self, robot_state, terrain_context) -> Tuple[bool, Optional[FailureType], dict]:
        """Detect when robot behavior doesn't match terrain expectations"""
        
        # This is more advanced - checks if the robot's response matches
        # what we'd expect for the given terrain
        
        if not hasattr(terrain_context, 'friction') or len(self.velocity_history) < 5:
            return False, None, {}
        
        # Get local terrain properties
        local_friction = self._get_local_terrain_friction(robot_state.position, terrain_context)
        
        # Check if velocity response matches terrain
        recent_velocities = list(self.velocity_history)[-5:]
        expected_velocity = self._estimate_expected_velocity(local_friction)
        actual_velocity = np.mean(recent_velocities)
        
        # If actual velocity is much lower than expected, might indicate terrain mismatch
        velocity_ratio = actual_velocity / max(expected_velocity, 0.1)
        is_mismatch = velocity_ratio < 0.5  # Actual much lower than expected
        
        details = {
            'expected_velocity': expected_velocity,
            'actual_velocity': actual_velocity,
            'velocity_ratio': velocity_ratio,
            'local_friction': local_friction
        }
        
        return is_mismatch, FailureType.TERRAIN_MISMATCH if is_mismatch else None, details
    
    def _estimate_expected_energy(self, terrain_context):
        """Estimate expected energy consumption for terrain type"""
        
        # Simple heuristic - adjust based on your terrain data structure
        base_energy = 1.0
        
        if hasattr(terrain_context, 'friction'):
            # Higher friction = more energy needed
            avg_friction = np.mean(terrain_context.friction)
            friction_multiplier = 1.0 / max(avg_friction, 0.1)
            base_energy *= friction_multiplier
        
        if hasattr(terrain_context, 'elevation'):
            # Steeper slopes = more energy
            elevation_variance = np.var(terrain_context.elevation)
            slope_multiplier = 1.0 + elevation_variance * 0.5
            base_energy *= slope_multiplier
            
        return base_energy
    
    def _get_local_terrain_friction(self, position, terrain_context):
        """Get friction value at robot's current position"""
        if hasattr(terrain_context, 'friction'):
            # Simple nearest neighbor - adjust based on your data structure
            return np.mean(terrain_context.friction)  # Placeholder
        return 1.0
    
    def _estimate_expected_velocity(self, friction):
        """Estimate expected velocity for given friction"""
        # Simple model: velocity inversely related to friction
        return max(1.0 / friction, 0.1)
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def reset(self):
        """Reset the recognizer for a new navigation episode"""
        self.velocity_history.clear()
        self.energy_history.clear()
        self.position_history.clear()
        self.heading_history.clear()
        self.initial_goal_distance = None
        self.best_goal_distance = float('inf')

# Simple test function to verify the failure recognizer works
def test_failure_recognizer():
    """Test the failure recognition system with sample data"""
    
    recognizer = FailureRecognizer()
    
    # Simulate a stuck scenario
    print("Testing stuck detection:")
    for i in range(60):  # 6 seconds at 10Hz
        # Simulate stuck robot (very low velocity)
        mock_state = type('State', (), {
            'velocity': np.array([0.05, 0.02]),  # Very low velocity
            'position': np.array([1.0 + i*0.001, 2.0]),  # Barely moving
            'heading': 0.0
        })()
        
        mock_action = type('Action', (), {'throttle': np.array([0.8, 0.9])})()  # High throttle
        mock_goal = np.array([5.0, 5.0])
        mock_terrain = type('Terrain', (), {'friction': np.array([0.8])})()
        
        is_failure, failure_type, details = recognizer.check_for_failure(
            mock_state, mock_action, mock_goal, mock_terrain
        )
        
        if is_failure:
            print(f"  Failure detected at step {i}: {failure_type}")
            print(f"  Details: {details}")
            break
    
    print("\nFailure recognition test complete!")

if __name__ == "__main__":
    test_failure_recognizer()
