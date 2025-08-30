"""
Failure Recognition System for TRADYN AGV Navigation
==================================================

This module provides failure detection and learning capabilities for terrain-aware
autonomous ground vehicle navigation. It identifies navigation failures, stores
knowledge about what went wrong, and enables knowledge transfer between similar
terrain conditions.

The system addresses the research question:
"When the vehicle makes a non-fatal failed attempt to navigate a complex terrain,
how can the vehicle recognize the failure and make a new attempt that leverages
knowledge of past failures, and how can knowledge gained on one slope be applied
to subsequent slopes?"

Classes:
    FailureType: Enumeration of different failure categories
    TerrainContext: Represents terrain characteristics for knowledge transfer
    FailureEvent: Data structure for storing failure information
    FailureKnowledgeBase: Stores and manages failure experiences
    TerrainFailureDetector: Detects navigation failures on terrain
    FailureLearningSystem: Main system for failure recognition and learning
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from datetime import datetime
import pickle
import json


class FailureType(Enum):
    """Categorizes different types of navigation failures."""
    
    # Energy-related failures
    ENERGY_DEPLETION = "energy_depletion"           # Used too much energy
    INEFFICIENT_PATH = "inefficient_path"           # Path was wasteful
    
    # Motion-related failures  
    STUCK_POSITION = "stuck_position"               # Vehicle got stuck
    OSCILLATION = "oscillation"                     # Repeated back-and-forth motion
    POOR_PROGRESS = "poor_progress"                 # Moving too slowly toward goal
    
    # Terrain-specific failures
    SLOPE_TOO_STEEP = "slope_too_steep"            # Couldn't handle slope angle
    HIGH_FRICTION_TRAP = "high_friction_trap"      # Caught in high-friction area
    TERRAIN_MISMATCH = "terrain_mismatch"          # Terrain different than expected
    
    # Prediction failures
    DYNAMICS_MISMATCH = "dynamics_mismatch"        # Model prediction vs reality
    CONTEXT_ERROR = "context_error"                # Wrong terrain/robot context
    
    # Planning failures
    INFEASIBLE_PLAN = "infeasible_plan"           # Generated plan was impossible
    TIMEOUT = "timeout"                           # Planning took too long


@dataclass
class TerrainContext:
    """Represents terrain characteristics for similarity matching."""
    
    slope_angle: float                    # Terrain slope in degrees
    friction_coefficient: float          # Average friction coefficient
    terrain_roughness: float             # Surface roughness measure
    obstacle_density: float              # Density of obstacles
    terrain_type: str                    # e.g., "grass", "gravel", "mud"
    
    def similarity(self, other: 'TerrainContext') -> float:
        """Calculate similarity score with another terrain context (0-1)."""
        if not isinstance(other, TerrainContext):
            return 0.0
            
        # Weight different factors for similarity
        weights = {
            'slope': 0.3,
            'friction': 0.25, 
            'roughness': 0.2,
            'obstacles': 0.15,
            'type': 0.1
        }
        
        # Normalized differences (smaller = more similar)
        slope_diff = abs(self.slope_angle - other.slope_angle) / 90.0  # Max 90 degrees
        friction_diff = abs(self.friction_coefficient - other.friction_coefficient) / 2.0  # Max friction ~2
        roughness_diff = abs(self.terrain_roughness - other.roughness) / 1.0  # Normalized roughness
        obstacle_diff = abs(self.obstacle_density - other.obstacle_density) / 1.0  # Normalized density
        
        # Type similarity (exact match or not)
        type_diff = 0.0 if self.terrain_type == other.terrain_type else 1.0
        
        # Calculate weighted similarity
        similarity_score = (
            weights['slope'] * (1.0 - slope_diff) +
            weights['friction'] * (1.0 - friction_diff) +
            weights['roughness'] * (1.0 - roughness_diff) +
            weights['obstacles'] * (1.0 - obstacle_diff) +
            weights['type'] * (1.0 - type_diff)
        )
        
        return max(0.0, min(1.0, similarity_score))


@dataclass 
class RobotContext:
    """Represents robot characteristics that affect navigation."""
    
    mass: float                          # Robot mass in kg
    max_velocity: float                  # Maximum velocity capability
    max_acceleration: float              # Maximum acceleration capability
    wheel_radius: float                  # Wheel radius
    battery_level: float                 # Current battery level (0-1)


@dataclass
class FailureEvent:
    """Stores information about a navigation failure."""
    
    failure_id: str                      # Unique identifier
    timestamp: datetime                  # When failure occurred
    failure_type: FailureType           # Category of failure
    severity: float                     # Severity score (0-1)
    
    # Context information
    terrain_context: TerrainContext     # Terrain characteristics
    robot_context: RobotContext         # Robot state
    
    # Navigation details
    start_position: np.ndarray          # Starting position [x, y]
    goal_position: np.ndarray           # Target position [x, y]
    failure_position: np.ndarray        # Where failure occurred [x, y]
    planned_trajectory: List[np.ndarray] # Original planned path
    actual_trajectory: List[np.ndarray]  # Actual path taken
    
    # Failure metrics
    energy_consumed: float              # Energy used before failure
    time_elapsed: float                 # Time from start to failure
    progress_made: float                # Progress toward goal (0-1)
    
    # Learning information
    attempted_solutions: List[str] = field(default_factory=list)  # What was tried
    recovery_success: bool = False      # Whether recovery worked
    recovery_strategy: Optional[str] = None  # How recovery was achieved
    lessons_learned: List[str] = field(default_factory=list)  # Key insights


class TerrainFailureDetector:
    """Detects various types of navigation failures during terrain traversal."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize detector with configuration parameters."""
        self.config = config or {}
        self.default_thresholds = {
            'min_progress_rate': 0.1,       # Minimum progress per time unit
            'max_energy_rate': 0.8,         # Maximum energy consumption rate
            'stuck_position_threshold': 0.05, # Max position change to be "stuck"
            'stuck_time_threshold': 5.0,    # Seconds before considering stuck
            'oscillation_threshold': 0.1,   # Position variance for oscillation
            'oscillation_time_window': 10.0, # Time window to check oscillation
            'max_slope_angle': 45.0,        # Maximum navigable slope
            'prediction_error_threshold': 0.5, # Max allowable prediction error
        }
        self.thresholds = {**self.default_thresholds, **self.config}
        
        # Internal state tracking
        self.position_history: List[Tuple[float, np.ndarray]] = []  # (timestamp, position)
        self.energy_history: List[Tuple[float, float]] = []  # (timestamp, energy_used)
        self.last_check_time = 0.0
        
    def update_state(self, timestamp: float, position: np.ndarray, 
                    energy_used: float, velocity: np.ndarray = None):
        """Update internal state with new navigation data."""
        self.position_history.append((timestamp, position.copy()))
        self.energy_history.append((timestamp, energy_used))
        
        # Keep only recent history for efficiency
        max_history_time = 30.0  # Keep 30 seconds of history
        cutoff_time = timestamp - max_history_time
        
        self.position_history = [(t, p) for t, p in self.position_history if t >= cutoff_time]
        self.energy_history = [(t, e) for t, e in self.energy_history if t >= cutoff_time]
        
    def detect_stuck_position(self) -> Optional[Dict[str, Any]]:
        """Detect if vehicle is stuck in one location."""
        if len(self.position_history) < 2:
            return None
            
        # Check recent positions
        current_time = self.position_history[-1][0]
        stuck_threshold = self.thresholds['stuck_time_threshold']
        
        # Find positions within stuck time window
        recent_positions = [
            pos for t, pos in self.position_history 
            if current_time - t <= stuck_threshold
        ]
        
        if len(recent_positions) < 2:
            return None
            
        # Calculate position variance
        positions = np.array(recent_positions)
        position_variance = np.var(positions, axis=0)
        max_variance = np.max(position_variance)
        
        if max_variance < self.thresholds['stuck_position_threshold']:
            return {
                'failure_type': FailureType.STUCK_POSITION,
                'position_variance': max_variance,
                'time_stuck': stuck_threshold,
                'severity': min(1.0, stuck_threshold / 10.0)  # More severe with longer stuck time
            }
            
        return None
        
    def detect_poor_progress(self, goal_position: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect if progress toward goal is too slow."""
        if len(self.position_history) < 2:
            return None
            
        # Calculate progress rate
        start_time, start_pos = self.position_history[0]
        current_time, current_pos = self.position_history[-1]
        
        time_elapsed = current_time - start_time
        if time_elapsed < 1.0:  # Need at least 1 second of data
            return None
            
        # Distance covered and distance remaining
        distance_covered = np.linalg.norm(current_pos - start_pos)
        distance_to_goal = np.linalg.norm(goal_position - current_pos)
        total_distance = np.linalg.norm(goal_position - start_pos)
        
        progress_rate = distance_covered / time_elapsed
        expected_progress_rate = self.thresholds['min_progress_rate']
        
        if progress_rate < expected_progress_rate:
            progress_ratio = distance_covered / total_distance if total_distance > 0 else 0
            return {
                'failure_type': FailureType.POOR_PROGRESS,
                'progress_rate': progress_rate,
                'expected_rate': expected_progress_rate,
                'progress_ratio': progress_ratio,
                'severity': min(1.0, (expected_progress_rate - progress_rate) / expected_progress_rate)
            }
            
        return None
        
    def detect_oscillation(self) -> Optional[Dict[str, Any]]:
        """Detect if vehicle is oscillating back and forth."""
        if len(self.position_history) < 5:
            return None
            
        # Check positions in recent time window
        current_time = self.position_history[-1][0]
        time_window = self.thresholds['oscillation_time_window']
        
        recent_positions = [
            pos for t, pos in self.position_history 
            if current_time - t <= time_window
        ]
        
        if len(recent_positions) < 5:
            return None
            
        # Calculate position variance - high variance indicates oscillation
        positions = np.array(recent_positions)
        position_variance = np.var(positions, axis=0)
        max_variance = np.max(position_variance)
        
        # Also check for direction changes
        if len(recent_positions) >= 3:
            direction_changes = 0
            for i in range(2, len(recent_positions)):
                # Vector from position i-2 to i-1
                v1 = recent_positions[i-1] - recent_positions[i-2]
                # Vector from position i-1 to i
                v2 = recent_positions[i] - recent_positions[i-1]
                
                # Check if direction changed significantly (dot product < 0)
                if len(v1) > 0 and len(v2) > 0:
                    dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    if dot_product < 0:  # Direction change > 90 degrees
                        direction_changes += 1
                        
            oscillation_score = direction_changes / max(1, len(recent_positions) - 2)
            
            if oscillation_score > 0.3 and max_variance > self.thresholds['oscillation_threshold']:
                return {
                    'failure_type': FailureType.OSCILLATION,
                    'position_variance': max_variance,
                    'direction_changes': direction_changes,
                    'oscillation_score': oscillation_score,
                    'severity': min(1.0, oscillation_score)
                }
                
        return None
        
    def detect_energy_depletion(self, battery_level: float) -> Optional[Dict[str, Any]]:
        """Detect excessive energy consumption or depletion."""
        if len(self.energy_history) < 2:
            return None
            
        # Calculate energy consumption rate
        start_time, start_energy = self.energy_history[0]
        current_time, current_energy = self.energy_history[-1]
        
        time_elapsed = current_time - start_time
        if time_elapsed < 1.0:
            return None
            
        energy_consumed = current_energy - start_energy
        energy_rate = energy_consumed / time_elapsed
        max_energy_rate = self.thresholds['max_energy_rate']
        
        # Check for excessive energy consumption rate
        if energy_rate > max_energy_rate:
            return {
                'failure_type': FailureType.ENERGY_DEPLETION,
                'energy_rate': energy_rate,
                'max_allowed_rate': max_energy_rate,
                'battery_level': battery_level,
                'severity': min(1.0, energy_rate / max_energy_rate - 1.0)
            }
            
        # Check for critically low battery
        if battery_level < 0.1:  # Less than 10% battery
            return {
                'failure_type': FailureType.ENERGY_DEPLETION,
                'energy_rate': energy_rate,
                'battery_level': battery_level,
                'severity': min(1.0, (0.1 - battery_level) / 0.1)
            }
            
        return None
        
    def detect_terrain_failure(self, terrain_context: TerrainContext, 
                             predicted_trajectory: List[np.ndarray],
                             actual_trajectory: List[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Detect failures related to terrain characteristics."""
        failures = []
        
        # Check if slope is too steep
        if terrain_context.slope_angle > self.thresholds['max_slope_angle']:
            failures.append({
                'failure_type': FailureType.SLOPE_TOO_STEEP,
                'slope_angle': terrain_context.slope_angle,
                'max_slope': self.thresholds['max_slope_angle'],
                'severity': min(1.0, (terrain_context.slope_angle - self.thresholds['max_slope_angle']) / 45.0)
            })
            
        # Check for high friction areas causing problems
        if terrain_context.friction_coefficient > 1.5:  # High friction
            # If we're also making poor progress, it might be a friction trap
            poor_progress = self.detect_poor_progress(actual_trajectory[-1] if actual_trajectory else np.array([0, 0]))
            if poor_progress:
                failures.append({
                    'failure_type': FailureType.HIGH_FRICTION_TRAP,
                    'friction_coefficient': terrain_context.friction_coefficient,
                    'combined_with': poor_progress['failure_type'],
                    'severity': min(1.0, terrain_context.friction_coefficient / 2.0)
                })
                
        # Check prediction accuracy
        if len(predicted_trajectory) > 0 and len(actual_trajectory) > 0:
            min_length = min(len(predicted_trajectory), len(actual_trajectory))
            prediction_errors = []
            
            for i in range(min_length):
                error = np.linalg.norm(predicted_trajectory[i] - actual_trajectory[i])
                prediction_errors.append(error)
                
            avg_prediction_error = np.mean(prediction_errors)
            
            if avg_prediction_error > self.thresholds['prediction_error_threshold']:
                failures.append({
                    'failure_type': FailureType.DYNAMICS_MISMATCH,
                    'prediction_error': avg_prediction_error,
                    'max_error': np.max(prediction_errors),
                    'threshold': self.thresholds['prediction_error_threshold'],
                    'severity': min(1.0, avg_prediction_error / self.thresholds['prediction_error_threshold'])
                })
                
        return failures[0] if failures else None  # Return first detected failure
        
    def check_all_failures(self, goal_position: np.ndarray, 
                          battery_level: float,
                          terrain_context: TerrainContext,
                          predicted_trajectory: List[np.ndarray] = None,
                          actual_trajectory: List[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Check for all types of failures and return detected failures."""
        detected_failures = []
        
        # Check each failure type
        failure_checks = [
            self.detect_stuck_position(),
            self.detect_poor_progress(goal_position),
            self.detect_oscillation(),
            self.detect_energy_depletion(battery_level),
        ]
        
        # Add terrain-specific checks if trajectories provided
        if predicted_trajectory and actual_trajectory:
            terrain_failure = self.detect_terrain_failure(
                terrain_context, predicted_trajectory, actual_trajectory
            )
            if terrain_failure:
                failure_checks.append(terrain_failure)
                
        # Collect non-None failures
        detected_failures = [f for f in failure_checks if f is not None]
        
        return detected_failures


class FailureKnowledgeBase:
    """Stores and manages knowledge about navigation failures."""
    
    def __init__(self, storage_path: str = "failure_knowledge.pkl"):
        """Initialize knowledge base with optional persistent storage."""
        self.storage_path = storage_path
        self.failure_events: List[FailureEvent] = []
        self.terrain_patterns: Dict[str, List[str]] = {}  # terrain_type -> common failure types
        self.recovery_strategies: Dict[str, List[str]] = {}  # failure_type -> recovery strategies
        self.load_knowledge()
        
    def add_failure_event(self, failure_event: FailureEvent):
        """Add a new failure event to the knowledge base."""
        self.failure_events.append(failure_event)
        
        # Update patterns
        terrain_type = failure_event.terrain_context.terrain_type
        failure_type = failure_event.failure_type.value
        
        if terrain_type not in self.terrain_patterns:
            self.terrain_patterns[terrain_type] = []
        if failure_type not in self.terrain_patterns[terrain_type]:
            self.terrain_patterns[terrain_type].append(failure_type)
            
        # Update recovery strategies if recovery was successful
        if failure_event.recovery_success and failure_event.recovery_strategy:
            if failure_type not in self.recovery_strategies:
                self.recovery_strategies[failure_type] = []
            if failure_event.recovery_strategy not in self.recovery_strategies[failure_type]:
                self.recovery_strategies[failure_type].append(failure_event.recovery_strategy)
                
        self.save_knowledge()
        
    def find_similar_terrains(self, terrain_context: TerrainContext, 
                             min_similarity: float = 0.7) -> List[FailureEvent]:
        """Find failure events from similar terrain contexts."""
        similar_failures = []
        
        for event in self.failure_events:
            similarity = terrain_context.similarity(event.terrain_context)
            if similarity >= min_similarity:
                similar_failures.append((event, similarity))
                
        # Sort by similarity (highest first)
        similar_failures.sort(key=lambda x: x[1], reverse=True)
        return [event for event, similarity in similar_failures]
        
    def get_failure_patterns(self, terrain_type: str) -> List[str]:
        """Get common failure types for a terrain type."""
        return self.terrain_patterns.get(terrain_type, [])
        
    def get_recovery_strategies(self, failure_type: FailureType) -> List[str]:
        """Get known recovery strategies for a failure type."""
        return self.recovery_strategies.get(failure_type.value, [])
        
    def get_lessons_learned(self, terrain_context: TerrainContext, 
                           failure_type: FailureType = None) -> List[str]:
        """Get lessons learned from similar terrain and/or failure type."""
        lessons = []
        
        # Find similar terrain experiences
        similar_events = self.find_similar_terrains(terrain_context)
        
        for event in similar_events:
            if failure_type is None or event.failure_type == failure_type:
                lessons.extend(event.lessons_learned)
                
        # Remove duplicates while preserving order
        unique_lessons = []
        for lesson in lessons:
            if lesson not in unique_lessons:
                unique_lessons.append(lesson)
                
        return unique_lessons
        
    def save_knowledge(self):
        """Save knowledge base to persistent storage."""
        try:
            with open(self.storage_path, 'wb') as f:
                data = {
                    'failure_events': self.failure_events,
                    'terrain_patterns': self.terrain_patterns,
                    'recovery_strategies': self.recovery_strategies
                }
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Could not save knowledge base: {e}")
            
    def load_knowledge(self):
        """Load knowledge base from persistent storage."""
        try:
            with open(self.storage_path, 'rb') as f:
                data = pickle.load(f)
                self.failure_events = data.get('failure_events', [])
                self.terrain_patterns = data.get('terrain_patterns', {})
                self.recovery_strategies = data.get('recovery_strategies', {})
        except FileNotFoundError:
            # First time running, no saved knowledge yet
            pass
        except Exception as e:
            print(f"Warning: Could not load knowledge base: {e}")


class FailureLearningSystem:
    """Main system for failure recognition and learning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the failure learning system."""
        self.config = config or {}
        self.detector = TerrainFailureDetector(config)
        self.knowledge_base = FailureKnowledgeBase(
            storage_path=config.get('knowledge_storage_path', 'failure_knowledge.pkl')
        )
        self.current_navigation_id = None
        self.current_failure_event = None
        
    def get_terrain_insights(self, terrain_context: TerrainContext) -> Dict[str, Any]:
        """Get insights about navigating similar terrain types."""
        insights = {
            'similar_experiences': 0,
            'common_failures': [],
            'success_rate': 0.0,
            'recommended_approaches': [],
            'risk_factors': []
        }
        
        # Find similar terrain experiences
        similar_events = self.knowledge_base.find_similar_terrains(terrain_context, min_similarity=0.6)
        insights['similar_experiences'] = len(similar_events)
        
        if similar_events:
            # Analyze common failures
            failure_counts = {}
            successful_recoveries = 0
            
            for event in similar_events:
                failure_type = event.failure_type.value
                failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
                if event.recovery_success:
                    successful_recoveries += 1
                    
            # Sort failures by frequency
            insights['common_failures'] = sorted(
                failure_counts.items(), key=lambda x: x[1], reverse=True
            )
            
            # Calculate success rate
            insights['success_rate'] = successful_recoveries / len(similar_events)
            
            # Get recommended approaches from successful cases
            for event in similar_events:
                if event.recovery_success and event.recovery_strategy:
                    if event.recovery_strategy not in insights['recommended_approaches']:
                        insights['recommended_approaches'].append(event.recovery_strategy)
                        
            # Identify risk factors
            if terrain_context.slope_angle > 30:
                insights['risk_factors'].append("Steep slope - risk of sliding")
            if terrain_context.friction_coefficient > 1.2:
                insights['risk_factors'].append("High friction - risk of getting stuck")
            if terrain_context.obstacle_density > 0.3:
                insights['risk_factors'].append("Dense obstacles - risk of collision")
                
        return insights
        
    def export_knowledge_summary(self, filepath: str = "failure_knowledge_summary.json"):
        """Export a human-readable summary of learned knowledge."""
        summary = {
            'total_experiences': len(self.knowledge_base.failure_events),
            'terrain_patterns': self.knowledge_base.terrain_patterns,
            'recovery_strategies': self.knowledge_base.recovery_strategies,
            'failure_statistics': {},
            'terrain_statistics': {},
            'recent_lessons': []
        }
        
        # Calculate failure statistics
        failure_counts = {}
        severity_by_type = {}
        
        for event in self.knowledge_base.failure_events:
            failure_type = event.failure_type.value
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
            
            if failure_type not in severity_by_type:
                severity_by_type[failure_type] = []
            severity_by_type[failure_type].append(event.severity)
            
        for failure_type, count in failure_counts.items():
            avg_severity = np.mean(severity_by_type[failure_type])
            summary['failure_statistics'][failure_type] = {
                'count': count,
                'average_severity': float(avg_severity),
                'frequency': count / len(self.knowledge_base.failure_events)
            }
            
        # Calculate terrain statistics
        terrain_counts = {}
        for event in self.knowledge_base.failure_events:
            terrain_type = event.terrain_context.terrain_type
            terrain_counts[terrain_type] = terrain_counts.get(terrain_type, 0) + 1
            
        summary['terrain_statistics'] = terrain_counts
        
        # Get recent lessons (last 10 unique lessons)
        recent_lessons = []
        for event in reversed(self.knowledge_base.failure_events[-50:]):  # Last 50 events
            for lesson in event.lessons_learned:
                if lesson not in recent_lessons:
                    recent_lessons.append(lesson)
                    if len(recent_lessons) >= 10:
                        break
            if len(recent_lessons) >= 10:
                break
                
        summary['recent_lessons'] = recent_lessons
        
        # Save summary
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Knowledge summary exported to {filepath}")
        except Exception as e:
            print(f"Could not export summary: {e}")
            
        return summary


# Example usage and integration functions
def create_terrain_context_from_data(terrain_data: Dict[str, Any]) -> TerrainContext:
    """Helper function to create TerrainContext from terrain analysis data."""
    return TerrainContext(
        slope_angle=terrain_data.get('slope_angle', 0.0),
        friction_coefficient=terrain_data.get('friction_coefficient', 0.7),
        terrain_roughness=terrain_data.get('roughness', 0.1),
        obstacle_density=terrain_data.get('obstacle_density', 0.0),
        terrain_type=terrain_data.get('terrain_type', 'unknown')
    )


def create_robot_context_from_state(robot_state: Dict[str, Any]) -> RobotContext:
    """Helper function to create RobotContext from robot state data."""
    return RobotContext(
        mass=robot_state.get('mass', 20.0),
        max_velocity=robot_state.get('max_velocity', 2.0),
        max_acceleration=robot_state.get('max_acceleration', 1.0),
        wheel_radius=robot_state.get('wheel_radius', 0.1),
        battery_level=robot_state.get('battery_level', 1.0)
    )failure_event = None
        
    def start_navigation(self, navigation_id: str, start_pos: np.ndarray, 
                        goal_pos: np.ndarray, terrain_context: TerrainContext,
                        robot_context: RobotContext):
        """Start monitoring a new navigation attempt."""
        self.current_navigation_id = navigation_id
        self.detector.position_history.clear()
        self.detector.energy_history.clear()
        
        # Check for lessons from similar terrains
        similar_events = self.knowledge_base.find_similar_terrains(terrain_context)
        
        if similar_events:
            print(f"Found {len(similar_events)} similar terrain experiences:")
            for event in similar_events[:3]:  # Show top 3
                similarity = terrain_context.similarity(event.terrain_context)
                print(f"  - {event.failure_type.value} (similarity: {similarity:.2f})")
                
            # Get relevant lessons
            lessons = self.knowledge_base.get_lessons_learned(terrain_context)
            if lessons:
                print("Lessons learned:")
                for lesson in lessons[:5]:  # Show top 5 lessons
                    print(f"  - {lesson}")
                    
        return similar_events, self.knowledge_base.get_lessons_learned(terrain_context)
        
    def update_navigation_state(self, timestamp: float, position: np.ndarray,
                               energy_used: float, velocity: np.ndarray = None,
                               battery_level: float = 1.0):
        """Update the current navigation state for failure detection."""
        self.detector.update_state(timestamp, position, energy_used, velocity)
        
    def check_for_failures(self, goal_position: np.ndarray, battery_level: float,
                          terrain_context: TerrainContext,
                          predicted_trajectory: List[np.ndarray] = None,
                          actual_trajectory: List[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Check for navigation failures."""
        return self.detector.check_all_failures(
            goal_position, battery_level, terrain_context, 
            predicted_trajectory, actual_trajectory
        )
        
    def record_failure(self, failure_info: Dict[str, Any], 
                      terrain_context: TerrainContext,
                      robot_context: RobotContext,
                      start_pos: np.ndarray, goal_pos: np.ndarray,
                      failure_pos: np.ndarray,
                      planned_trajectory: List[np.ndarray],
                      actual_trajectory: List[np.ndarray],
                      energy_consumed: float, time_elapsed: float) -> FailureEvent:
        """Record a failure event for learning."""
        
        # Calculate progress made
        total_distance = np.linalg.norm(goal_pos - start_pos)
        progress_distance = np.linalg.norm(failure_pos - start_pos)
        progress_made = progress_distance / total_distance if total_distance > 0 else 0
        
        failure_event = FailureEvent(
            failure_id=f"{self.current_navigation_id}_failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            failure_type=FailureType(failure_info['failure_type']),
            severity=failure_info.get('severity', 0.5),
            terrain_context=terrain_context,
            robot_context=robot_context,
            start_position=start_pos,
            goal_position=goal_pos,
            failure_position=failure_pos,
            planned_trajectory=planned_trajectory,
            actual_trajectory=actual_trajectory,
            energy_consumed=energy_consumed,
            time_elapsed=time_elapsed,
            progress_made=progress_made
        )
        
        self.current_failure_event = failure_event
        return failure_event
        
    def suggest_recovery_strategies(self, failure_type: FailureType, 
                                   terrain_context: TerrainContext) -> List[str]:
        """Suggest recovery strategies based on past experiences."""
        strategies = []
        
        # Get general recovery strategies for this failure type
        general_strategies = self.knowledge_base.get_recovery_strategies(failure_type)
        strategies.extend(general_strategies)
        
        # Get strategies from similar terrains
        similar_events = self.knowledge_base.find_similar_terrains(terrain_context)
        for event in similar_events:
            if event.failure_type == failure_type and event.recovery_success:
                if event.recovery_strategy and event.recovery_strategy not in strategies:
                    strategies.append(event.recovery_strategy)
                    
        # Add default strategies if none found
        if not strategies:
            default_strategies = {
                FailureType.STUCK_POSITION: ["Try different path", "Increase power", "Backup and retry"],
                FailureType.ENERGY_DEPLETION: ["Find more efficient path", "Reduce speed", "Take breaks"],
                FailureType.SLOPE_TOO_STEEP: ["Find alternative route", "Approach at angle", "Use momentum"],
                FailureType.HIGH_FRICTION_TRAP: ["Increase power", "Find alternative path", "Use different trajectory"],
                FailureType.POOR_PROGRESS: ["Increase speed", "Check for obstacles", "Re-plan route"],
                FailureType.OSCILLATION: ["Reduce control gains", "Smooth trajectory", "Check sensor calibration"]
            }
            strategies.extend(default_strategies.get(failure_type, ["Re-plan and retry"]))
            
        return strategies
        
    def record_recovery_attempt(self, recovery_strategy: str, success: bool,
                               lessons_learned: List[str] = None):
        """Record the outcome of a recovery attempt."""
        if self.current_failure_event:
            self.current_failure_event.attempted_solutions.append(recovery_strategy)
            self.current_failure_event.recovery_success = success
            if success:
                self.current_failure_event.recovery_strategy = recovery_strategy
            if lessons_learned:
                self.current_failure_event.lessons_learned.extend(lessons_learned)
                
            # Add to knowledge base if recovery succeeded or we learned something
            if success or lessons_learned:
                self.knowledge_base.add_failure_event(self.current_failure_event)
                
    def finish_navigation(self, success: bool):
        """Finish monitoring the current navigation attempt."""
        if not success and self.current_failure_event:
            # Even if we didn't recover, we can still learn
            self.knowledge_base.add_failure_event(self.current_failure_event)
            
        self.current_navigation_id = None
        self.current_
