"""
Failure Recognition System for TRADYN AGV Navigation - Minimal Working Version
============================================================================

This is a simplified but complete version that addresses your research question
without optional advanced features.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import pickle
import json


class FailureType(Enum):
    """Categorizes different types of failures that can occur."""
    
    # Energy-related failures
    ENERGY_DEPLETION = "energy_depletion"
    INEFFICIENT_PATH = "inefficient_path"
    
    # Motion-related failures  
    STUCK_POSITION = "stuck_position"
    OSCILLATION = "oscillation"
    POOR_PROGRESS = "poor_progress"
    
    # Terrain-specific failures
    SLOPE_TOO_STEEP = "slope_too_steep"
    HIGH_FRICTION_TRAP = "high_friction_trap"
    TERRAIN_MISMATCH = "terrain_mismatch"
    
    # Prediction failures
    DYNAMICS_MISMATCH = "dynamics_mismatch"
    CONTEXT_ERROR = "context_error"
    
    # Planning failures
    INFEASIBLE_PLAN = "infeasible_plan"
    TIMEOUT = "timeout"


@dataclass
class TerrainContext:
    """Represents terrain characteristics for similarity matching."""
    
    slope_angle: float
    friction_coefficient: float
    terrain_roughness: float
    obstacle_density: float
    terrain_type: str
    
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
        slope_diff = abs(self.slope_angle - other.slope_angle) / 90.0
        friction_diff = abs(self.friction_coefficient - other.friction_coefficient) / 2.0
        roughness_diff = abs(self.terrain_roughness - other.terrain_roughness) / 1.0
        obstacle_diff = abs(self.obstacle_density - other.obstacle_density) / 1.0
        
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
    
    mass: float
    max_velocity: float
    max_acceleration: float
    wheel_radius: float
    battery_level: float


@dataclass
class FailureEvent:
    """Stores information about a navigation failure."""
    
    failure_id: str
    timestamp: datetime
    failure_type: FailureType
    severity: float
    
    # Context information
    terrain_context: TerrainContext
    robot_context: RobotContext
    
    # Navigation details
    start_position: np.ndarray
    goal_position: np.ndarray
    failure_position: np.ndarray
    planned_trajectory: List[np.ndarray]
    actual_trajectory: List[np.ndarray]
    
    # Failure metrics
    energy_consumed: float
    time_elapsed: float
    progress_made: float
    
    # Learning information
    attempted_solutions: List[str] = field(default_factory=list)
    recovery_success: bool = False
    recovery_strategy: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)


class TerrainFailureDetector:
    """Detects various types of navigation failures during terrain traversal."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize detector with configuration parameters."""
        self.config = config or {}
        self.default_thresholds = {
            'min_progress_rate': 0.1,
            'max_energy_rate': 0.8,
            'stuck_position_threshold': 0.05,
            'stuck_time_threshold': 5.0,
            'oscillation_threshold': 0.1,
            'oscillation_time_window': 10.0,
            'max_slope_angle': 45.0,
            'prediction_error_threshold': 0.5,
        }
        self.thresholds = {**self.default_thresholds, **self.config}
        
        # Internal state tracking
        self.position_history: List[Tuple[float, np.ndarray]] = []
        self.energy_history: List[Tuple[float, float]] = []
        
    def update_state(self, timestamp: float, position: np.ndarray, 
                    energy_used: float, velocity: np.ndarray = None):
        """Update internal state with new navigation data."""
        self.position_history.append((timestamp, position.copy()))
        self.energy_history.append((timestamp, energy_used))
        
        # Keep only recent history for efficiency
        max_history_time = 30.0
        cutoff_time = timestamp - max_history_time
        
        self.position_history = [(t, p) for t, p in self.position_history if t >= cutoff_time]
        self.energy_history = [(t, e) for t, e in self.energy_history if t >= cutoff_time]
        
    def detect_stuck_position(self) -> Optional[Dict[str, Any]]:
        """Detect if vehicle is stuck in one location."""
        if len(self.position_history) < 2:
            return None
            
        current_time = self.position_history[-1][0]
        stuck_threshold = self.thresholds['stuck_time_threshold']
        
        recent_positions = [
            pos for t, pos in self.position_history 
            if current_time - t <= stuck_threshold
        ]
        
        if len(recent_positions) < 2:
            return None
            
        positions = np.array(recent_positions)
        position_variance = np.var(positions, axis=0)
        max_variance = np.max(position_variance)
        
        if max_variance < self.thresholds['stuck_position_threshold']:
            return {
                'failure_type': FailureType.STUCK_POSITION,
                'position_variance': max_variance,
                'time_stuck': stuck_threshold,
                'severity': min(1.0, stuck_threshold / 10.0)
            }
            
        return None
        
    def detect_poor_progress(self, goal_position: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect if progress toward goal is too slow."""
        if len(self.position_history) < 2:
            return None
            
        start_time, start_pos = self.position_history[0]
        current_time, current_pos = self.position_history[-1]
        
        time_elapsed = current_time - start_time
        if time_elapsed < 1.0:
            return None
            
        distance_covered = np.linalg.norm(current_pos - start_pos)
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
        
    def detect_energy_depletion(self, battery_level: float) -> Optional[Dict[str, Any]]:
        """Detect excessive energy consumption or depletion."""
        if len(self.energy_history) < 2:
            return None
            
        start_time, start_energy = self.energy_history[0]
        current_time, current_energy = self.energy_history[-1]
        
        time_elapsed = current_time - start_time
        if time_elapsed < 1.0:
            return None
            
        energy_consumed = current_energy - start_energy
        energy_rate = energy_consumed / time_elapsed
        max_energy_rate = self.thresholds['max_energy_rate']
        
        if energy_rate > max_energy_rate:
            return {
                'failure_type': FailureType.ENERGY_DEPLETION,
                'energy_rate': energy_rate,
                'max_allowed_rate': max_energy_rate,
                'battery_level': battery_level,
                'severity': min(1.0, energy_rate / max_energy_rate - 1.0)
            }
            
        if battery_level < 0.1:
            return {
                'failure_type': FailureType.ENERGY_DEPLETION,
                'energy_rate': energy_rate,
                'battery_level': battery_level,
                'severity': min(1.0, (0.1 - battery_level) / 0.1)
            }
            
        return None
        
    def check_all_failures(self, goal_position: np.ndarray, 
                          battery_level: float,
                          terrain_context: TerrainContext,
                          predicted_trajectory: List[np.ndarray] = None,
                          actual_trajectory: List[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Check for all types of failures and return detected failures."""
        detected_failures = []
        
        failure_checks = [
            self.detect_stuck_position(),
            self.detect_poor_progress(goal_position),
            self.detect_energy_depletion(battery_level),
        ]
        
        detected_failures = [f for f in failure_checks if f is not None]
        
        return detected_failures


class FailureKnowledgeBase:
    """Stores and manages knowledge about navigation failures."""
    
    def __init__(self, storage_path: str = "failure_knowledge.pkl"):
        """Initialize knowledge base with optional persistent storage."""
        self.storage_path = storage_path
        self.failure_events: List[FailureEvent] = []
        self.terrain_patterns: Dict[str, List[str]] = {}
        self.recovery_strategies: Dict[str, List[str]] = {}
        self.load_knowledge()
        
    def add_failure_event(self, failure_event: FailureEvent):
        """Add a new failure event to the knowledge base."""
        self.failure_events.append(failure_event)
        
        terrain_type = failure_event.terrain_context.terrain_type
        failure_type = failure_event.failure_type.value
        
        if terrain_type not in self.terrain_patterns:
            self.terrain_patterns[terrain_type] = []
        if failure_type not in self.terrain_patterns[terrain_type]:
            self.terrain_patterns[terrain_type].append(failure_type)
            
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
        
        similar_events = self.find_similar_terrains(terrain_context)
        
        for event in similar_events:
            if failure_type is None or event.failure_type == failure_type:
                lessons.extend(event.lessons_learned)
                
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
        
    def start_navigation(self, navigation_id: str, start_pos: np.ndarray, 
                        goal_pos: np.ndarray, terrain_context: TerrainContext,
                        robot_context: RobotContext):
        """Start monitoring a new navigation attempt."""
        self.current_navigation_id = navigation_id
        self.detector.position_history.clear()
        self.detector.energy_history.clear()
        
        similar_events = self.knowledge_base.find_similar_terrains(terrain_context)
        lessons = self.knowledge_base.get_lessons_learned(terrain_context)
        
        return similar_events, lessons
        
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
        
        total_distance = np.linalg.norm(goal_pos - start_pos)
        progress_distance = np.linalg.norm(failure_pos - start_pos)
        progress_made = progress_distance / total_distance if total_distance > 0 else 0
        
        failure_event = FailureEvent(
            failure_id=f"{self.current_navigation_id}_failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            failure_type=failure_info['failure_type'],
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
        
        general_strategies = self.knowledge_base.get_recovery_strategies(failure_type)
        strategies.extend(general_strategies)
        
        similar_events = self.knowledge_base.find_similar_terrains(terrain_context)
        for event in similar_events:
            if event.failure_type == failure_type and event.recovery_success:
                if event.recovery_strategy and event.recovery_strategy not in strategies:
                    strategies.append(event.recovery_strategy)
                    
        if not strategies:
            default_strategies = {
                FailureType.STUCK_POSITION: ["Try different path", "Increase power", "Backup and retry"],
                FailureType.ENERGY_DEPLETION: ["Find more efficient path", "Reduce speed", "Take breaks"],
                FailureType.SLOPE_TOO_STEEP: ["Find alternative route", "Approach at angle", "Use momentum"],
                FailureType.HIGH_FRICTION_TRAP: ["Increase power", "Find alternative path", "Use different trajectory"],
                FailureType.POOR_PROGRESS: ["Increase speed", "Check for obstacles", "Re-plan route"],
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
                
            if success or lessons_learned:
                self.knowledge_base.add_failure_event(self.current_failure_event)
                
    def finish_navigation(self, success: bool):
        """Finish monitoring the current navigation attempt."""
        if not success and self.current_failure_event:
            self.knowledge_base.add_failure_event(self.current_failure_event)
            
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
        
        similar_events = self.knowledge_base.find_similar_terrains(terrain_context, min_similarity=0.6)
        insights['similar_experiences'] = len(similar_events)
        
        if similar_events:
            failure_counts = {}
            successful_recoveries = 0
            
            for event in similar_events:
                failure_type = event.failure_type.value
                failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
                if event.recovery_success:
                    successful_recoveries += 1
                    
            insights['common_failures'] = sorted(
                failure_counts.items(), key=lambda x: x[1], reverse=True
            )
            
            insights['success_rate'] = successful_recoveries / len(similar_events)
            
            for event in similar_events:
                if event.recovery_success and event.recovery_strategy:
                    if event.recovery_strategy not in insights['recommended_approaches']:
                        insights['recommended_approaches'].append(event.recovery_strategy)
                        
            if terrain_context.slope_angle > 30:
                insights['risk_factors'].append("Steep slope - risk of sliding")
            if terrain_context.friction_coefficient > 1.2:
                insights['risk_factors'].append("High friction - risk of getting stuck")
            if terrain_context.obstacle_density > 0.3:
                insights['risk_factors'].append("Dense obstacles - risk of collision")
                
        return insights


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
    )


if __name__ == "__main__":
    print("TRADYN AGV Failure Recognition System - Minimal Version")
    print("=" * 60)
    
    config = {
        'min_progress_rate': 0.05,
        'stuck_time_threshold': 3.0,
        'knowledge_storage_path': 'tradyn_failure_knowledge.pkl'
    }
    
    failure_system = FailureLearningSystem(config)
    
    steep_grass_slope = TerrainContext(
        slope_angle=35.0,
        friction_coefficient=0.8,
        terrain_roughness=0.3,
        obstacle_density=0.1,
        terrain_type="grass"
    )
    
    robot = RobotContext(
        mass=25.0,
        max_velocity=1.5,
        max_acceleration=0.8,
        wheel_radius=0.12,
        battery_level=0.9
    )
    
    print("✅ System initialized successfully!")
    print("Ready for integration with TRADYN navigation!")
