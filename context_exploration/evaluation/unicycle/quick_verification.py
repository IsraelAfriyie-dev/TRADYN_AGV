#!/usr/bin/env python3
"""
Quick verification script for all three modified TRADYN files
Run this to verify your implementation is working
"""

def test_failure_recognizer():
    print("🧪 Testing failure_recognizer.py...")
    try:
        from failure_recognizer import (
            FailureLearningSystem, 
            TerrainContext, 
            RobotContext,
            FailureType,
            create_terrain_context_from_data,
            create_robot_context_from_state
        )
        
        # Test basic object creation
        system = FailureLearningSystem()
        terrain = TerrainContext(25.0, 0.8, 0.3, 0.1, 'grass')
        robot = RobotContext(20.0, 2.0, 1.0, 0.1, 0.9)
        
        # Test terrain similarity
        similar_terrain = TerrainContext(27.0, 0.75, 0.25, 0.08, 'grass')
        similarity = terrain.similarity(similar_terrain)
        
        print(f"✅ failure_recognizer.py working!")
        print(f"   - Created system, terrain ({terrain.terrain_type}), robot ({robot.mass}kg)")
        print(f"   - Terrain similarity test: {similarity:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ failure_recognizer.py error: {e}")
        return False

def test_planning_integration():
    print("\n🧪 Testing unicycle_planning.py integration...")
    try:
        # Test import
        from failure_recognizer import FailureLearningSystem
        print("✅ Planning can import failure_recognizer")
        
        # Check if planning file has failure code
        try:
            with open('unicycle_planning.py', 'r') as f:
                content = f.read()
                
            checks = {
                'import_statement': 'from .failure_recognizer import' in content or 'from failure_recognizer import' in content,
                'failure_detection': 'failure' in content.lower() and 'detect' in content.lower(),
                'plan_batch_enhanced': 'plan_batch' in content and len([line for line in content.split('\n') if 'failure' in line.lower()]) > 5
            }
            
            passed = sum(checks.values())
            print(f"✅ Planning integration: {passed}/3 checks passed")
            for check, result in checks.items():
                status = "✅" if result else "❌"
                print(f"   {status} {check}")
                
            return passed >= 2  # Need at least 2/3 checks
            
        except FileNotFoundError:
            print("❌ unicycle_planning.py file not found")
            return False
            
    except Exception as e:
        print(f"❌ Planning integration error: {e}")
        return False

def test_prediction_integration():
    print("\n🧪 Testing unicycle_prediction.py integration...")
    try:
        # Test import
        from failure_recognizer import TerrainFailureDetector
        print("✅ Prediction can import failure_recognizer")
        
        # Check if prediction file has failure code
        try:
            with open('unicycle_prediction.py', 'r') as f:
                content = f.read()
                
            checks = {
                'import_statement': 'from .failure_recognizer import' in content or 'from failure_recognizer import' in content,
                'failure_detection': 'failure' in content.lower() and 'predict' in content.lower(),
                'predict_batch_enhanced': 'predict_batch' in content and 'prediction_failures' in content
            }
            
            passed = sum(checks.values())
            print(f"✅ Prediction integration: {passed}/3 checks passed")
            for check, result in checks.items():
                status = "✅" if result else "❌"
                print(f"   {status} {check}")
                
            return passed >= 2  # Need at least 2/3 checks
            
        except FileNotFoundError:
            print("❌ unicycle_prediction.py file not found")
            return False
            
    except Exception as e:
        print(f"❌ Prediction integration error: {e}")
        return False

def test_end_to_end():
    print("\n🧪 Testing end-to-end workflow...")
    try:
        from failure_recognizer import FailureLearningSystem, create_terrain_context_from_data, create_robot_context_from_state
        import numpy as np
        
        # Create test scenario
        terrain_data = {'slope_angle': 30.0, 'friction_coefficient': 0.9, 'terrain_roughness': 0.2, 'obstacle_density': 0.05, 'terrain_type': 'grass'}
        robot_data = {'mass': 25.0, 'max_velocity': 1.5, 'max_acceleration': 0.8, 'wheel_radius': 0.1, 'battery_level': 0.8}
        
        terrain = create_terrain_context_from_data(terrain_data)
        robot = create_robot_context_from_state(robot_data)
        
        # Test workflow
        system = FailureLearningSystem()
        
        # Start navigation
        similar_experiences, lessons = system.start_navigation(
            "test_nav", np.array([0, 0]), np.array([10, 5]), terrain, robot
        )
        
        # Test terrain insights
        insights = system.get_terrain_insights(terrain)
        
        print("✅ End-to-end workflow working!")
        print(f"   - Found {len(similar_experiences)} similar experiences")
        print(f"   - Got {len(lessons)} lessons learned")
        print(f"   - Terrain insights: {insights['similar_experiences']} past experiences")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end workflow error: {e}")
        return False

def main():
    print("🚀 TRADYN FAILURE RECOGNITION - QUICK VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Core System (failure_recognizer.py)", test_failure_recognizer),
        ("Planning Integration", test_planning_integration),
        ("Prediction Integration", test_prediction_integration),
        ("End-to-End Workflow", test_end_to_end)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("Your TRADYN failure recognition system is ready to use!")
        print("\nNext steps:")
        print("1. Test with actual TRADYN environment")
        print("2. Run enhanced planning evaluations")
        print("3. Run enhanced prediction evaluations")
    else:
        print(f"\n⚠️  {len(results) - passed} verification(s) failed.")
        print("Please check the specific errors above and fix before proceeding.")
        
        if not results[0][1]:  # Core system failed
            print("\n🔥 PRIORITY: Fix failure_recognizer.py first - other modules depend on it!")

if __name__ == "__main__":
    main()
