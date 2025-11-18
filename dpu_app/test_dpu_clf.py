"""
Test script for the refactored dpu_clf.py with Pacman environment.

Tests the main functions to ensure they work correctly with the Pacman gym environment.
"""

import sys
import os

# Change working directory to multiagent so layout loading works
original_dir = os.getcwd()
multiagent_dir = os.path.join(os.path.dirname(__file__), 'multiagent')
os.chdir(multiagent_dir)
sys.path.insert(0, multiagent_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import numpy as np
from pacman_gym_env import PacmanGymEnv
sys.path.insert(0, original_dir)
from dpu_clf import (
    load_agent,
    capture_agent_path,
    evaluate_agent,
    will_it_stuck,
    is_illegal_move,
    actions_cells_locations,
    GridAdapter
)

def test_grid_adapter():
    """Test the GridAdapter class with Pacman environment."""
    print("\n" + "="*60)
    print("TEST 1: GridAdapter")
    print("="*60)
    
    env = PacmanGymEnv(layout_name='smallClassic', num_ghosts=2, render_mode=None)
    env.reset()
    
    adapter = GridAdapter(env)
    print(f"✓ GridAdapter created")
    print(f"  Width: {adapter.width}")
    print(f"  Height: {adapter.height}")
    print(f"  Type: {adapter._type}")
    
    # Test grid access
    cell = adapter.get(1, 1)
    print(f"  Cell at (1,1): {cell}")
    
    wall_cell = adapter.get(0, 0)
    print(f"  Cell at (0,0): {wall_cell}")
    
    return True


def test_load_agent():
    """Test loading a Pacman PPO agent."""
    print("\n" + "="*60)
    print("TEST 2: Load Agent")
    print("="*60)
    
    env = PacmanGymEnv(layout_name='smallClassic', num_ghosts=2, render_mode=None)
    
    # Try multiple model paths (we're in multiagent dir now)
    model_paths = [
        'test_model/best_model.zip',
        'test_model/ppo_pacman_final.zip',
        'test_model/checkpoints/ppo_pacman_100000_steps.zip',
    ]
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"✗ Model not found: {model_path}")
            continue
        
        print(f"Loading model from: {model_path}")
        try:
            agent = load_agent(env, model_path)
            print(f"✓ Agent loaded successfully")
            print(f"  Type: {type(agent)}")
            return agent, env
        except Exception as e:
            print(f"✗ Failed to load agent: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("✗ No valid model could be loaded!")
    return False


def test_is_illegal_move():
    """Test illegal move detection."""
    print("\n" + "="*60)
    print("TEST 3: Illegal Move Detection")
    print("="*60)
    
    # Test cases
    test_cases = [
        (0, (5, 5), (5, 4), False, "North movement (legal)"),
        (1, (5, 5), (5, 6), False, "South movement (legal)"),
        (2, (5, 5), (6, 5), False, "East movement (legal)"),
        (3, (5, 5), (4, 5), False, "West movement (legal)"),
        (4, (5, 5), (5, 5), False, "Stop action (legal)"),
        (0, (5, 5), (5, 5), True, "North blocked by wall (illegal)"),
        (2, (5, 5), (5, 5), True, "East blocked by wall (illegal)"),
    ]
    
    all_passed = True
    for action, last_pos, current_pos, expected, description in test_cases:
        result = is_illegal_move(action, last_pos, current_pos)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} {description}: {result}")
    
    return all_passed


def test_capture_agent_path(agent, env):
    """Test capturing agent's path in the environment."""
    print("\n" + "="*60)
    print("TEST 4: Capture Agent Path")
    print("="*60)
    
    if agent is None or env is None:
        print("✗ Skipping test (agent or env not available)")
        return False
    
    try:
        # Create a copy of the environment
        env_copy = copy.deepcopy(env)
        env_copy.reset()
        
        print("Running agent and capturing path...")
        move_sequence, illegal_moves, total_reward, legal_actions = capture_agent_path(env_copy, agent)
        
        print(f"✓ Captured agent path successfully")
        print(f"  Total moves: {len(move_sequence)}")
        print(f"  Illegal moves: {illegal_moves}")
        print(f"  Total reward: {total_reward}")
        print(f"  Legal actions count: {len(legal_actions)}")
        
        if len(move_sequence) > 0:
            print(f"  First 5 moves: {move_sequence[:5]}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to capture agent path: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_actions_cells_locations():
    """Test converting move sequence to cell locations."""
    print("\n" + "="*60)
    print("TEST 5: Actions to Cell Locations")
    print("="*60)
    
    # Create a test move sequence
    move_sequence = [
        ("right", "east"),
        ("right", "east"),
        ("down", "south"),
        ("down", "south"),
        ("left", "west"),
    ]
    
    try:
        locations = actions_cells_locations(move_sequence)
        print(f"✓ Converted move sequence to cell locations")
        print(f"  Input moves: {len(move_sequence)}")
        print(f"  Output locations: {len(locations)}")
        print(f"  Path: {locations}")
        
        # Verify path makes sense
        if len(locations) == len(move_sequence) + 1:  # +1 for starting position
            print(f"  ✓ Correct number of locations")
            return True
        else:
            print(f"  ✗ Expected {len(move_sequence) + 1} locations, got {len(locations)}")
            return False
    except Exception as e:
        print(f"✗ Failed to convert actions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_will_it_stuck(agent, env):
    """Test checking if agent gets stuck."""
    print("\n" + "="*60)
    print("TEST 6: Will It Stuck")
    print("="*60)
    
    if agent is None or env is None:
        print("✗ Skipping test (agent or env not available)")
        return False
    
    try:
        # Create a fresh environment
        test_env = PacmanGymEnv(layout_name='smallClassic', num_ghosts=2, max_steps=50)
        test_env.reset()
        
        print("Checking if agent gets stuck (max_steps=50)...")
        stuck = will_it_stuck(agent, test_env)
        
        print(f"✓ Test completed")
        print(f"  Agent gets stuck: {stuck}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to test stuck detection: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluate_agent(agent, env):
    """Test agent evaluation over multiple episodes."""
    print("\n" + "="*60)
    print("TEST 7: Evaluate Agent")
    print("="*60)
    
    if agent is None or env is None:
        print("✗ Skipping test (agent or env not available)")
        return False
    
    try:
        # Create a fresh environment for evaluation
        eval_env = PacmanGymEnv(layout_name='smallClassic', num_ghosts=2, render_mode=None)
        
        print("Evaluating agent over 5 episodes...")
        avg_reward, avg_illegal, avg_moves, max_steps_count = evaluate_agent(
            eval_env, agent, num_episodes=5
        )
        
        print(f"✓ Evaluation completed")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average illegal moves: {avg_illegal}")
        print(f"  Average moves per episode: {avg_moves}")
        print(f"  Episodes reaching max steps: {max_steps_count}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to evaluate agent: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING REFACTORED DPU_CLF WITH PACMAN")
    print("="*60)
    
    results = {}
    
    # Test 1: GridAdapter
    results['GridAdapter'] = test_grid_adapter()
    
    # Test 2: Load Agent
    agent, env = test_load_agent()
    results['Load Agent'] = (agent is not None)
    
    # Test 3: Illegal Move Detection
    results['Illegal Move Detection'] = test_is_illegal_move()
    
    # Test 4: Capture Agent Path
    results['Capture Agent Path'] = test_capture_agent_path(agent, env)
    
    # Test 5: Actions to Cell Locations
    results['Actions to Cells'] = test_actions_cells_locations()
    
    # Test 6: Will It Stuck
    results['Will It Stuck'] = test_will_it_stuck(agent, env)
    
    # Test 7: Evaluate Agent
    results['Evaluate Agent'] = test_evaluate_agent(agent, env)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The refactored code works with Pacman!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
