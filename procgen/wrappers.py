"""
Gym wrappers for Procgen environments
"""
import numpy as np

# Try to import both gym and gymnasium
try:
    import gym
    from gym.core import ActionWrapper as GymActionWrapper
except ImportError:
    gym = None
    GymActionWrapper = None

try:
    import gymnasium as gymn
    from gymnasium.core import ActionWrapper as GymnActionWrapper
except ImportError:
    gymn = None
    GymnActionWrapper = None


# Use gymnasium wrapper if available (for SB3 2.x compatibility), otherwise gym
if GymnActionWrapper is not None:
    BaseActionWrapper = GymnActionWrapper
elif GymActionWrapper is not None:
    BaseActionWrapper = GymActionWrapper
else:
    raise ImportError("Neither gymnasium nor gym is available")


class DiscreteActionWrapper(BaseActionWrapper):
    """
    Wrapper to limit the action space to a subset of actions.
    Also ensures Gymnasium API compatibility for stable-baselines3 2.x.
    
    This is useful for simplifying the action space by removing unused or redundant actions.
    
    Example for FruitBot (original has 15 actions):
        # Use only: left, right, stay, throw
        env = DiscreteActionWrapper(env, [1, 7, 4, 9])
        # Now env.action_space is Discrete(4) instead of Discrete(15)
        # action 0 -> original action 1 (left)
        # action 1 -> original action 7 (right)
        # action 2 -> original action 4 (stay)
        # action 3 -> original action 9 (throw)
    """
    
    def __init__(self, env, action_map):
        """
        Args:
            env: The environment to wrap
            action_map: List of original action indices to keep.
                       New action space will be Discrete(len(action_map))
        """
        super().__init__(env)
        self.action_map = np.array(action_map, dtype=np.int32)
        
        # Always use gymnasium.spaces for SB3 2.x compatibility
        if gymn is not None:
            self.action_space = gymn.spaces.Discrete(len(action_map))
            
            # Also convert observation space to gymnasium if needed
            if hasattr(env.observation_space, '__module__') and 'gym.' in str(env.observation_space.__module__):
                # Convert gym.spaces.Box to gymnasium.spaces.Box
                obs_space = env.observation_space
                if hasattr(obs_space, 'low') and hasattr(obs_space, 'high'):
                    self.observation_space = gymn.spaces.Box(
                        low=obs_space.low,
                        high=obs_space.high,
                        shape=obs_space.shape,
                        dtype=obs_space.dtype
                    )
        else:
            raise ImportError("gymnasium is required but not available")
    
    def reset(self, **kwargs):
        """Reset environment with Gymnasium API (returns obs, info)"""
        # Extract seed if provided (Gymnasium style)
        seed = kwargs.pop('seed', None)
        
        # Handle seeding for old Gym environments
        if seed is not None and hasattr(self.env, 'seed') and not hasattr(self.env, 'unwrapped'):
            try:
                self.env.seed(seed)
            except TypeError:
                # If seed() doesn't accept arguments, try without
                pass
        
        # Handle old gym API (returns just obs) vs new gymnasium API (returns obs, info)
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
            # Old gym doesn't accept seed in reset
            result = self.env.reset()
            
        if isinstance(result, tuple):
            # Already returns (obs, info) - gymnasium style
            return result
        else:
            # Old gym style - just obs
            return result, {}
    
    def seed(self, seed=None):
        """Handle seed() calls for compatibility with old Gym API"""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return None
    
    def step(self, action):
        """Step with mapped action, ensuring Gymnasium API (returns obs, reward, terminated, truncated, info)"""
        # Map the action to original action space
        original_action = self.action_map[action]
        
        # Call the base environment
        result = self.env.step(original_action)
        
        # Handle old gym API (4 values) vs new gymnasium API (5 values)
        if len(result) == 4:
            # Old gym: (obs, reward, done, info)
            obs, reward, done, info = result
            # Convert to gymnasium: (obs, reward, terminated, truncated, info)
            return obs, reward, done, False, info
        else:
            # Already gymnasium format: (obs, reward, terminated, truncated, info)
            return result
        
    def action(self, act):
        """Map the new action to the original action space"""
        return self.action_map[act]
    
    def reverse_action(self, act):
        """Map original action back to new action space (for compatibility)"""
        # Find where act appears in action_map
        matches = np.where(self.action_map == act)[0]
        if len(matches) > 0:
            return matches[0]
        # If action not in map, return first action as default
        return 0


# Predefined action mappings for common use cases
FRUITBOT_BASIC_ACTIONS = [1, 7, 4, 9]  # left, right, stay, throw (4 core actions)
FRUITBOT_FULL_DIRECTIONS = [4, 1, 7, 5, 3, 9]  # stay, left, right, up, down, throw
FRUITBOT_MINIMAL_ACTIONS = [1, 7, 9]  # left, right, throw (3 actions)
CARDINAL_DIRECTIONS = [4, 1, 7, 5, 3]  # stay, left, right, up, down (no action buttons)
MOVEMENT_ONLY = [1, 7, 5, 3]  # left, right, up, down (no stay, no actions)


def make_fruitbot_basic(env):
    """Wrap FruitBot env with basic 4-action space: left, right, stay, throw"""
    return DiscreteActionWrapper(env, FRUITBOT_BASIC_ACTIONS)


def make_fruitbot_minimal(env):
    """Wrap FruitBot env with minimal 3-action space: left, right, throw"""
    return DiscreteActionWrapper(env, FRUITBOT_MINIMAL_ACTIONS)
