"""
Custom wrappers for Procgen environments
"""
from typing import Any, Tuple
import numpy as np
from gym3.wrapper import Wrapper
from gym import spaces  # Import spaces from gym
import gym


class ReducedActionWrapper(gym.ActionWrapper):
    """
    Gym wrapper that:
    - exposes a reduced Discrete(k) action space to the agent
    - maps reduced actions into the original procgen actions.
    """
    def __init__(self, env, valid_actions=None):
        super().__init__(env)
        if valid_actions is None:
            valid_actions = [1, 4, 7, 9]

        self.valid_actions = np.array(valid_actions, dtype=np.int64)

        # What the agent sees:
        self.action_space = spaces.Discrete(len(self.valid_actions))

    def action(self, act):
        """
        Map Discrete(len(valid_actions)) -> original action in Discrete(15).
        SB3 usually sends a scalar or a 0-dim array.
        """
        if isinstance(act, np.ndarray):
            # VecEnv / SB3 sometimes give numpy scalars/arrays
            act = int(act)
        return int(self.valid_actions[act])
    
    # def seed(self, seed=None):
    #     """
    #     Propagate the seed to the underlying environment.
    #     If no seed is provided, generate a random seed.
    #     """
    #     if seed is None:
    #         seed = np.random.randint(0, 2**31 - 1)
    #     if hasattr(self.env, "seed"):
    #         return self.env.seed(seed)
    #     return None


class InfoRgbRenderWrapper(Wrapper):
    """
    Wrapper that caches the high-resolution RGB image from info['rgb']
    for use with render() calls in rgb_array mode.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._cached_rgb = None
    
    def observe(self) -> Tuple[Any, Any, Any]:
        reward, obs, first = self.env.observe()
        # Cache the high-res RGB from info if available
        info = self.env.get_info()
        if info and len(info) > 0 and 'rgb' in info[0]:
            self._cached_rgb = info[0]['rgb'].copy()
        return reward, obs, first
    
    def get_cached_rgb(self) -> np.ndarray:
        """Return the cached high-resolution RGB image"""
        return self._cached_rgb

    # def seed(self, seed=None):
    #     """
    #     Propagate the seed to the underlying environment.
    #     If no seed is provided, generate a random seed.
    #     """
    #     if seed is None:
    #         seed = np.random.randint(0, 2**31 - 1)
    #     if hasattr(self.env, "seed"):
    #         return self.env.seed(seed)
    #     return None


class StayBonusWrapper(Wrapper):
    """
    Wrapper that gives bonus reward when action == STAY (mapped action 2 in reduced space)
    """
    
    def __init__(self, env, stay_bonus=0.1):
        super().__init__(env)
        self.stay_bonus = stay_bonus
        self._last_action = None
    
    def act(self, ac: Any) -> None:
        # Store the action to check later (this is in the REDUCED action space)
        self._last_action = ac
        return self.env.act(ac)
    
    def observe(self) -> Tuple[Any, Any, Any]:
        reward, obs, first = self.env.observe()
        
        # Add bonus reward if action was 2 (STAY in reduced action space)
        # STAY is index 2 in [LEFT=0, RIGHT=1, STAY=2, THROW=3]
        if self._last_action is not None:
            # Handle both single action and batch of actions
            if np.any(self._last_action == 2):
                reward = reward + self.stay_bonus
        
        return reward, obs, first

    # def seed(self, seed):
    #     """Propagate the seed to the underlying environment."""
    #     if hasattr(self.env, "seed"):
    #         return self.env.seed(seed)
    #     return None


# FruitBot specific actions
FRUITBOT_BASIC_ACTIONS = [1, 7, 4, 9]  # LEFT, RIGHT, STAY, THROW


def make_fruitbot_basic(env):
    """Apply FruitBot wrappers with basic 4-action space"""
    return StayBonusWrapper(ReducedActionWrapper(env, valid_actions=FRUITBOT_BASIC_ACTIONS))