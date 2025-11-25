"""
Custom Gym adapter for Procgen with proper render support
"""
import numpy as np
from gym3 import ToGymEnv

class RenderableToGymEnv(ToGymEnv):
    """
    Adapts a gym3 environment to the gym interface with proper render support
    """
    def __init__(self, env, render_mode=None):
        super().__init__(env)
        self.render_mode = render_mode
        self._last_info = None

    def reset(self):
        obs = super().reset()
        self._last_info = self.env.get_info()[0] if self.env.get_info() else {}
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self._last_info = info
        return obs, reward, done, info

    def render(self, mode=None):
        mode = self.render_mode
        info = self._last_info or (self.env.get_info()[0] if self.env.get_info() else {})
        if mode == "rgb_array":
            if 'rgb' in info:
                return info['rgb']
            else:
                return self.observation
        elif mode == "human":
            return None
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def seed(self, seed=None):
        """
        Propagate the seed to the underlying environment.
        If no seed is provided, generate a random seed.
        """
        
        return self.env.seed()

    def close(self):
        super().close()
