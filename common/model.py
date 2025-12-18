"""
IMPALA-CNN architecture for Procgen environments.

Standard configuration optimized for Procgen:
- depths=[16, 32, 32]
- embedding_size=256
"""

import torch
import torch.nn as nn
try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class ImpalaCNN(BaseFeaturesExtractor):
    """
    IMPALA-CNN feature extractor for visual RL.
    
    Uses Conv -> MaxPool -> ResBlock pattern, which is much more powerful
    than the default NatureCNN for Procgen-style environments.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # Standard IMPALA depths for Procgen
        depths = [16, 32, 32]
        
        layers = []
        for depth in depths:
            layers.extend([
                nn.Conv2d(n_input_channels, depth, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                # Residual-like layers
                nn.Conv2d(depth, depth, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(depth, depth, kernel_size=3, padding=1),
                nn.ReLU(),
            ])
            n_input_channels = depth
        
        self.cnn = nn.Sequential(*layers)
        
        # Calculate flattened size
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *observation_space.shape)).flatten(1).shape[1]
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class ImpalaModel(ActorCriticPolicy):
    """
    PPO policy with IMPALA-CNN feature extractor.
    
    Drop-in replacement for "CnnPolicy" with better architecture for Procgen.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['features_extractor_class'] = ImpalaCNN
        kwargs['features_extractor_kwargs'] = {'features_dim': 256}
        super().__init__(*args, **kwargs)
