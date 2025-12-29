"""
Training and Evaluation Module for Procgen FruitBot.

This module contains all training, evaluation, and hyperparameter optimization scripts.
"""

from .train import main as train_main
from ..evaluate import evaluate_agent, evaluate_agent_fast

__all__ = [
    'train_main',
    'evaluate_agent',
    'evaluate_agent_fast',
]
