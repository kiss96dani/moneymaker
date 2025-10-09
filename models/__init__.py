"""
Models package for ML-based predictions.

Provides ModelWrapper for inference and ModelTrainer for training.
"""
from .model_wrapper import ModelWrapper
from .trainer import ModelTrainer

__all__ = ['ModelWrapper', 'ModelTrainer']
