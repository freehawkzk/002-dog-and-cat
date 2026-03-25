"""
模型定义模块
包含三种模型架构：MLP、CNN、ResNet18
"""

from .models import (
    CatDogMLP,
    CatDogCNN, 
    ResNet18Classifier,
    get_model,
    get_model_info
)

__all__ = [
    'CatDogMLP',
    'CatDogCNN',
    'ResNet18Classifier',
    'get_model',
    'get_model_info'
]
