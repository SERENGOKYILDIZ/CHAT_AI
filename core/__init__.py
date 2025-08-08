#!/usr/bin/env python3
"""
Core Package - Advanced AI Chatbot
Bu paket chatbot'un temel bileşenlerini içerir
"""

from .advanced_chatbot import AdvancedChatbot
from .web_design import WebDesignManager
from .data_collector import DataCollector
from .model_trainer import ModelTrainer

__version__ = "3.0.0"
__author__ = "Advanced AI Chatbot Team"

__all__ = [
    'AdvancedChatbot',
    'WebDesignManager',
    'DataCollector',
    'ModelTrainer'
]