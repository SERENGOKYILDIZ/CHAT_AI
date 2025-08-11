#!/usr/bin/env python3
"""
Core Package - Advanced AI Chatbot
Bu paket chatbot'un temel bileşenlerini içerir
"""

from .web_design import WebDesignManager
from .data_collector import DataCollector
from .model_trainer import ModelTrainer
from .smart_chatbot import SmartChatbot
from .personality import PersonalityManager
from .context_manager import ContextManager
from .logger import ChatbotLogger, logger, log_info, log_error, log_performance, log_user_action, log_api_call, log_chat_interaction

__version__ = "3.0.0"
__author__ = "Advanced AI Chatbot Team"

__all__ = [
    'WebDesignManager',
    'DataCollector',
    'ModelTrainer',
    'SmartChatbot',
    'PersonalityManager',
    'ContextManager',
    'ChatbotLogger',
    'logger',
    'log_info',
    'log_error',
    'log_performance',
    'log_user_action',
    'log_api_call',
    'log_chat_interaction'
]