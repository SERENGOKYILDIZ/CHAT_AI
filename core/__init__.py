#!/usr/bin/env python3
"""
Core Package - Advanced AI Chatbot
Bu paket chatbot'un temel bileşenlerini içerir
"""

from .web_design import WebDesignManager
from .smart_chatbot import SmartChatbot
from .personality import PersonalityManager
from .context_manager import ContextManager
from .logger import ChatbotLogger, logger, log_info, log_error, log_performance, log_user_action, log_api_call, log_chat_interaction
from .multi_chat_system import MultiChatManager, create_session, add_message, get_session_data, update_title, delete_session, get_all_sessions, get_session_stats

__version__ = "3.0.0"
__author__ = "Advanced AI Chatbot Team"

__all__ = [
    'WebDesignManager',
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
    'log_chat_interaction',
    'MultiChatManager',
    'create_session',
    'add_message',
    'get_session_data',
    'update_title',
    'delete_session',
    'get_all_sessions',
    'get_session_stats'
]