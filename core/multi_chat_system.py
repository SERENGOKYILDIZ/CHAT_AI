#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi Chat System
Çoklu sohbet sistemi - Her sohbet bağımsız ve kalıcı.
Sohbetler hiçbir zaman kaybolmaz.
"""

import json
import time
import uuid
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict

@dataclass
class ChatMessage:
    """Tek bir chat mesajını temsil eder"""
    id: str
    type: str  # "user" veya "bot"
    content: str
    timestamp: str
    session_id: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        return cls(
            id=data['id'],
            type=data['type'],
            content=data['content'],
            timestamp=data['timestamp'],
            session_id=data['session_id']
        )

@dataclass
class ChatSession:
    """Çoklu chat session yönetimi"""
    id: str
    title: str
    created: str
    last_modified: str
    message_count: int = 0
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatSession':
        return cls(
            id=data['id'],
            title=data['title'],
            created=data['created'],
            last_modified=data['last_modified'],
            message_count=data.get('message_count', 0),
            is_active=data.get('is_active', True)
        )

class MultiChatManager:
    """Çoklu chat yönetim sistemi"""
    
    def __init__(self, storage_file: str = "data/multi_chat_sessions.json"):
        self.storage_file = storage_file
        self.sessions: Dict[str, ChatSession] = {}
        self.messages: Dict[str, List[ChatMessage]] = {}
        self.lock = threading.RLock()
        self.load_data()
    
    def create_session(self, title: str = "Yeni Sohbet") -> str:
        """Yeni session oluşturur"""
        with self.lock:
            session_id = f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            now = datetime.now().isoformat()
            
            session = ChatSession(
                id=session_id,
                title=title,
                created=now,
                last_modified=now
            )
            
            self.sessions[session_id] = session
            self.messages[session_id] = []
            
            self.save_data()
            return session_id
    
    def add_message(self, session_id: str, message_type: str, content: str) -> str:
        """Session'a mesaj ekler"""
        with self.lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            # Create message
            message_id = f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            now = datetime.now().isoformat()
            
            message = ChatMessage(
                id=message_id,
                type=message_type,
                content=content,
                timestamp=now,
                session_id=session_id
            )
            
            # Add to messages
            if session_id not in self.messages:
                self.messages[session_id] = []
            self.messages[session_id].append(message)
            
            # Update session
            session = self.sessions[session_id]
            session.message_count = len(self.messages[session_id])
            session.last_modified = now
            
            # Auto-update title for first user message
            if (message_type == "user" and 
                session.message_count == 1 and 
                session.title == "Yeni Sohbet"):
                auto_title = content[:30] + ('...' if len(content) > 30 else '')
                self.update_session_title(session_id, auto_title)
            
            self.save_data()
            return message_id
    
    def get_session_messages(self, session_id: str) -> List[ChatMessage]:
        """Session'daki mesajları döner"""
        return self.messages.get(session_id, []).copy()
    
    def get_session_data(self, session_id: str) -> Optional[Dict]:
        """Session ve mesajlarını birlikte döner"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        messages = self.messages.get(session_id, [])
        
        return {
            'session': session.to_dict(),
            'messages': [msg.to_dict() for msg in messages]
        }
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Session başlığını günceller"""
        with self.lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            session.title = title
            session.last_modified = datetime.now().isoformat()
            
            self.save_data()
            return True
    
    def delete_session(self, session_id: str) -> bool:
        """Session'ı siler"""
        with self.lock:
            if session_id not in self.sessions:
                return False
            
            # Remove session and messages
            del self.sessions[session_id]
            if session_id in self.messages:
                del self.messages[session_id]
            
            self.save_data()
            return True
    
    def get_all_sessions(self) -> List[Dict]:
        """Tüm session'ları döner"""
        return [session.to_dict() for session in self.sessions.values()]
    
    def get_active_sessions(self) -> List[Dict]:
        """Aktif session'ları döner"""
        active = []
        for session in self.sessions.values():
            if session.is_active:
                active.append(session.to_dict())
        return active
    
    def get_session_stats(self) -> Dict:
        """Session istatistiklerini döner"""
        total_sessions = len(self.sessions)
        active_sessions = len(self.get_active_sessions())
        total_messages = sum(len(msgs) for msgs in self.messages.values())
        
        return {
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'total_messages': total_messages,
            'average_messages_per_session': total_messages / total_sessions if total_sessions > 0 else 0
        }
    
    def save_data(self) -> None:
        """Verileri dosyaya kaydeder"""
        try:
            data = {
                'sessions': [session.to_dict() for session in self.sessions.values()],
                'messages': {
                    session_id: [msg.to_dict() for msg in msgs]
                    for session_id, msgs in self.messages.items()
                }
            }
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.error(f"Data save error: {e}")
    
    def load_data(self) -> None:
        """Verileri dosyadan yükler"""
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load sessions
            self.sessions = {}
            for session_data in data.get('sessions', []):
                session = ChatSession.from_dict(session_data)
                self.sessions[session.id] = session
            
            # Load messages
            self.messages = {}
            for session_id, messages_data in data.get('messages', {}).items():
                self.messages[session_id] = [
                    ChatMessage.from_dict(msg_data) for msg_data in messages_data
                ]
                
        except FileNotFoundError:
            # Create new file
            self.save_data()
        except Exception as e:
            logging.error(f"Data load error: {e}")

# Global instance
multi_chat_manager = MultiChatManager()

# Convenience functions
def create_session(title: str = "Yeni Sohbet") -> str:
    return multi_chat_manager.create_session(title)

def add_message(session_id: str, message_type: str, content: str) -> str:
    return multi_chat_manager.add_message(session_id, message_type, content)

def get_session_data(session_id: str) -> Optional[Dict]:
    return multi_chat_manager.get_session_data(session_id)

def update_title(session_id: str, title: str) -> bool:
    return multi_chat_manager.update_session_title(session_id, title)

def delete_session(session_id: str) -> bool:
    return multi_chat_manager.delete_session(session_id)

def get_all_sessions() -> List[Dict]:
    return multi_chat_manager.get_all_sessions()

def get_session_stats() -> Dict:
    return multi_chat_manager.get_session_stats()

if __name__ == "__main__":
    # Test the multi chat system
    print("🚀 Multi Chat System Test")
    
    # Create session
    session_id = create_session("Test Sohbeti")
    print(f"✅ Session created: {session_id}")
    
    # Add messages
    add_message(session_id, "user", "Merhaba! Nasılsın?")
    add_message(session_id, "bot", "Merhaba! Ben iyiyim, teşekkür ederim.")
    
    # Get session data
    session_data = get_session_data(session_id)
    print(f"📝 Session has {len(session_data['messages'])} messages")
    print(f"📋 Title: {session_data['session']['title']}")
    
    # Stats
    stats = get_session_stats()
    print(f"📊 Stats: {stats}")
    
    print("✅ Test completed!")
