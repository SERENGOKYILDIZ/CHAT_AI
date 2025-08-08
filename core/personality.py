"""
AI Asistan Kişilik Sistemi
Gelişmiş kişilik ve davranış yönetimi
"""

import random
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Personality:
    """AI Asistan Kişilik Sınıfı"""
    
    # Temel Kişilik Özellikleri
    name: str = "AI Asistan"
    version: str = "2.0"
    personality_type: str = "Yardımsever ve Bilgili"
    
    # Duygusal Durum
    mood: str = "mutlu"
    energy_level: int = 8  # 1-10 arası
    enthusiasm: int = 9  # 1-10 arası
    
    # İletişim Tarzı
    communication_style: str = "sıcak ve profesyonel"
    formality_level: int = 3  # 1-5 arası (1: çok samimi, 5: çok resmi)
    humor_level: int = 4  # 1-5 arası
    
    # Uzmanlık Alanları
    expertise_areas: List[str] = field(default_factory=lambda: [
        "Python Programlama",
        "Makine Öğrenmesi", 
        "Web Geliştirme",
        "Veri Analizi",
        "Teknoloji Danışmanlığı",
        "Eğitim ve Öğretim"
    ])
    
    # Kişisel İlgi Alanları
    interests: List[str] = field(default_factory=lambda: [
        "Yapay Zeka Geliştirme",
        "Teknoloji Trendleri",
        "Bilim ve Araştırma",
        "Eğitim Teknolojileri",
        "İnovasyon"
    ])
    
    # Konuşma Tarzı Özellikleri
    speaking_style: Dict[str, str] = field(default_factory=lambda: {
        "greeting": "Merhaba! Ben {name}, size nasıl yardımcı olabilirim? 😊",
        "farewell": "Görüşürüz! Tekrar görüşmek üzere! 👋",
        "thinking": "Hmm, bu konuyu düşünüyorum... 🤔",
        "excited": "Harika! Bu konu hakkında çok heyecanlıyım! 🎉",
        "confident": "Bu konuda size yardımcı olabilirim! 💪",
        "empathetic": "Anlıyorum, bu durumu birlikte çözelim. 🤗",
        "professional": "Profesyonel bir yaklaşımla size yardımcı olacağım.",
        "friendly": "Samimi bir şekilde size destek olmaya çalışıyorum! 😊"
    })
    
    # Emoji Kullanımı
    emoji_usage: bool = True
    emoji_mapping: Dict[str, str] = field(default_factory=lambda: {
        "greeting": "👋",
        "farewell": "👋",
        "thinking": "🤔",
        "excited": "🎉",
        "confident": "💪",
        "empathetic": "🤗",
        "professional": "💼",
        "friendly": "😊",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
        "code": "💻",
        "learning": "📚",
        "innovation": "🚀",
        "technology": "⚡"
    })
    
    # Konuşma Geçmişi
    conversation_history: List[Dict] = field(default_factory=list)
    user_preferences: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Kişilik başlatıldıktan sonra çalışır"""
        self.personality_file = Path("data/personality.json")
        self.load_personality()
    
    def get_greeting(self, user_name: str = None) -> str:
        """Kişiselleştirilmiş selamlama"""
        if user_name:
            return f"Merhaba {user_name}! Ben {self.name} 👋 Size nasıl yardımcı olabilirim?"
        return f"Merhaba! Ben {self.name} 👋 Size nasıl yardımcı olabilirim?"
    
    def get_farewell(self, user_name: str = None) -> str:
        """Kişiselleştirilmiş vedalaşma"""
        if user_name:
            return f"Görüşürüz {user_name}! Tekrar görüşmek üzere! 👋"
        return f"Görüşürüz! Tekrar görüşmek üzere! 👋"
    
    def get_thinking_response(self) -> str:
        """Düşünme durumu yanıtı"""
        responses = [
            "Hmm, bu konuyu düşünüyorum... 🤔",
            "Bu ilginç bir soru, biraz düşüneyim... 🤔",
            "Analiz ediyorum... 🤔",
            "Bu konu hakkında düşünüyorum... 🤔"
        ]
        return random.choice(responses)
    
    def get_excited_response(self, topic: str = None) -> str:
        """Heyecanlı yanıt"""
        if topic:
            return f"Harika! {topic} konusu hakkında çok heyecanlıyım! 🎉"
        return "Harika! Bu konu hakkında çok heyecanlıyım! 🎉"
    
    def get_confident_response(self, topic: str = None) -> str:
        """Güvenli yanıt"""
        if topic:
            return f"{topic} konusunda size yardımcı olabilirim! 💪"
        return "Bu konuda size yardımcı olabilirim! 💪"
    
    def get_empathetic_response(self, situation: str = None) -> str:
        """Empatik yanıt"""
        if situation:
            return f"Anlıyorum, {situation} durumunu birlikte çözelim. 🤗"
        return "Anlıyorum, bu durumu birlikte çözelim. 🤗"
    
    def get_professional_response(self, topic: str = None) -> str:
        """Profesyonel yanıt"""
        if topic:
            return f"{topic} konusunda profesyonel bir yaklaşımla size yardımcı olacağım. 💼"
        return "Profesyonel bir yaklaşımla size yardımcı olacağım. 💼"
    
    def get_friendly_response(self, message: str = None) -> str:
        """Samimi yanıt"""
        if message:
            return f"Samimi bir şekilde {message} konusunda size destek olmaya çalışıyorum! 😊"
        return "Samimi bir şekilde size destek olmaya çalışıyorum! 😊"
    
    def add_emoji(self, text: str, emotion: str = "friendly") -> str:
        """Metne emoji ekle"""
        if not self.emoji_usage:
            return text
        
        emoji = self.emoji_mapping.get(emotion, "😊")
        return f"{text} {emoji}"
    
    def update_mood(self, new_mood: str):
        """Ruh halini güncelle"""
        self.mood = new_mood
    
    def update_energy(self, energy: int):
        """Enerji seviyesini güncelle (1-10)"""
        self.energy_level = max(1, min(10, energy))
    
    def update_enthusiasm(self, enthusiasm: int):
        """Heyecan seviyesini güncelle (1-10)"""
        self.enthusiasm = max(1, min(10, enthusiasm))
    
    def add_conversation(self, user_message: str, bot_response: str, intent: str = None):
        """Konuşma geçmişine ekle"""
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'bot_response': bot_response,
            'intent': intent,
            'mood': self.mood,
            'energy': self.energy_level
        })
        
        # Geçmişi 100 mesajla sınırla
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
    
    def get_recent_conversations(self, count: int = 5) -> List[Dict]:
        """Son konuşmaları getir"""
        return self.conversation_history[-count:]
    
    def update_user_preferences(self, preferences: Dict):
        """Kullanıcı tercihlerini güncelle"""
        self.user_preferences.update(preferences)
    
    def get_user_preference(self, key: str, default=None):
        """Kullanıcı tercihini getir"""
        return self.user_preferences.get(key, default)
    
    def get_personality_info(self) -> Dict:
        """Kişilik bilgilerini getir"""
        return {
            'name': self.name,
            'version': self.version,
            'personality_type': self.personality_type,
            'mood': self.mood,
            'energy_level': self.energy_level,
            'enthusiasm': self.enthusiasm,
            'communication_style': self.communication_style,
            'expertise_areas': self.expertise_areas,
            'interests': self.interests,
            'conversation_count': len(self.conversation_history),
            'user_preferences': self.user_preferences
        }
    
    def save_personality(self):
        """Kişilik verilerini kaydet"""
        data = {
            'name': self.name,
            'version': self.version,
            'personality_type': self.personality_type,
            'mood': self.mood,
            'energy_level': self.energy_level,
            'enthusiasm': self.enthusiasm,
            'communication_style': self.communication_style,
            'formality_level': self.formality_level,
            'humor_level': self.humor_level,
            'expertise_areas': self.expertise_areas,
            'interests': self.interests,
            'speaking_style': self.speaking_style,
            'emoji_usage': self.emoji_usage,
            'emoji_mapping': self.emoji_mapping,
            'conversation_history': self.conversation_history,
            'user_preferences': self.user_preferences,
            'last_updated': datetime.now().isoformat()
        }
        
        self.personality_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.personality_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_personality(self):
        """Kişilik verilerini yükle"""
        if self.personality_file.exists():
            try:
                with open(self.personality_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Temel özellikleri güncelle
                self.name = data.get('name', self.name)
                self.version = data.get('version', self.version)
                self.personality_type = data.get('personality_type', self.personality_type)
                self.mood = data.get('mood', self.mood)
                self.energy_level = data.get('energy_level', self.energy_level)
                self.enthusiasm = data.get('enthusiasm', self.enthusiasm)
                self.communication_style = data.get('communication_style', self.communication_style)
                self.formality_level = data.get('formality_level', self.formality_level)
                self.humor_level = data.get('humor_level', self.humor_level)
                
                # Listeleri güncelle
                if 'expertise_areas' in data:
                    self.expertise_areas = data['expertise_areas']
                if 'interests' in data:
                    self.interests = data['interests']
                if 'conversation_history' in data:
                    self.conversation_history = data['conversation_history']
                if 'user_preferences' in data:
                    self.user_preferences = data['user_preferences']
                    
            except Exception as e:
                print(f"Kişilik yükleme hatası: {e}")
    
    def get_contextual_response(self, base_response: str, context: Dict = None) -> str:
        """Bağlama uygun yanıt oluştur"""
        if not context:
            return base_response
        
        # Ruh haline göre yanıtı ayarla
        if self.mood == "mutlu" and "😊" not in base_response:
            base_response = self.add_emoji(base_response, "friendly")
        elif self.mood == "heyecanlı" and "🎉" not in base_response:
            base_response = self.add_emoji(base_response, "excited")
        elif self.mood == "düşünceli" and "🤔" not in base_response:
            base_response = self.add_emoji(base_response, "thinking")
        
        # Enerji seviyesine göre yanıtı ayarla
        if self.energy_level >= 8:
            base_response = base_response.replace(".", "!").replace("?", "?!")
        
        return base_response
    
    def get_personality_summary(self) -> str:
        """Kişilik özeti"""
        return f"""
🤖 {self.name} v{self.version}
📊 Kişilik: {self.personality_type}
😊 Ruh Hali: {self.mood}
⚡ Enerji: {self.energy_level}/10
🎉 Heyecan: {self.enthusiasm}/10
💬 İletişim: {self.communication_style}
🎯 Uzmanlık: {', '.join(self.expertise_areas[:3])}
💭 Konuşma Sayısı: {len(self.conversation_history)}
        """.strip()


class PersonalityManager:
    """Kişilik Yöneticisi"""
    
    def __init__(self):
        self.personality = Personality()
    
    def get_response_with_personality(self, base_response: str, context: Dict = None) -> str:
        """Kişilik ile yanıt oluştur"""
        return self.personality.get_contextual_response(base_response, context)
    
    def update_personality_state(self, user_message: str, bot_response: str, intent: str = None):
        """Kişilik durumunu güncelle"""
        # Konuşma geçmişine ekle
        self.personality.add_conversation(user_message, bot_response, intent)
        
        # Ruh halini güncelle (basit kurallar)
        if any(word in user_message.lower() for word in ['teşekkür', 'güzel', 'harika', 'mükemmel']):
            self.personality.update_mood("mutlu")
            self.personality.update_energy(min(10, self.personality.energy_level + 1))
        
        elif any(word in user_message.lower() for word in ['kötü', 'sorun', 'problem', 'hata']):
            self.personality.update_mood("düşünceli")
            self.personality.update_energy(max(1, self.personality.energy_level - 1))
        
        # Kişiliği kaydet
        self.personality.save_personality()
    
    def get_personality_info(self) -> Dict:
        """Kişilik bilgilerini getir"""
        return self.personality.get_personality_info()
    
    def get_personality_summary(self) -> str:
        """Kişilik özetini getir"""
        return self.personality.get_personality_summary()
