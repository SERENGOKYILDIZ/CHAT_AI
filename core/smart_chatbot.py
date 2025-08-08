"""
Gelişmiş AI Asistan - Kişilik Sistemi ile Entegre
Akıllı yanıt sistemi ve kişilik yönetimi
"""

import re
import json
import random
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .personality import PersonalityManager
from .model_trainer import ModelTrainer


class SmartChatbot:
    """Gelişmiş AI Asistan - Kişilik Sistemi ile Entegre"""
    
    def __init__(self, model_size: str = "medium"):
        # Logging
        self.logger = self._setup_logging()
        
        # Kişilik sistemi
        self.personality_manager = PersonalityManager()
        
        # Model sistemi
        self.model_trainer = ModelTrainer()
        self.model_size = model_size
        self.classifier = None
        self.vectorizer = None
        self.responses = {}
        
        # Konuşma geçmişi
        self.conversation_history = []
        
        # İstatistikler
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0
        }
        
        # Model yükle
        self._load_model()
        self._load_responses()
        
        self.logger.info(f"🤖 SmartChatbot başlatıldı - Model: {model_size}")
    
    def _setup_logging(self) -> logging.Logger:
        """Logging ayarları"""
        logger = logging.getLogger('SmartChatbot')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_model(self):
        """Model yükle"""
        try:
            model_path = Path(f"models/{self.model_size}_model.pkl")
            vectorizer_path = Path(f"models/{self.model_size}_vectorizer.pkl")
            
            if model_path.exists() and vectorizer_path.exists():
                self.classifier = self.model_trainer.load_model(str(model_path))
                self.vectorizer = self.model_trainer.load_model(str(vectorizer_path))
                self.logger.info(f"✅ Model yüklendi: {self.model_size}")
            else:
                self.logger.warning(f"⚠️ Model bulunamadı: {self.model_size}")
                
        except Exception as e:
            self.logger.error(f"❌ Model yükleme hatası: {e}")
    
    def _load_responses(self):
        """Yanıt verilerini yükle"""
        try:
            with open("data/training_datasets.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for intent, content in data.items():
                if "responses" in content:
                    self.responses[intent] = content["responses"]
            
            self.logger.info(f"✅ {len(self.responses)} intent yüklendi")
            
        except Exception as e:
            self.logger.error(f"❌ Yanıt yükleme hatası: {e}")
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Intent tahmin et"""
        try:
            if not self.classifier or not self.vectorizer:
                return "unknown", 0.0
            
            # Text'i vektörize et
            X = self.vectorizer.transform([text])
            
            # Tahmin yap
            intent = self.classifier.predict(X)[0]
            confidence = max(self.classifier.predict_proba(X)[0])
            
            return intent, confidence
            
        except Exception as e:
            self.logger.error(f"❌ Intent tahmin hatası: {e}")
            return "unknown", 0.0
    
    def _analyze_sentiment(self, text: str) -> str:
        """Basit duygu analizi"""
        positive_words = ['güzel', 'harika', 'mükemmel', 'teşekkür', 'sevgi', 'mutlu', 'iyi']
        negative_words = ['kötü', 'berbat', 'korkunç', 'sorun', 'problem', 'hata', 'üzgün']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _handle_math_calculation(self, match) -> str:
        """Matematik işlemi yap"""
        try:
            num1 = int(match.group(1))
            operator = match.group(2)
            num2 = int(match.group(3))
            
            if operator == '+':
                result = num1 + num2
            elif operator == '-':
                result = num1 - num2
            elif operator == '*':
                result = num1 * num2
            elif operator == '/':
                if num2 == 0:
                    return "Sıfıra bölme hatası! ❌"
                result = num1 / num2
            else:
                return "Desteklenmeyen işlem! ❌"
            
            return f"{num1} {operator} {num2} = {result} ✅"
            
        except Exception as e:
            return f"Hesaplama hatası: {e} ❌"
    
    def _generate_smart_response(self, user_input: str, intent: str, confidence: float) -> str:
        """Akıllı yanıt oluştur"""
        # Matematik işlemi kontrolü
        math_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', user_input)
        if math_match:
            return self._handle_math_calculation(math_match)
        
        # Intent tabanlı yanıt
        if confidence > 0.6 and intent in self.responses:
            base_response = random.choice(self.responses[intent])
        else:
            # Fallback yanıtlar
            fallback_responses = [
                "Bu konu hakkında size yardımcı olmaya çalışıyorum! 🤔",
                "İlginç bir soru, biraz daha detay verebilir misiniz? 🤔",
                "Bu konuda size nasıl yardımcı olabilirim? 🤔",
                "Anlıyorum, bu konu hakkında düşünüyorum... 🤔",
                "Size en iyi şekilde yardımcı olmaya çalışıyorum! 😊"
            ]
            base_response = random.choice(fallback_responses)
        
        return base_response
    
    def get_response(self, user_input: str, session_id: str = "default") -> str:
        """Ana yanıt fonksiyonu"""
        start_time = datetime.now()
        
        try:
            self.stats['total_requests'] += 1
            
            # Input temizleme
            user_input = user_input.strip()
            if not user_input:
                return "Lütfen bir mesaj yazın! 😊"
            
            # Intent tahmin et
            intent, confidence = self.predict_intent(user_input)
            
            # Duygu analizi
            sentiment = self._analyze_sentiment(user_input)
            
            # Akıllı yanıt oluştur
            base_response = self._generate_smart_response(user_input, intent, confidence)
            
            # Kişilik ile yanıtı geliştir
            context = {
                'intent': intent,
                'confidence': confidence,
                'sentiment': sentiment,
                'session_id': session_id
            }
            
            final_response = self.personality_manager.get_response_with_personality(
                base_response, context
            )
            
            # Kişilik durumunu güncelle
            self.personality_manager.update_personality_state(
                user_input, final_response, intent
            )
            
            # Konuşma geçmişine ekle
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'user': user_input,
                'bot': final_response,
                'intent': intent,
                'confidence': confidence,
                'sentiment': sentiment
            })
            
            # İstatistikleri güncelle
            response_time = (datetime.now() - start_time).total_seconds()
            self.stats['successful_requests'] += 1
            self.stats['average_response_time'] = (
                (self.stats['average_response_time'] * (self.stats['successful_requests'] - 1) + response_time) 
                / self.stats['successful_requests']
            )
            
            self.logger.info(f"✅ Yanıt oluşturuldu - Intent: {intent}, Confidence: {confidence:.3f}")
            
            return final_response
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            self.logger.error(f"❌ Yanıt oluşturma hatası: {e}")
            return "Üzgünüm, bir hata oluştu. Tekrar deneyebilir misiniz? 😔"
    
    def get_personality_info(self) -> Dict:
        """Kişilik bilgilerini getir"""
        return self.personality_manager.get_personality_info()
    
    def get_personality_summary(self) -> str:
        """Kişilik özetini getir"""
        return self.personality_manager.get_personality_summary()
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Konuşma geçmişini getir"""
        return self.conversation_history[-limit:]
    
    def get_stats(self) -> Dict:
        """İstatistikleri getir"""
        return {
            **self.stats,
            'personality_info': self.get_personality_info(),
            'model_size': self.model_size,
            'loaded_intents': len(self.responses)
        }
    
    def get_timestamp(self) -> str:
        """Şu anki zaman damgası"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def train_model(self, model_size: str = None):
        """Model eğit"""
        if model_size:
            self.model_size = model_size
        
        try:
            self.logger.info(f"🚀 Model eğitimi başlatılıyor: {self.model_size}")
            self.model_trainer.train_model(self.model_size)
            self._load_model()
            self.logger.info("✅ Model eğitimi tamamlandı")
            
        except Exception as e:
            self.logger.error(f"❌ Model eğitimi hatası: {e}")
    
    def save_conversation(self):
        """Konuşma geçmişini kaydet"""
        try:
            conversation_file = Path("data/conversation_history.json")
            conversation_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            
            self.logger.info("✅ Konuşma geçmişi kaydedildi")
            
        except Exception as e:
            self.logger.error(f"❌ Konuşma kaydetme hatası: {e}")
    
    def load_conversation(self):
        """Konuşma geçmişini yükle"""
        try:
            conversation_file = Path("data/conversation_history.json")
            
            if conversation_file.exists():
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                
                self.logger.info(f"✅ {len(self.conversation_history)} konuşma yüklendi")
            
        except Exception as e:
            self.logger.error(f"❌ Konuşma yükleme hatası: {e}")
    
    def clear_conversation(self):
        """Konuşma geçmişini temizle"""
        self.conversation_history = []
        self.logger.info("✅ Konuşma geçmişi temizlendi")
    
    def get_help_info(self) -> str:
        """Yardım bilgilerini getir"""
        return f"""
🤖 {self.personality_manager.personality.name} v{self.personality_manager.personality.version}

💬 Size şu konularda yardımcı olabilirim:
• Günlük sohbet ve selamlaşma
• Teknik sorunlar ve programlama
• Öğrenme ve eğitim konuları
• Matematik hesaplamaları
• Kişilik ve ruh hali yönetimi

🎯 Uzmanlık Alanlarım:
{', '.join(self.personality_manager.personality.expertise_areas)}

😊 Ruh Halim: {self.personality_manager.personality.mood}
⚡ Enerji Seviyem: {self.personality_manager.personality.energy_level}/10

💡 Nasıl kullanabilirsiniz:
• Doğal dil ile konuşun
• Sorularınızı sorun
• Matematik işlemleri yapın (örn: 5 + 3)
• Kişilik hakkında sorular sorun

Her zaman size yardımcı olmaya hazırım! 😊
        """.strip()
