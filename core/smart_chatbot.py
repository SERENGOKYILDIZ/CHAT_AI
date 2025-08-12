#!/usr/bin/env python3
"""
Smart Chatbot - Gelişmiş AI Chatbot Engine
Kişilik, ML model ve bağlam yönetimini entegre eder
"""

from .personality import PersonalityManager
from .context_manager import ContextManager
from .ai_engine import AIEngine
from .logger import log_info, log_error, log_performance, log_user_action
import random
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

class SmartChatbot:
    def __init__(self, model_size: str = "medium"):
        """SmartChatbot başlat"""
        self.model_size = model_size
        self.start_time = time.time()
        
        # Core Managers
        self.personality_manager = PersonalityManager()
        self.context_manager = ContextManager()
        
        # New AI Engine
        self.ai_engine = AIEngine(model_size)
        
        # ML Model (Legacy support - removed)
        # self.classifier = None
        # self.vectorizer = None
        
        # Responses
        self.responses = {}
        
        # Stats
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0,
            'average_response_time': 0.0,
            'ai_engine_requests': 0,
            'start_time': time.time()
        }
        
        # Conversation history
        self.conversation_history = []
        
        # Load components
        self._load_model()
        self._load_responses()
        
        log_info(f"🤖 SmartChatbot başlatıldı", model_size=model_size)
    
    def _load_model(self):
        """ML modelini yükle (Legacy support)"""
        try:
            model_path = Path(f"data/models/chatbot_model_{self.model_size}.pkl")
            vectorizer_path = Path(f"data/models/vectorizer_{self.model_size}.pkl")
            
            if model_path.exists() and vectorizer_path.exists():
                # Legacy model loading removed - using AI Engine only
                log_info(f"⚠️ Legacy ML model dosyaları bulunamadı, AI Engine kullanılıyor", 
                        model_path=str(model_path), vectorizer_path=str(vectorizer_path))
            else:
                log_info(f"⚠️ Legacy ML model dosyaları bulunamadı, AI Engine kullanılıyor", 
                        model_path=str(model_path), vectorizer_path=str(vectorizer_path))
                
        except Exception as e:
            log_error(e, context={"operation": "legacy_model_loading", "model_size": self.model_size})
            log_info("⚠️ Legacy model yüklenemedi, AI Engine kullanılıyor")
    
    def _load_responses(self):
        """Yanıt verilerini yükle"""
        try:
            responses_file = Path("data/responses.json")
            if responses_file.exists():
                import json
                with open(responses_file, 'r', encoding='utf-8') as f:
                    self.responses = json.load(f)
                log_info("✅ Yanıt verileri yüklendi", response_count=len(self.responses))
            else:
                log_info("⚠️ Yanıt dosyası bulunamadı, varsayılan yanıtlar kullanılıyor")
                self.responses = {
                    "greeting": ["Merhaba!", "Selam!", "Hoş geldiniz!"],
                    "farewell": ["Görüşürüz!", "Hoşça kalın!", "İyi günler!"],
                    "unknown": ["Anlamadım, tekrar eder misiniz?", "Bu konuda bilgim yok."]
                }
        except Exception as e:
            log_error(e, context={"operation": "responses_loading"})
            log_info("⚠️ Yanıt verileri yüklenemedi, varsayılan yanıtlar kullanılıyor")
    
    def predict_intent(self, user_input: str) -> Tuple[str, float]:
        """Kullanıcı mesajının amacını tahmin et"""
        try:
            # Try AI Engine first
            if hasattr(self.ai_engine, 'nlp_processor'):
                intent = self.ai_engine.nlp_processor.extract_intent(user_input)
                # Simple confidence calculation based on pattern matching
                confidence = self._calculate_intent_confidence(user_input, intent)
                return intent, confidence
            
            # Fallback to keyword-based
            return self._keyword_based_intent(user_input), 0.6
            
        except Exception as e:
            log_error(e, context={"operation": "intent_prediction"})
            return "unknown", 0.5
    
    def _calculate_intent_confidence(self, user_input: str, intent: str) -> float:
        """Intent confidence hesapla"""
        try:
            # Pattern-based confidence calculation
            patterns = {
                'question': [r'\?', r'nedir', r'nasıl', r'nerede', r'ne zaman', r'kim'],
                'greeting': [r'merhaba', r'selam', r'hoş geldin', r'günaydın'],
                'farewell': [r'görüşürüz', r'hoşça kal', r'iyi günler', r'iyi akşamlar'],
                'casual': [r'nasılsın', r'naber', r'iyi misin'],
                'formal': [r'lütfen', r'rica ederim', r'teşekkürler'],
                'explanation': [r'açıkla', r'anlat', r'detay', r'bilgi']
            }
            
            if intent in patterns:
                for pattern in patterns[intent]:
                    if re.search(pattern, user_input.lower()):
                        return 0.9  # High confidence for pattern match
                return 0.7  # Medium confidence for intent match
            else:
                return 0.6  # Default confidence
                
        except Exception:
            return 0.5
    
    def _keyword_based_intent(self, user_input: str) -> str:
        """Keyword-based intent detection"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['merhaba', 'selam', 'hoş geldin']):
            return 'greeting'
        elif any(word in user_input_lower for word in ['görüşürüz', 'hoşça kal', 'iyi günler']):
            return 'farewell'
        elif '?' in user_input or any(word in user_input_lower for word in ['nedir', 'nasıl', 'nerede']):
            return 'question'
        elif any(word in user_input_lower for word in ['açıkla', 'anlat', 'detay']):
            return 'explanation'
        else:
            return 'general'
    
    def _analyze_sentiment(self, text: str) -> str:
        """Metin duygu analizi"""
        try:
            positive_words = ['güzel', 'harika', 'mükemmel', 'sevimli', 'hoş', 'iyi']
            negative_words = ['kötü', 'berbat', 'korkunç', 'üzgün', 'kızgın', 'kötü']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return 'positive'
            elif negative_count > positive_count:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _handle_math_calculation(self, user_input: str) -> Tuple[bool, str]:
        """Matematik hesaplaması yap"""
        try:
            # Simple math pattern detection
            math_pattern = r'(\d+[\+\-\*\/]\d+)'
            matches = re.findall(math_pattern, user_input)
            
            if matches:
                for match in matches:
                    try:
                        result = eval(match)
                        return True, f"{match} = {result}"
                    except:
                        continue
            
            # Check for calculation keywords
            calc_keywords = ['hesapla', 'topla', 'çıkar', 'çarp', 'böl']
            if any(keyword in user_input.lower() for keyword in calc_keywords):
                # Extract numbers and operation
                numbers = re.findall(r'\d+', user_input)
                if len(numbers) >= 2:
                    # Simple addition for now
                    result = sum(int(num) for num in numbers)
                    return True, f"Sayıların toplamı: {result}"
            
            return False, ""
            
        except Exception:
            return False, ""
    
    def _generate_smart_response(self, user_input: str, intent: str, confidence: float) -> str:
        """Akıllı yanıt üretimi"""
        try:
            # Check for math calculation
            is_math, math_result = self._handle_math_calculation(user_input)
            if is_math:
                return math_result
            
            # Intent-based response generation
            if intent == 'greeting':
                responses = self.responses.get('greeting', ['Merhaba!', 'Selam!'])
                return random.choice(responses)
            
            elif intent == 'farewell':
                responses = self.responses.get('farewell', ['Görüşürüz!', 'Hoşça kalın!'])
                return random.choice(responses)
            
            elif intent == 'question':
                # Try to answer based on knowledge
                return "Bu konuda size yardımcı olmaya çalışayım. Daha spesifik bir soru sorabilir misiniz?"
            
            elif intent == 'explanation':
                return "Bu konuyu detaylı olarak açıklayayım. Hangi yönü hakkında daha fazla bilgi istiyorsunuz?"
            
            else:
                # Default response
                responses = self.responses.get('unknown', ['Anlamadım, tekrar eder misiniz?'])
                return random.choice(responses)
                
        except Exception as e:
            log_error(e, context={"operation": "smart_response_generation"})
            return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."
    
    def get_response(self, user_input: str, session_id: str = "default") -> str:
        """Ana yanıt üretimi"""
        start_time = time.time()
        
        try:
            # Update stats
            self.stats['total_requests'] += 1
            
            # Get context
            context = self.context_manager.get_context(session_id)
            
            # Try AI Engine first (primary method)
            try:
                ai_response = self.ai_engine.generate_response(user_input, context)
                if ai_response and not ai_response.get('error'):
                    self.stats['ai_engine_requests'] += 1
                    self.stats['successful_requests'] += 1
                    
                    # Update context
                    self.context_manager.add_message(session_id, 'user', user_input)
                    self.context_manager.add_message(session_id, 'bot', ai_response['response'])
                    
                    # Log performance
                    response_time = time.time() - start_time
                    self._update_response_time_stats(response_time)
                    
                    log_performance("ai_engine_response", response_time, {
                        "session_id": session_id,
                        "intent": ai_response.get('intent', 'unknown'),
                        "confidence": ai_response.get('confidence', 0.0)
                    })
                    
                    return ai_response['response']
                    
            except Exception as e:
                log_error(e, context={"operation": "ai_engine_response", "session_id": session_id})
                # Continue to fallback methods
            
            # Fallback to legacy methods
            try:
                # Intent prediction
                intent, confidence = self.predict_intent(user_input)
                
                # Generate response
                response = self._generate_smart_response(user_input, intent, confidence)
                
                # Update context
                self.context_manager.add_message(session_id, 'user', user_input)
                self.context_manager.add_message(session_id, 'bot', response)
                
                # Update stats
                self.stats['legacy_model_requests'] += 1
                self.stats['successful_requests'] += 1
                
                # Log performance
                response_time = time.time() - start_time
                self._update_response_time_stats(response_time)
                
                log_performance("legacy_response", response_time, {
                    "session_id": session_id,
                    "intent": intent,
                    "confidence": confidence
                })
                
                return response
                
            except Exception as e:
                log_error(e, context={"operation": "legacy_response_generation", "session_id": session_id})
                self.stats['failed_requests'] += 1
                
                # Emergency fallback
                return "Üzgünüm, şu anda teknik bir sorun yaşıyorum. Lütfen daha sonra tekrar deneyin."
            
        except Exception as e:
            log_error(e, context={"operation": "get_response", "session_id": session_id})
            self.stats['failed_requests'] += 1
            return "Bir hata oluştu. Lütfen tekrar deneyin."
    
    def _update_response_time_stats(self, response_time: float):
        """Yanıt süresi istatistiklerini güncelle"""
        self.stats['total_response_time'] += response_time
        total_requests = self.stats['successful_requests']
        if total_requests > 0:
            self.stats['average_response_time'] = self.stats['total_response_time'] / total_requests
    
    def get_personality_info(self) -> Dict[str, Any]:
        """Kişilik bilgilerini getir"""
        try:
            base_personality = self.personality_manager.get_personality()
            
            # Add AI Engine capabilities
            ai_capabilities = self.ai_engine.get_ai_capabilities()
            
            return {
                **base_personality,
                'ai_capabilities': ai_capabilities,
                'model_size': self.model_size,
                'engine_type': 'AI Engine + Legacy ML'
            }
            
        except Exception as e:
            log_error(e, context={"operation": "personality_info_retrieval"})
            return {'error': 'Kişilik bilgileri alınamadı'}
    
    def get_context_info(self, session_id: str) -> Dict[str, Any]:
        """Bağlam bilgilerini getir"""
        try:
            context = self.context_manager.get_context(session_id)
            
            # Add AI Engine memory info
            memory_info = self.ai_engine.memory_system.get_memory_usage()
            
            return {
                **context,
                'ai_memory': memory_info,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(e, context={"operation": "context_info_retrieval", "session_id": session_id})
            return {'error': 'Bağlam bilgileri alınamadı'}
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri getir"""
        try:
            # Combine legacy and AI Engine stats
            ai_stats = self.ai_engine.get_performance_stats()
            
            return {
                **self.stats,
                'ai_engine_stats': ai_stats,
                'uptime': time.time() - self.start_time,
                'success_rate': (self.stats['successful_requests'] / max(self.stats['total_requests'], 1)) * 100,
                'ai_engine_usage_rate': (self.stats['ai_engine_requests'] / max(self.stats['total_requests'], 1)) * 100
            }
            
        except Exception as e:
            log_error(e, context={"operation": "stats_retrieval"})
            return {'error': 'İstatistikler alınamadı'}
    
    def get_timestamp(self) -> str:
        """Mevcut timestamp'i getir"""
        return datetime.now().isoformat()
    
    def retrain_model(self, model_size: str = None):
        """Model'i yeniden eğit"""
        try:
            if model_size:
                self.model_size = model_size
            
            log_info(f"🔄 Model yeniden eğitimi başlatılıyor", model_size=self.model_size)
            
            # Retrain legacy model
            if hasattr(self.model_trainer, 'train_model'):
                training_result = self.model_trainer.train_model(self.model_size)
                log_info("✅ Legacy model yeniden eğitildi", result=training_result)
            
            # Reload AI Engine
            self.ai_engine = AIEngine(self.model_size)
            log_info("✅ AI Engine yeniden yüklendi", model_size=self.model_size)
            
        except Exception as e:
            log_error(e, context={"operation": "model_retraining", "model_size": model_size})
    
    def get_model_info(self) -> Dict[str, Any]:
        """Model bilgilerini getir"""
        try:
            return {
                'model_size': self.model_size,
                'legacy_model_loaded': self.classifier is not None,
                'ai_engine_loaded': hasattr(self, 'ai_engine'),
                'ai_capabilities': self.ai_engine.get_ai_capabilities() if hasattr(self, 'ai_engine') else {},
                'personality_loaded': hasattr(self, 'personality_manager'),
                'context_loaded': hasattr(self, 'context_manager'),
                'timestamp': self.get_timestamp()
            }
            
        except Exception as e:
            log_error(e, context={"operation": "model_info_retrieval"})
            return {'error': 'Model bilgileri alınamadı'}
