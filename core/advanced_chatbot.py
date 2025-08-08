#!/usr/bin/env python3
"""
Advanced Scalable Chatbot - Enterprise Ready
Büyük ölçekli modeller için tasarlanmış gelişmiş chatbot sistemi
"""

import re
import json
import datetime
import numpy as np
import pickle
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import random
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib

# ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

@dataclass
class ConversationContext:
    """Konuşma bağlamı sınıfı"""
    session_id: str
    user_id: str = "default"
    current_topic: str = ""
    conversation_flow: List[Dict] = field(default_factory=list)
    user_preferences: Dict = field(default_factory=dict)
    conversation_summary: str = ""
    context_keywords: Set[str] = field(default_factory=set)
    emotion_history: List[str] = field(default_factory=list)
    intent_history: List[str] = field(default_factory=list)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_updated: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def update_context(self, user_input: str, bot_response: str, intent: str = "", emotion: str = "neutral"):
        """Bağlamı güncelle"""
        self.conversation_flow.append({
            'user_input': user_input,
            'bot_response': bot_response,
            'intent': intent,
            'emotion': emotion,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        self.intent_history.append(intent)
        self.emotion_history.append(emotion)
        self.last_updated = datetime.datetime.now()
        
        # Bağlam anahtar kelimelerini güncelle
        self._extract_context_keywords(user_input)
        
        # Konuşma özetini güncelle
        self._update_conversation_summary()
    
    def _extract_context_keywords(self, text: str):
        """Metinden bağlam anahtar kelimelerini çıkar"""
        # Basit anahtar kelime çıkarma
        words = re.findall(r'\b\w+\b', text.lower())
        # Stop words'leri filtrele
        stop_words = {'ve', 'veya', 'ile', 'için', 'bu', 'şu', 'o', 'bir', 'da', 'de'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        self.context_keywords.update(keywords[:5])  # En fazla 5 anahtar kelime
    
    def _update_conversation_summary(self):
        """Konuşma özetini güncelle"""
        if len(self.conversation_flow) > 0:
            recent_messages = self.conversation_flow[-3:]  # Son 3 mesaj
            summary_parts = []
            
            for msg in recent_messages:
                if msg['intent']:
                    summary_parts.append(f"Kullanıcı {msg['intent']} konusunda konuştu")
            
            if summary_parts:
                self.conversation_summary = ". ".join(summary_parts) + "."
    
    def get_context_for_response(self, max_messages: int = 3) -> Dict:
        """Yanıt için bağlam bilgilerini döndür"""
        recent_messages = self.conversation_flow[-max_messages:] if self.conversation_flow else []
        
        return {
            'current_topic': self.current_topic,
            'recent_messages': recent_messages,
            'user_preferences': self.user_preferences,
            'context_keywords': list(self.context_keywords),
            'conversation_summary': self.conversation_summary,
            'emotion_trend': self._get_emotion_trend(),
            'intent_trend': self._get_intent_trend()
        }
    
    def _get_emotion_trend(self) -> str:
        """Duygu trendini hesapla"""
        if len(self.emotion_history) < 2:
            return "neutral"
        
        recent_emotions = self.emotion_history[-3:]
        positive_count = sum(1 for e in recent_emotions if e == 'positive')
        negative_count = sum(1 for e in recent_emotions if e == 'negative')
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _get_intent_trend(self) -> str:
        """Intent trendini hesapla"""
        if len(self.intent_history) < 2:
            return ""
        
        return self.intent_history[-1] if self.intent_history else ""

@dataclass
class ModelConfig:
    """Model konfigürasyon sınıfı"""
    name: str
    max_features: int
    ngram_range: Tuple[int, int]
    min_df: int
    max_df: float
    alpha: float
    ensemble_models: int
    confidence_threshold: float
    description: str

class DataManager:
    """Veri yönetimi sınıfı"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def load_training_data(self, config_path: str = "config/model_config.json") -> Dict:
        """Konfigürasyona göre eğitim verilerini yükle"""
        all_data = {"patterns": [], "labels": [], "responses": {}}
        
        # Konfigürasyonu yükle
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {"data_sources": {"primary": {"path": "data/training_datasets.json", "enabled": True}}}
        
        # Veri kaynaklarını yükle
        for source_name, source_config in config.get("data_sources", {}).items():
            if not source_config.get("enabled", False):
                continue
                
            source_path = source_config.get("path")
            source_type = source_config.get("type", "json")
            weight = source_config.get("weight", 1.0)
            
            if source_type == "json" and os.path.exists(source_path):
                data = self._load_json_data(source_path, weight)
                self._merge_data(all_data, data)
            elif source_type == "pickle" and os.path.exists(source_path):
                data = self._load_pickle_data(source_path, weight)
                self._merge_data(all_data, data)
        
        return all_data
    
    def _load_json_data(self, file_path: str, weight: float) -> Dict:
        """JSON veri dosyasını yükle"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = {"patterns": [], "labels": [], "responses": {}}
        
        # Tüm intent kategorilerini işle
        for category_name, category_data in data.items():
            if category_name == "metadata":
                continue
                
            if isinstance(category_data, dict):
                for intent, intent_data in category_data.items():
                    if isinstance(intent_data, dict):
                        patterns = intent_data.get("patterns", [])
                        responses = intent_data.get("responses", [])
                        
                        # Ağırlığa göre örnekleri çoğalt
                        repeat_count = max(1, int(weight))
                        for _ in range(repeat_count):
                            result["patterns"].extend(patterns)
                            result["labels"].extend([intent] * len(patterns))
                        
                        result["responses"][intent] = responses
        
        return result
    
    def _load_pickle_data(self, file_path: str, weight: float) -> Dict:
        """Pickle veri dosyasını yükle"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        result = {"patterns": [], "labels": [], "responses": {}}
        
        if "training_data" in data:
            for intent, patterns in data["training_data"].items():
                repeat_count = max(1, int(weight))
                for _ in range(repeat_count):
                    result["patterns"].extend(patterns)
                    result["labels"].extend([intent] * len(patterns))
                
                # Basit yanıtlar oluştur
                result["responses"][intent] = [
                    f"{intent.title()} konusunda size yardımcı olabilirim.",
                    f"Bu {intent} hakkında ne bilmek istiyorsunuz?",
                    f"{intent.title()} konusunda size destek olabilirim."
                ]
        
        return result
    
    def _merge_data(self, target: Dict, source: Dict):
        """İki veri setini birleştir"""
        target["patterns"].extend(source["patterns"])
        target["labels"].extend(source["labels"])
        
        for intent, responses in source["responses"].items():
            if intent not in target["responses"]:
                target["responses"][intent] = []
            target["responses"][intent].extend(responses)
    
    def save_model(self, model_data: Dict, model_name: str, version: str = None):
        """Modeli kaydet"""
        if version is None:
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = self.data_dir / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{model_name}_v{version}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # En son modeli işaret et
        latest_path = model_dir / f"{model_name}_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        return str(model_path)
    
    def load_model(self, model_name: str, version: str = "latest") -> Optional[Dict]:
        """Modeli yükle"""
        model_dir = self.data_dir / "models" / model_name
        
        if version == "latest":
            model_path = model_dir / f"{model_name}_latest.pkl"
        else:
            model_path = model_dir / f"{model_name}_v{version}.pkl"
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        
        return None

class ContextManager:
    """Bağlam yöneticisi sınıfı"""
    
    def __init__(self, context_dir: str = "storage/contexts"):
        self.context_dir = Path(context_dir)
        self.context_dir.mkdir(parents=True, exist_ok=True)
        self.active_contexts: Dict[str, ConversationContext] = {}
        self.context_patterns = {
            'greeting': ['merhaba', 'selam', 'günaydın', 'iyi günler'],
            'farewell': ['görüşürüz', 'hoşça kal', 'güle güle', 'bye'],
            'question': ['nasıl', 'neden', 'ne zaman', 'nerede', 'kim', 'hangi'],
            'confirmation': ['evet', 'tamam', 'doğru', 'kesinlikle'],
            'negation': ['hayır', 'yok', 'değil', 'olmaz'],
            'help': ['yardım', 'destek', 'açıkla', 'nasıl yapılır'],
            'thanks': ['teşekkür', 'sağol', 'teşekkürler', 'thanks']
        }
    
    def create_context(self, session_id: str, user_id: str = "default") -> ConversationContext:
        """Yeni bağlam oluştur"""
        context = ConversationContext(session_id=session_id, user_id=user_id)
        self.active_contexts[session_id] = context
        return context
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Bağlamı al"""
        return self.active_contexts.get(session_id)
    
    def update_context(self, session_id: str, user_input: str, bot_response: str, 
                      intent: str = "", emotion: str = "neutral"):
        """Bağlamı güncelle"""
        context = self.get_context(session_id)
        if context:
            context.update_context(user_input, bot_response, intent, emotion)
    
    def analyze_context(self, session_id: str) -> Dict:
        """Bağlam analizi yap"""
        context = self.get_context(session_id)
        if not context:
            return {}
        
        return {
            'session_duration': (datetime.datetime.now() - context.created_at).total_seconds(),
            'message_count': len(context.conversation_flow),
            'unique_intents': len(set(context.intent_history)),
            'emotion_distribution': self._get_emotion_distribution(context.emotion_history),
            'conversation_coherence': self._calculate_coherence(context),
            'user_engagement': self._calculate_engagement(context)
        }
    
    def _get_emotion_distribution(self, emotions: List[str]) -> Dict[str, int]:
        """Duygu dağılımını hesapla"""
        distribution = defaultdict(int)
        for emotion in emotions:
            distribution[emotion] += 1
        return dict(distribution)
    
    def _calculate_coherence(self, context: ConversationContext) -> float:
        """Konuşma tutarlılığını hesapla"""
        if len(context.conversation_flow) < 2:
            return 1.0
        
        # Basit tutarlılık hesaplama
        intent_changes = 0
        for i in range(1, len(context.intent_history)):
            if context.intent_history[i] != context.intent_history[i-1]:
                intent_changes += 1
        
        return 1.0 - (intent_changes / max(len(context.intent_history) - 1, 1))
    
    def _calculate_engagement(self, context: ConversationContext) -> float:
        """Kullanıcı katılımını hesapla"""
        if not context.conversation_flow:
            return 0.0
        
        # Mesaj uzunluğu ve duygu analizi bazında katılım
        avg_message_length = sum(len(msg['user_input']) for msg in context.conversation_flow) / len(context.conversation_flow)
        positive_emotions = sum(1 for emotion in context.emotion_history if emotion == 'positive')
        
        engagement = (avg_message_length / 50.0) * 0.5 + (positive_emotions / max(len(context.emotion_history), 1)) * 0.5
        return min(engagement, 1.0)
    
    def save_context(self, session_id: str):
        """Bağlamı kaydet"""
        context = self.get_context(session_id)
        if context:
            context_file = self.context_dir / f"{session_id}.json"
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'session_id': context.session_id,
                    'user_id': context.user_id,
                    'current_topic': context.current_topic,
                    'conversation_flow': context.conversation_flow,
                    'user_preferences': context.user_preferences,
                    'conversation_summary': context.conversation_summary,
                    'context_keywords': list(context.context_keywords),
                    'emotion_history': context.emotion_history,
                    'intent_history': context.intent_history,
                    'created_at': context.created_at.isoformat(),
                    'last_updated': context.last_updated.isoformat()
                }, f, ensure_ascii=False, indent=2)
    
    def load_context(self, session_id: str) -> Optional[ConversationContext]:
        """Bağlamı yükle"""
        context_file = self.context_dir / f"{session_id}.json"
        if context_file.exists():
            with open(context_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            context = ConversationContext(
                session_id=data['session_id'],
                user_id=data['user_id'],
                current_topic=data.get('current_topic', ''),
                conversation_flow=data.get('conversation_flow', []),
                user_preferences=data.get('user_preferences', {}),
                conversation_summary=data.get('conversation_summary', ''),
                context_keywords=set(data.get('context_keywords', [])),
                emotion_history=data.get('emotion_history', []),
                intent_history=data.get('intent_history', []),
                created_at=datetime.datetime.fromisoformat(data['created_at']),
                last_updated=datetime.datetime.fromisoformat(data['last_updated'])
            )
            
            self.active_contexts[session_id] = context
            return context
        
        return None
    
    def generate_context_aware_response(self, base_response: str, context: ConversationContext) -> str:
        """Bağlam farkında yanıt oluştur"""
        if not context.conversation_flow:
            return base_response
        
        # Bağlam bilgilerini al
        context_info = context.get_context_for_response()
        
        # Konuşma özetini kullan
        if context_info['conversation_summary']:
            summary_enhancement = f" {context_info['conversation_summary']}"
            if len(base_response + summary_enhancement) < 200:  # Mesaj çok uzun olmasın
                base_response += summary_enhancement
        
        # Duygu trendini kullan
        emotion_trend = context_info['emotion_trend']
        if emotion_trend == 'positive' and '😊' not in base_response:
            base_response += " 😊"
        elif emotion_trend == 'negative' and '😔' not in base_response:
            base_response += " 😔"
        
        # Anahtar kelimeleri kullan
        if context_info['context_keywords']:
            relevant_keywords = list(context_info['context_keywords'])[:2]
            if relevant_keywords:
                keyword_enhancement = f" ({', '.join(relevant_keywords)} hakkında konuşuyoruz)"
                if len(base_response + keyword_enhancement) < 250:
                    base_response += keyword_enhancement
        
        return base_response

class AdvancedChatbot:
    """Gelişmiş Ölçeklenebilir Chatbot"""
    
    def __init__(self, model_size: str = "medium", config_path: str = "config/model_config.json"):
        """
        Gelişmiş chatbot başlatıcı
        
        Args:
            model_size: "small", "medium", "large", "enterprise"
            config_path: Konfigürasyon dosyası yolu
        """
        self.name = f"AI Asistan {model_size.title()}"
        self.model_size = model_size
        self.conversation_history = []
        
        # Logging ayarla
        self._setup_logging()
        
        # Veri yöneticisi
        self.data_manager = DataManager()
        
        # Bağlam yöneticisi
        self.context_manager = ContextManager()
        
        # Konfigürasyonu yükle
        self.config = self._load_config(config_path)
        self.model_config = self._get_model_config()
        
        # ML bileşenleri
        self.vectorizer = None
        self.label_encoder = None
        self.classifier = None
        self.responses = {}
        self.model_metadata = {}
        
        # NLTK verilerini indir
        if ML_AVAILABLE:
            self._download_nltk_data()
        
        # Modeli yükle veya oluştur
        self._initialize_model()
    
    def _setup_logging(self):
        """Logging sistemini ayarla"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "chatbot.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"AdvancedChatbot_{self.model_size}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Konfigürasyon dosyasını yükle"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            self.logger.warning(f"Konfigürasyon dosyası bulunamadı: {config_path}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Varsayılan konfigürasyon"""
        return {
            "model_configurations": {
                "medium": {
                    "name": "Medium Model",
                    "max_features": 2000,
                    "ngram_range": [1, 3],
                    "min_df": 1,
                    "max_df": 0.95,
                    "alpha": 0.1,
                    "ensemble_models": 3,
                    "confidence_threshold": 0.2,
                    "description": "Default medium model"
                }
            }
        }
    
    def _get_model_config(self) -> ModelConfig:
        """Model konfigürasyonunu al"""
        config_data = self.config.get("model_configurations", {}).get(self.model_size, {})
        
        if not config_data:
            self.logger.warning(f"Model konfigürasyonu bulunamadı: {self.model_size}")
            config_data = self.config["model_configurations"]["medium"]
        
        return ModelConfig(
            name=config_data.get("name", f"{self.model_size} Model"),
            max_features=config_data.get("max_features", 2000),
            ngram_range=tuple(config_data.get("ngram_range", [1, 3])),
            min_df=config_data.get("min_df", 1),
            max_df=config_data.get("max_df", 0.95),
            alpha=config_data.get("alpha", 0.1),
            ensemble_models=config_data.get("ensemble_models", 3),
            confidence_threshold=config_data.get("confidence_threshold", 0.2),
            description=config_data.get("description", "")
        )
    
    def _download_nltk_data(self):
        """NLTK verilerini indir"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            self.logger.warning(f"NLTK verileri indirilemedi: {e}")
    
    def _initialize_model(self):
        """Modeli başlat"""
        # Önce kaydedilmiş modeli yüklemeyi dene
        saved_model = self.data_manager.load_model(f"advanced_{self.model_size}")
        
        if saved_model and ML_AVAILABLE:
            self._load_saved_model(saved_model)
            self.logger.info(f"Kaydedilmiş model yüklendi: {self.model_size}")
        elif ML_AVAILABLE:
            self._create_new_model()
            self.logger.info(f"Yeni model oluşturuldu: {self.model_size}")
        else:
            self.logger.warning("ML kütüphaneleri bulunamadı. Basit pattern matching kullanılacak.")
    
    def _load_saved_model(self, model_data: Dict):
        """Kaydedilmiş modeli yükle"""
        self.classifier = model_data.get("classifier")
        self.vectorizer = model_data.get("vectorizer")
        self.label_encoder = model_data.get("label_encoder")
        self.responses = model_data.get("responses", {})
        self.model_metadata = model_data.get("metadata", {})
    
    def _create_new_model(self):
        """Yeni model oluştur"""
        self.logger.info("Yeni model eğitiliyor...")
        
        # Eğitim verilerini yükle
        training_data = self.data_manager.load_training_data()
        
        if not training_data["patterns"]:
            self.logger.error("Eğitim verisi bulunamadı!")
            return
        
        self.logger.info(f"Toplam eğitim örneği: {len(training_data['patterns'])}")
        self.logger.info(f"Benzersiz intent sayısı: {len(set(training_data['labels']))}")
        
        # Yanıtları kaydet
        self.responses = training_data["responses"]
        
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.model_config.max_features,
            ngram_range=self.model_config.ngram_range,
            min_df=self.model_config.min_df,
            max_df=self.model_config.max_df,
            stop_words='english'
        )
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        # Verileri hazırla
        X = self.vectorizer.fit_transform([self._preprocess_text(text) for text in training_data["patterns"]])
        y = self.label_encoder.fit_transform(training_data["labels"])
        
        # Ensemble classifier oluştur
        self.classifier = self._create_ensemble_classifier()
        
        # Modeli eğit
        self.classifier.fit(X, y)
        
        # Model metadata
        self.model_metadata = {
            "model_size": self.model_size,
            "total_samples": len(training_data["patterns"]),
            "total_intents": len(set(training_data["labels"])),
            "created_at": datetime.datetime.now().isoformat(),
            "config": self.model_config.__dict__
        }
        
        # Modeli kaydet
        self._save_model()
        
        self.logger.info("Model eğitimi tamamlandı!")
    
    def _create_ensemble_classifier(self):
        """Ensemble classifier oluştur"""
        estimators = []
        
        # Farklı alpha değerleri ile Naive Bayes modelleri
        for i in range(min(3, self.model_config.ensemble_models)):
            alpha = self.model_config.alpha * (10 ** i)
            estimators.append((f'nb_{i}', MultinomialNB(alpha=alpha)))
        
        # Büyük modeller için ek classifierlar
        if self.model_config.ensemble_models > 3:
            estimators.append(('rf', RandomForestClassifier(n_estimators=100, random_state=42)))
        
        if self.model_config.ensemble_models > 4:
            estimators.append(('svm', SVC(probability=True, random_state=42)))
        
        return VotingClassifier(estimators=estimators, voting='soft')
    
    def _preprocess_text(self, text: str) -> str:
        """Metni ön işle"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def _save_model(self):
        """Modeli kaydet"""
        model_data = {
            "classifier": self.classifier,
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder,
            "responses": self.responses,
            "metadata": self.model_metadata
        }
        
        model_path = self.data_manager.save_model(model_data, f"advanced_{self.model_size}")
        self.logger.info(f"Model kaydedildi: {model_path}")
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Intent tahmin et"""
        if not self.classifier or not self.vectorizer:
            return 'default', 0.0
        
        try:
            processed_text = self._preprocess_text(text)
            X = self.vectorizer.transform([processed_text])
            
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)
            confidence = np.max(probabilities)
            
            intent = self.label_encoder.inverse_transform([prediction])[0]
            
            return intent, confidence
            
        except Exception as e:
            self.logger.error(f"Intent tahmin hatası: {e}")
            return 'default', 0.0
    
    def get_response(self, user_input: str, session_id: str = "default") -> str:
        """Kullanıcı girdisine yanıt ver"""
        user_input = user_input.lower().strip()
        
        # Konuşma geçmişine ekle
        self.conversation_history.append({
            'user': user_input,
            'timestamp': self.get_timestamp()
        })
        
        # Bağlam yönetimi
        context = self.context_manager.get_context(session_id)
        if not context:
            context = self.context_manager.create_context(session_id)
        
        # Matematik işlemi kontrolü
        math_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', user_input)
        if math_match:
            response = self._handle_math_calculation(math_match)
            self.context_manager.update_context(session_id, user_input, response, "math_calculation", "neutral")
            return response
        
        # Intent tahmin et
        intent = ""
        confidence = 0.0
        if self.classifier:
            intent, confidence = self.predict_intent(user_input)
            
            self.logger.info(f"Intent: {intent}, Confidence: {confidence:.3f}")
        
        # Duygu analizi
        sentiment = self._analyze_sentiment(user_input)
        
        # Yanıt oluştur
        response = self._generate_response_with_context(user_input, intent, confidence, context)
        
        # Bağlamı güncelle
        self.context_manager.update_context(session_id, user_input, response, intent, sentiment)
        
        return response
    
    def _generate_response_with_context(self, user_input: str, intent: str, confidence: float, context: ConversationContext) -> str:
        """Bağlam farkında yanıt oluştur"""
        base_response = ""
        
        # Intent tabanlı yanıt
        if confidence > self.model_config.confidence_threshold:
            if intent in self.responses and self.responses[intent]:
                base_response = random.choice(self.responses[intent])
            else:
                base_response = self._fallback_response(user_input)
        else:
            base_response = self._fallback_response(user_input)
        
        # Bağlam farkında yanıt oluştur
        context_aware_response = self.context_manager.generate_context_aware_response(base_response, context)
        
        # Duygu analizi ekle
        sentiment = self._analyze_sentiment(user_input)
        if sentiment == 'positive' and '😊' not in context_aware_response:
            context_aware_response += " 😊"
        elif sentiment == 'negative' and '😔' not in context_aware_response:
            context_aware_response += " 😔"
        
        return context_aware_response
    
    def _handle_math_calculation(self, match) -> str:
        """Matematik hesaplaması"""
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
                    return "Sıfıra bölme hatası!"
                result = num1 / num2
            
            return f"Sonuç: {num1} {operator} {num2} = {result} 🧮"
        
        except Exception as e:
            return f"Hesaplama hatası: {str(e)}"
    
    def _analyze_sentiment(self, text: str) -> str:
        """Duygu analizi"""
        positive_words = ['güzel', 'harika', 'mükemmel', 'mutlu', 'iyi', 'süper']
        negative_words = ['kötü', 'berbat', 'üzgün', 'mutsuz', 'kızgın', 'sinirli']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _fallback_response(self, user_input: str) -> str:
        """Fallback yanıt"""
        responses = [
            "İlginç! Biraz daha açıklayabilir misiniz?",
            "Bu konu hakkında daha fazla bilgi verebilir misiniz?",
            "Size bu konuda nasıl yardımcı olabilirim?",
            "Anladım. Size nasıl destek olabilirim?"
        ]
        return random.choice(responses)
    
    def get_timestamp(self) -> str:
        """Zaman damgası"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_stats(self) -> Dict:
        """Chatbot istatistikleri"""
        return {
            'name': self.name,
            'model_size': self.model_size,
            'total_conversations': len(self.conversation_history),
            'ml_available': ML_AVAILABLE,
            'model_active': self.classifier is not None,
            'total_intents': len(self.responses) if self.responses else 0,
            'confidence_threshold': self.model_config.confidence_threshold,
            'model_metadata': self.model_metadata,
            'created_at': self.get_timestamp()
        }
    
    def retrain_model(self):
        """Modeli yeniden eğit"""
        self.logger.info("Model yeniden eğitiliyor...")
        self._create_new_model()
    
    def evaluate_model(self) -> Dict:
        """Model performansını değerlendir"""
        if not self.classifier:
            return {'error': 'Model yüklenmemiş'}
        
        test_cases = [
            ("merhaba", "greeting"),
            ("teşekkürler", "gratitude"),
            ("görüşürüz", "farewell"),
            ("yardım", "help"),
            ("matematik", "math"),
            ("mutlu", "positive"),
            ("üzgün", "negative"),
            ("şaka yap", "joke"),
            ("müzik öner", "music"),
            ("spor hakkında", "sports")
        ]
        
        correct = 0
        total = len(test_cases)
        results = []
        
        for text, expected in test_cases:
            intent, confidence = self.predict_intent(text)
            is_correct = intent == expected
            if is_correct:
                correct += 1
            
            results.append({
                'text': text,
                'predicted': intent,
                'expected': expected,
                'confidence': confidence,
                'correct': is_correct
            })
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'model_size': self.model_size,
            'results': results
        }
    
    # Bağlam yönetimi metodları
    def get_context_info(self, session_id: str = "default") -> Dict:
        """Bağlam bilgilerini döndür"""
        context = self.context_manager.get_context(session_id)
        if not context:
            return {"error": "Bağlam bulunamadı"}
        
        return {
            'session_id': context.session_id,
            'user_id': context.user_id,
            'current_topic': context.current_topic,
            'message_count': len(context.conversation_flow),
            'conversation_summary': context.conversation_summary,
            'context_keywords': list(context.context_keywords),
            'emotion_trend': context._get_emotion_trend(),
            'intent_trend': context._get_intent_trend(),
            'session_duration': (datetime.datetime.now() - context.created_at).total_seconds(),
            'last_updated': context.last_updated.isoformat()
        }
    
    def analyze_conversation_context(self, session_id: str = "default") -> Dict:
        """Konuşma bağlamını analiz et"""
        return self.context_manager.analyze_context(session_id)
    
    def save_conversation_context(self, session_id: str = "default"):
        """Konuşma bağlamını kaydet"""
        self.context_manager.save_context(session_id)
        self.logger.info(f"Bağlam kaydedildi: {session_id}")
    
    def load_conversation_context(self, session_id: str = "default") -> bool:
        """Konuşma bağlamını yükle"""
        context = self.context_manager.load_context(session_id)
        if context:
            self.logger.info(f"Bağlam yüklendi: {session_id}")
            return True
        else:
            self.logger.warning(f"Bağlam bulunamadı: {session_id}")
            return False
    
    def clear_conversation_context(self, session_id: str = "default"):
        """Konuşma bağlamını temizle"""
        if session_id in self.context_manager.active_contexts:
            del self.context_manager.active_contexts[session_id]
            self.logger.info(f"Bağlam temizlendi: {session_id}")
    
    def get_context_aware_response(self, user_input: str, session_id: str = "default") -> str:
        """Bağlam farkında yanıt al"""
        return self.get_response(user_input, session_id)
    
    def get_conversation_history(self, session_id: str = "default", max_messages: int = 10) -> List[Dict]:
        """Konuşma geçmişini al"""
        context = self.context_manager.get_context(session_id)
        if not context:
            return []
        
        return context.conversation_flow[-max_messages:] if context.conversation_flow else []
    
    def update_user_preferences(self, session_id: str, preferences: Dict):
        """Kullanıcı tercihlerini güncelle"""
        context = self.context_manager.get_context(session_id)
        if context:
            context.user_preferences.update(preferences)
            self.logger.info(f"Kullanıcı tercihleri güncellendi: {session_id}")
    
    def get_user_preferences(self, session_id: str = "default") -> Dict:
        """Kullanıcı tercihlerini al"""
        context = self.context_manager.get_context(session_id)
        if context:
            return context.user_preferences
        return {}