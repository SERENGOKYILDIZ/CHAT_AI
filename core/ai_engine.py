#!/usr/bin/env python3
"""
AI Engine - Gelişmiş Yapay Zeka Motoru
ChatGPT benzeri gelişmiş AI özellikleri sağlar
"""

import json
import re
import random
import time
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment import SentimentIntensityAnalyzer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .logger import log_info, log_error, log_performance

class AIEngine:
    """Gelişmiş AI motoru - ChatGPT benzeri özellikler"""
    
    def __init__(self, model_size: str = "medium"):
        self.model_size = model_size
        self.start_time = time.time()
        
        # AI Components
        self.nlp_processor = NLPProcessor()
        self.knowledge_base = KnowledgeBase()
        self.reasoning_engine = ReasoningEngine()
        self.creativity_module = CreativityModule()
        self.memory_system = MemorySystem()
        
        # Configuration
        self.config = self._load_config()
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'average_response_time': 0.0,
            'knowledge_queries': 0,
            'creative_responses': 0
        }
        
        log_info(f"🚀 AI Engine başlatıldı", model_size=model_size)
    
    def _load_config(self) -> Dict[str, Any]:
        """AI konfigürasyonunu yükle"""
        try:
            config_path = Path("config/ai_config.json")
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    # Extract ai_engine section or use default
                    if 'ai_engine' in config_data:
                        return config_data['ai_engine']
                    else:
                        return self._get_default_config()
            else:
                return self._get_default_config()
        except Exception as e:
            log_error(e, context={"operation": "ai_config_loading"})
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Varsayılan AI konfigürasyonu"""
        return {
            "model_size": self.model_size,
            "max_context_length": 1000,
            "creativity_level": 0.7,
            "knowledge_threshold": 0.6,
            "reasoning_depth": 3,
            "memory_capacity": 1000,
            "response_variety": 0.8
        }
    
    def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ana AI yanıt üretimi"""
        start_time = time.time()
        
        try:
            # Input preprocessing
            processed_input = self.nlp_processor.preprocess(user_input)
            
            # Intent and entity extraction
            intent = self.nlp_processor.extract_intent(processed_input)
            entities = self.nlp_processor.extract_entities(processed_input)
            
            # Context analysis
            if context:
                context_analysis = self.reasoning_engine.analyze_context(context, processed_input)
            else:
                context_analysis = {}
            
            # Knowledge retrieval
            knowledge = self.knowledge_base.retrieve_relevant_knowledge(
                processed_input, intent, entities
            )
            
            # Response generation
            response = self._generate_intelligent_response(
                processed_input, intent, entities, knowledge, context_analysis
            )
            
            # Memory storage
            self.memory_system.store_interaction(
                user_input, response, intent, entities, context_analysis
            )
            
            # Performance tracking
            response_time = time.time() - start_time
            self._update_stats(response_time)
            
            return {
                'response': response,
                'intent': intent,
                'entities': entities,
                'confidence': knowledge.get('confidence', 0.8),
                'knowledge_sources': knowledge.get('sources', []),
                'reasoning_path': context_analysis.get('reasoning_path', []),
                'response_time': response_time,
                'creativity_level': self.config['creativity_level']
            }
            
        except Exception as e:
            log_error(e, context={"operation": "ai_response_generation", "user_input": user_input})
            return {
                'response': "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.",
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    def _generate_intelligent_response(self, processed_input: str, intent: str, 
                                     entities: List[str], knowledge: Dict[str, Any], 
                                     context_analysis: Dict[str, Any]) -> str:
        """Akıllı yanıt üretimi"""
        
        # Knowledge-based response
        if knowledge.get('confidence', 0) > self.config['knowledge_threshold']:
            response = self._generate_knowledge_based_response(knowledge, intent)
        else:
            # Creative response generation
            response = self.creativity_module.generate_creative_response(
                processed_input, intent, entities, context_analysis
            )
        
        # Context-aware enhancement
        if context_analysis.get('context_relevance', 0) > 0.5:
            response = self._enhance_with_context(response, context_analysis)
        
        # Personality and style adjustment
        response = self._apply_personality_style(response, intent)
        
        return response
    
    def _generate_knowledge_based_response(self, knowledge: Dict[str, Any], intent: str) -> str:
        """Bilgi tabanından yanıt üretimi"""
        try:
            primary_source = knowledge.get('primary_source', {})
            if primary_source:
                base_response = primary_source.get('content', '')
                
                # Intent-specific formatting
                if intent == 'question':
                    return f"{base_response}"
                elif intent == 'explanation':
                    return f"Detaylı açıklama: {base_response}"
                else:
                    return base_response
            else:
                return "Bu konuda bilgim var ama detayları bulamadım."
                
        except Exception as e:
            log_error(e, context={"operation": "knowledge_based_response"})
            return "Bilgi tabanından yanıt üretirken hata oluştu."
    
    def _enhance_with_context(self, response: str, context_analysis: Dict[str, Any]) -> str:
        """Bağlam bilgisi ile yanıtı geliştir"""
        try:
            context_info = context_analysis.get('context_info', {})
            
            if context_info.get('previous_topic'):
                response += f"\n\nBu konuyla ilgili daha önce konuşmuştuk. "
                response += f"O konuşmada {context_info['previous_topic']} hakkında bilgi vermiştim."
            
            if context_info.get('related_concepts'):
                response += f"\n\nİlgili konular: {', '.join(context_info['related_concepts'][:3])}"
            
            return response
            
        except Exception as e:
            log_error(e, context={"operation": "context_enhancement"})
            return response
    
    def _apply_personality_style(self, response: str, intent: str) -> str:
        """Kişilik ve stil uygulaması"""
        try:
            # Intent-based style adjustment
            if intent == 'casual':
                response = response.replace('.', ' 😊')
            elif intent == 'formal':
                response = response.replace('!', '.')
            elif intent == 'friendly':
                response = f"Merhaba! {response}"
            
            # Creativity injection
            if random.random() < self.config['creativity_level']:
                creative_elements = [
                    "💡 İlginç bir bilgi: ",
                    "🤔 Düşünürsek: ",
                    "✨ Özel olarak: ",
                    "🌟 Harika bir nokta: "
                ]
                if random.random() < 0.3:  # 30% chance
                    response = random.choice(creative_elements) + response
            
            return response
            
        except Exception as e:
            log_error(e, context={"operation": "personality_style_application"})
            return response
    
    def _update_stats(self, response_time: float):
        """İstatistikleri güncelle"""
        self.stats['total_requests'] += 1
        self.stats['successful_requests'] += 1
        
        # Average response time update
        current_avg = self.stats['average_response_time']
        total_requests = self.stats['total_requests']
        self.stats['average_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def get_ai_capabilities(self) -> Dict[str, Any]:
        """AI yeteneklerini listele"""
        return {
            'nlp_processing': True,
            'knowledge_retrieval': True,
            'context_understanding': True,
            'creative_response': True,
            'memory_system': True,
            'reasoning_engine': True,
            'model_size': self.model_size,
            'config': self.config
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Performans istatistiklerini getir"""
        return {
            **self.stats,
            'uptime': time.time() - self.start_time,
            'memory_usage': self.memory_system.get_memory_usage(),
            'knowledge_base_size': self.knowledge_base.get_size()
        }

class NLPProcessor:
    """Doğal dil işleme modülü"""
    
    def __init__(self):
        self.stop_words = set()
        self.lemmatizer = None
        self.sentiment_analyzer = None
        
        if ML_AVAILABLE:
            self._initialize_nlp()
    
    def _initialize_nlp(self):
        """NLP bileşenlerini başlat"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
            self.stop_words = set(stopwords.words('turkish'))
            self.lemmatizer = WordNetLemmatizer()
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
        except Exception as e:
            log_error(e, context={"operation": "nlp_initialization"})
    
    def preprocess(self, text: str) -> str:
        """Metin ön işleme"""
        try:
            # Lowercase conversion
            text = text.lower()
            
            # Remove special characters
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Tokenization and lemmatization
            if self.lemmatizer:
                tokens = word_tokenize(text)
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                         if token not in self.stop_words]
                text = ' '.join(tokens)
            
            return text.strip()
            
        except Exception as e:
            log_error(e, context={"operation": "text_preprocessing"})
            return text
    
    def extract_intent(self, text: str) -> str:
        """Metin amacını çıkar"""
        try:
            text_lower = text.lower()
            
            # Intent patterns
            intent_patterns = {
                'question': [r'\?', r'nedir', r'nasıl', r'nerede', r'ne zaman', r'kim'],
                'greeting': [r'merhaba', r'selam', r'hoş geldin', r'günaydın'],
                'farewell': [r'görüşürüz', r'hoşça kal', r'iyi günler', r'iyi akşamlar'],
                'casual': [r'nasılsın', r'naber', r'iyi misin'],
                'formal': [r'lütfen', r'rica ederim', r'teşekkürler'],
                'explanation': [r'açıkla', r'anlat', r'detay', r'bilgi']
            }
            
            for intent, patterns in intent_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        return intent
            
            return 'general'
            
        except Exception as e:
            log_error(e, context={"operation": "intent_extraction"})
            return 'general'
    
    def extract_entities(self, text: str) -> List[str]:
        """Metinden varlıkları çıkar"""
        try:
            entities = []
            
            # Named entity patterns
            entity_patterns = [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names
                r'\b\d{4}\b',  # Years
                r'\b\d+\.\d+\b',  # Numbers
                r'\b[A-Z]{2,}\b'  # Acronyms
            ]
            
            for pattern in entity_patterns:
                matches = re.findall(pattern, text)
                entities.extend(matches)
            
            return list(set(entities))
            
        except Exception as e:
            log_error(e, context={"operation": "entity_extraction"})
            return []

class KnowledgeBase:
    """Bilgi tabanı yönetimi"""
    
    def __init__(self):
        self.knowledge = {}
        self.vectorizer = None
        self._load_knowledge()
    
    def _load_knowledge(self):
        """Bilgi tabanını yükle"""
        try:
            knowledge_path = Path("data/knowledge_base.json")
            if knowledge_path.exists():
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    self.knowledge = json.load(f)
            
            # Initialize vectorizer if ML is available
            if ML_AVAILABLE and self.knowledge:
                self._initialize_vectorizer()
                
        except Exception as e:
            log_error(e, context={"operation": "knowledge_base_loading"})
    
    def _initialize_vectorizer(self):
        """TF-IDF vectorizer'ı başlat"""
        try:
            if self.knowledge:
                texts = [item['content'] for item in self.knowledge.values()]
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.vectorizer.fit(texts)
                
        except Exception as e:
            log_error(e, context={"operation": "vectorizer_initialization"})
    
    def retrieve_relevant_knowledge(self, query: str, intent: str, 
                                  entities: List[str]) -> Dict[str, Any]:
        """İlgili bilgiyi getir"""
        try:
            if not self.knowledge:
                return {'confidence': 0.0, 'sources': []}
            
            # Simple keyword matching
            relevant_items = []
            query_lower = query.lower()
            
            for key, item in self.knowledge.items():
                content = item.get('content', '').lower()
                tags = item.get('tags', [])
                
                # Content relevance
                content_score = sum(1 for word in query_lower.split() if word in content)
                
                # Tag relevance
                tag_score = sum(1 for entity in entities if entity.lower() in [tag.lower() for tag in tags])
                
                # Intent relevance
                intent_score = 1.0 if item.get('intent') == intent else 0.5
                
                total_score = content_score + tag_score + intent_score
                
                if total_score > 0:
                    relevant_items.append({
                        'key': key,
                        'item': item,
                        'score': total_score
                    })
            
            # Sort by relevance
            relevant_items.sort(key=lambda x: x['score'], reverse=True)
            
            if relevant_items:
                best_match = relevant_items[0]
                confidence = min(best_match['score'] / 10.0, 1.0)  # Normalize to 0-1
                
                return {
                    'confidence': confidence,
                    'primary_source': best_match['item'],
                    'sources': [item['item'] for item in relevant_items[:3]],
                    'total_matches': len(relevant_items)
                }
            
            return {'confidence': 0.0, 'sources': []}
            
        except Exception as e:
            log_error(e, context={"operation": "knowledge_retrieval"})
            return {'confidence': 0.0, 'sources': []}
    
    def get_size(self) -> int:
        """Bilgi tabanı boyutunu getir"""
        return len(self.knowledge)

class ReasoningEngine:
    """Akıl yürütme motoru"""
    
    def __init__(self):
        self.reasoning_patterns = self._load_reasoning_patterns()
    
    def _load_reasoning_patterns(self) -> Dict[str, Any]:
        """Akıl yürütme kalıplarını yükle"""
        return {
            'causal': ['çünkü', 'bu yüzden', 'sonuç olarak'],
            'comparative': ['benzer şekilde', 'aksine', 'karşılaştırıldığında'],
            'conditional': ['eğer', 'koşuluyla', 'şartıyla'],
            'temporal': ['önce', 'sonra', 'şimdi', 'daha önce']
        }
    
    def analyze_context(self, context, current_input: str) -> Dict[str, Any]:
        """Bağlam analizi yap"""
        try:
            analysis = {
                'context_relevance': 0.0,
                'context_info': {},
                'reasoning_path': [],
                'logical_connections': []
            }
            
            if not context:
                return analysis
            
            # Handle both ConversationContext objects and dictionaries
            if hasattr(context, 'messages'):
                # It's a ConversationContext object
                messages = context.messages if context.messages else []
            elif isinstance(context, dict):
                # It's a dictionary
                messages = context.get('messages', [])
            else:
                # Unknown type, treat as empty
                messages = []
            if messages:
                # Find relevant previous messages
                relevant_messages = []
                for msg in messages[-5:]:  # Last 5 messages
                    if self._is_relevant(msg, current_input):
                        relevant_messages.append(msg)
                
                if relevant_messages:
                    analysis['context_relevance'] = min(len(relevant_messages) / 5.0, 1.0)
                    analysis['context_info']['previous_topic'] = self._extract_topic(relevant_messages)
                    analysis['context_info']['related_concepts'] = self._extract_concepts(relevant_messages)
            
            # Logical reasoning
            analysis['logical_connections'] = self._find_logical_connections(context, current_input)
            analysis['reasoning_path'] = self._generate_reasoning_path(analysis['logical_connections'])
            
            return analysis
            
        except Exception as e:
            log_error(e, context={"operation": "context_analysis"})
            return {'context_relevance': 0.0, 'context_info': {}, 'reasoning_path': []}
    
    def _is_relevant(self, message: Dict[str, Any], current_input: str) -> bool:
        """Mesajın ilgili olup olmadığını kontrol et"""
        try:
            content = message.get('content', '').lower()
            current_lower = current_input.lower()
            
            # Simple word overlap
            common_words = set(content.split()) & set(current_lower.split())
            return len(common_words) > 0
            
        except Exception:
            return False
    
    def _extract_topic(self, messages: List[Dict[str, Any]]) -> str:
        """Mesajlardan konu çıkar"""
        try:
            if not messages:
                return ""
            
            # Simple topic extraction based on most common words
            all_words = []
            for msg in messages:
                content = msg.get('content', '')
                words = content.lower().split()
                all_words.extend(words)
            
            # Count word frequencies
            word_freq = {}
            for word in all_words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get most common word
            if word_freq:
                most_common = max(word_freq.items(), key=lambda x: x[1])
                return most_common[0]
            
            return ""
            
        except Exception:
            return ""
    
    def _extract_concepts(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Mesajlardan kavramları çıkar"""
        try:
            concepts = set()
            
            for msg in messages:
                content = msg.get('content', '')
                # Extract capitalized words as potential concepts
                words = re.findall(r'\b[A-Z][a-z]+\b', content)
                concepts.update(words)
            
            return list(concepts)[:5]  # Return top 5 concepts
            
        except Exception:
            return []
    
    def _find_logical_connections(self, context: Dict[str, Any], current_input: str) -> List[str]:
        """Mantıksal bağlantıları bul"""
        try:
            connections = []
            
            # Pattern-based logical connection detection
            for pattern_type, patterns in self.reasoning_patterns.items():
                for pattern in patterns:
                    if pattern in current_input.lower():
                        connections.append(f"{pattern_type}_reasoning")
            
            return connections
            
        except Exception:
            return []
    
    def _generate_reasoning_path(self, connections: List[str]) -> List[str]:
        """Akıl yürütme yolunu oluştur"""
        try:
            path = []
            
            for connection in connections:
                if 'causal' in connection:
                    path.append("Neden-sonuç ilişkisi tespit edildi")
                elif 'comparative' in connection:
                    path.append("Karşılaştırmalı analiz yapıldı")
                elif 'conditional' in connection:
                    path.append("Koşullu akıl yürütme uygulandı")
                elif 'temporal' in connection:
                    path.append("Zamansal sıralama analiz edildi")
            
            return path
            
        except Exception:
            return []

class CreativityModule:
    """Yaratıcılık modülü"""
    
    def __init__(self):
        self.creative_patterns = self._load_creative_patterns()
        self.creative_responses = self._load_creative_responses()
    
    def _load_creative_patterns(self) -> Dict[str, List[str]]:
        """Yaratıcı kalıpları yükle"""
        return {
            'metaphors': [
                "Bu durum tıpkı... gibi",
                "Bunu şöyle düşünebiliriz...",
                "Bir benzetme yaparsak..."
            ],
            'analogies': [
                "Benzer şekilde...",
                "Aynı mantıkla...",
                "Bu örnekte olduğu gibi..."
            ],
            'creative_questions': [
                "Peki ya şöyle düşünürsek?",
                "Bu konuyu farklı bir açıdan ele alalım mı?",
                "İlginç bir soru: ..."
            ]
        }
    
    def _load_creative_responses(self) -> Dict[str, List[str]]:
        """Yaratıcı yanıtları yükle"""
        return {
            'general': [
                "Bu konuda düşünürken ilginç bir bakış açısı geliyor aklıma...",
                "Farklı bir perspektiften bakarsak...",
                "Bu durumu analiz ederken şunu fark ettim..."
            ],
            'problem_solving': [
                "Sorunu çözmek için yaratıcı bir yaklaşım deneyelim...",
                "Standart çözümlerin yanında alternatif bir yol var...",
                "Bu problemi farklı açıdan ele alalım..."
            ]
        }
    
    def generate_creative_response(self, input_text: str, intent: str, 
                                 entities: List[str], context_analysis: Dict[str, Any]) -> str:
        """Yaratıcı yanıt üret"""
        try:
            # Base creative response
            if intent == 'question':
                base_response = random.choice(self.creative_responses['general'])
            else:
                base_response = random.choice(self.creative_responses['problem_solving'])
            
            # Add creative elements based on context
            if context_analysis.get('context_relevance', 0) > 0.5:
                base_response += " " + random.choice(self.creative_patterns['metaphors'])
            
            # Add entity-specific creativity
            if entities:
                entity = random.choice(entities)
                base_response += f" Özellikle {entity} konusunda..."
            
            # Add creative question
            if random.random() < 0.4:  # 40% chance
                creative_question = random.choice(self.creative_patterns['creative_questions'])
                base_response += f" {creative_question}"
            
            return base_response
            
        except Exception as e:
            log_error(e, context={"operation": "creative_response_generation"})
            return "Bu konuda yaratıcı bir yaklaşım geliştirmeye çalışıyorum..."

class MemorySystem:
    """Hafıza sistemi"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.short_term_memory = []
        self.long_term_memory = {}
        self.memory_index = {}
    
    def store_interaction(self, user_input: str, bot_response: str, intent: str,
                         entities: List[str], context_analysis: Dict[str, Any]):
        """Etkileşimi hafızaya kaydet"""
        try:
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input,
                'bot_response': bot_response,
                'intent': intent,
                'entities': entities,
                'context_analysis': context_analysis
            }
            
            # Store in short-term memory
            self.short_term_memory.append(interaction)
            
            # Maintain capacity
            if len(self.short_term_memory) > self.capacity:
                # Move to long-term memory
                old_interaction = self.short_term_memory.pop(0)
                self._store_in_long_term(old_interaction)
            
            # Update memory index
            self._update_memory_index(interaction)
            
        except Exception as e:
            log_error(e, context={"operation": "interaction_storage"})
    
    def _store_in_long_term(self, interaction: Dict[str, Any]):
        """Uzun süreli hafızaya kaydet"""
        try:
            # Create memory key based on content
            content_key = f"{interaction['intent']}_{hash(interaction['user_input'][:50])}"
            
            if content_key in self.long_term_memory:
                # Update existing memory
                self.long_term_memory[content_key]['frequency'] += 1
                self.long_term_memory[content_key]['last_accessed'] = interaction['timestamp']
            else:
                # Create new memory entry
                self.long_term_memory[content_key] = {
                    'interaction': interaction,
                    'frequency': 1,
                    'created': interaction['timestamp'],
                    'last_accessed': interaction['timestamp']
                }
                
        except Exception as e:
            log_error(e, context={"operation": "long_term_storage"})
    
    def _update_memory_index(self, interaction: Dict[str, Any]):
        """Hafıza indeksini güncelle"""
        try:
            # Index by entities
            for entity in interaction.get('entities', []):
                if entity not in self.memory_index:
                    self.memory_index[entity] = []
                self.memory_index[entity].append(interaction['timestamp'])
            
            # Index by intent
            intent = interaction.get('intent', '')
            if intent not in self.memory_index:
                self.memory_index[intent] = []
            self.memory_index[intent].append(interaction['timestamp'])
            
        except Exception as e:
            log_error(e, context={"operation": "memory_index_update"})
    
    def retrieve_memory(self, query: str, intent: str = None, 
                       entities: List[str] = None) -> List[Dict[str, Any]]:
        """Hafızadan ilgili bilgileri getir"""
        try:
            relevant_memories = []
            
            # Search in short-term memory
            for memory in self.short_term_memory:
                relevance_score = self._calculate_relevance(memory, query, intent, entities)
                if relevance_score > 0.5:
                    relevant_memories.append({
                        'memory': memory,
                        'relevance': relevance_score,
                        'type': 'short_term'
                    })
            
            # Search in long-term memory
            for key, memory_entry in self.long_term_memory.items():
                memory = memory_entry['interaction']
                relevance_score = self._calculate_relevance(memory, query, intent, entities)
                if relevance_score > 0.7:  # Higher threshold for long-term
                    relevant_memories.append({
                        'memory': memory,
                        'relevance': relevance_score,
                        'type': 'long_term',
                        'frequency': memory_entry['frequency']
                    })
            
            # Sort by relevance
            relevant_memories.sort(key=lambda x: x['relevance'], reverse=True)
            
            return relevant_memories[:5]  # Return top 5
            
        except Exception as e:
            log_error(e, context={"operation": "memory_retrieval"})
            return []
    
    def _calculate_relevance(self, memory: Dict[str, Any], query: str, 
                           intent: str = None, entities: List[str] = None) -> float:
        """Hafıza ilgililiğini hesapla"""
        try:
            relevance_score = 0.0
            
            # Query similarity
            user_input = memory.get('user_input', '').lower()
            query_lower = query.lower()
            
            # Word overlap
            query_words = set(query_lower.split())
            input_words = set(user_input.split())
            word_overlap = len(query_words & input_words) / max(len(query_words), 1)
            relevance_score += word_overlap * 0.4
            
            # Intent similarity
            if intent and memory.get('intent') == intent:
                relevance_score += 0.3
            
            # Entity similarity
            if entities:
                memory_entities = set(memory.get('entities', []))
                entity_overlap = len(set(entities) & memory_entities) / max(len(entities), 1)
                relevance_score += entity_overlap * 0.3
            
            return min(relevance_score, 1.0)
            
        except Exception:
            return 0.0
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Hafıza kullanım bilgilerini getir"""
        return {
            'short_term_count': len(self.short_term_memory),
            'long_term_count': len(self.long_term_memory),
            'index_size': len(self.memory_index),
            'capacity': self.capacity,
            'usage_percentage': (len(self.short_term_memory) / self.capacity) * 100
        }
