#!/usr/bin/env python3
"""
Enhanced Context Manager for Advanced AI Chatbot
Provides sophisticated context tracking, entity extraction, topic analysis, and logic reasoning
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK data download
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

@dataclass
class ConversationContext:
    """Enhanced conversation context with advanced tracking"""
    session_id: str
    created_at: datetime
    last_updated: datetime
    messages: List[Dict[str, Any]] = None
    entities: Dict[str, List[str]] = None
    topics: List[str] = None
    current_topic: str = "genel"
    topic_confidence: float = 0.0
    logical_connections: List[str] = None
    questions_asked: List[str] = None
    assumptions: List[str] = None
    sentiment_history: List[Dict[str, Any]] = None
    intent_history: List[Dict[str, Any]] = None
    user_preferences: Dict[str, Any] = None
    conversation_flow: List[str] = None
    context_switches: List[Dict[str, Any]] = None
    memory_anchors: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []
        if self.entities is None:
            self.entities = {"names": [], "places": [], "things": [], "dates": [], "numbers": []}
        if self.topics is None:
            self.topics = []
        if self.logical_connections is None:
            self.logical_connections = []
        if self.questions_asked is None:
            self.questions_asked = []
        if self.assumptions is None:
            self.assumptions = []
        if self.sentiment_history is None:
            self.sentiment_history = []
        if self.intent_history is None:
            self.intent_history = []
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.conversation_flow is None:
            self.conversation_flow = []
        if self.context_switches is None:
            self.context_switches = []
        if self.memory_anchors is None:
            self.memory_anchors = {}
    
    def add_message(self, user_message: str, bot_response: str, intent: str, confidence: float, sentiment: str = "neutral"):
        """Add a new message to the context"""
        message_data = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'bot_response': bot_response,
            'intent': intent,
            'confidence': confidence,
            'sentiment': sentiment
        }
        self.messages.append(message_data)
        self.last_updated = datetime.now()
        
        # Update conversation flow
        self.conversation_flow.append(intent)
        
        # Update intent history
        self.intent_history.append({
            'intent': intent,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update sentiment history
        self.sentiment_history.append({
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_recent_context(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        return self.messages[-count:] if self.messages else []
    
    def get_topic_context(self, topic: str) -> List[Dict[str, Any]]:
        """Get context related to a specific topic"""
        return [msg for msg in self.messages if topic.lower() in msg.get('user_message', '').lower()]
    
    def get_entity_context(self, entity_type: str, entity_value: str) -> List[Dict[str, Any]]:
        """Get context related to a specific entity"""
        return [msg for msg in self.messages if entity_value.lower() in msg.get('user_message', '').lower()]
    
    def add_memory_anchor(self, key: str, value: Any, importance: float = 0.5):
        """Add a memory anchor for important information"""
        self.memory_anchors[key] = {
            'value': value,
            'importance': importance,
            'timestamp': datetime.now().isoformat(),
            'access_count': 0
        }
    
    def get_memory_anchor(self, key: str) -> Optional[Any]:
        """Retrieve a memory anchor and update access count"""
        if key in self.memory_anchors:
            self.memory_anchors[key]['access_count'] += 1
            return self.memory_anchors[key]['value']
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        return data

class EnhancedEntityExtractor:
    """Advanced entity extraction with improved patterns and ML-like features"""
    
    def __init__(self):
        self.name_patterns = [
            # Turkish name patterns
            r'\b(?:ben|benim adım|adım|ismim)\s+([A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)',
            r'\b([A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)\s+(?:adında|isminde|denir|diyor)',
            r'\b([A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)\s+(?:benim|ben|kişi|adam|kadın)',
            r'\b(?:ben|benim)\s+([A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)',
            # English name patterns
            r'\b(?:my name is|i am|i\'m)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is my name|is me)',
            r'\b(?:call me|i\'m called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        self.place_patterns = [
            # Turkish place patterns
            r'\b([A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)\s+(?:şehri|kenti|il|ilçe|mahalle|sokak|cadde|bulvar)',
            r'\b([A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)\s+(?:da|de|ta|te)\s+(?:yaşıyorum|oturuyorum|çalışıyorum|bulunuyorum)',
            r'\b([A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)\s+(?:gidiyorum|gittim|gideceğim|ziyaret|seyahat)',
            r'\b(?:[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)\s+(?:da|de|ta|te)\s+(?:yaşıyorum|oturuyorum|çalışıyorum)',
            # English place patterns
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:city|town|village|country|state|province)',
            r'\b(?:i live in|i work in|i\'m from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is where|is my)'
        ]
        
        self.thing_patterns = [
            # Turkish thing patterns
            r'\b([A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)\s+(?:alacağım|aldım|satın aldım|kullandım)',
            r'\b([A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)\s+(?:kullanıyorum|kullandım|kullanacağım|öğreniyorum)',
            r'\b([A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)\s+(?:öğreniyorum|öğrendim|öğreneceğim|biliyorum)',
            r'\b(?:[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+(?:\s+[A-ZÇĞIİÖŞÜ][a-zçğıiöşü]+)*)\s+(?:alacağım|aldım|satın aldım)',
            # English thing patterns
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:i bought|i use|i learned|i know)',
            r'\b(?:i have|i own|i use)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is what|is my)'
        ]
        
        self.date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b(\d{1,2}\s+(?:ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+\d{2,4})\b',
            r'\b(\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{2,4})\b',
            r'\b(?:bugün|dün|yarın|geçen hafta|gelecek ay)\b',
            r'\b(?:today|yesterday|tomorrow|last week|next month)\b'
        ]
        
        self.number_patterns = [
            r'\b(\d+(?:\.\d+)?)\b',
            r'\b(\d+\s+(?:bin|milyon|milyar|trilyon))\b',
            r'\b(\d+\s+(?:thousand|million|billion|trillion))\b'
        ]
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('turkish') + stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities with enhanced patterns and NLP processing"""
        entities = {
            "names": [],
            "places": [],
            "things": [],
            "dates": [],
            "numbers": []
        }
        
        # Extract names
        for pattern in self.name_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["names"].extend([match.strip() for match in matches if match.strip()])
        
        # Extract places
        for pattern in self.place_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["places"].extend([match.strip() for match in matches if match.strip()])
        
        # Extract things
        for pattern in self.thing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["things"].extend([match.strip() for match in matches if match.strip()])
        
        # Extract dates
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["dates"].extend([match.strip() for match in matches if match.strip()])
        
        # Extract numbers
        for pattern in self.number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["numbers"].extend([match.strip() for match in matches if match.strip()])
        
        # Remove duplicates and empty strings
        for key in entities:
            entities[key] = list(set([item for item in entities[key] if item]))
        
        return entities
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        tokens = word_tokenize(text.lower())
        keywords = []
        
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                # Lemmatize the token
                lemma = self.lemmatizer.lemmatize(token)
                keywords.append(lemma)
        
        return keywords

class AdvancedTopicAnalyzer:
    """Advanced topic analysis with keyword clustering and context awareness"""
    
    def __init__(self):
        self.topic_keywords = {
            "teknoloji": ["bilgisayar", "yazılım", "programlama", "kod", "internet", "web", "uygulama", "telefon", "teknoloji", "ai", "yapay zeka"],
            "eğitim": ["öğrenmek", "eğitim", "kurs", "ders", "okul", "üniversite", "öğretmen", "öğrenci", "kitap", "çalışmak", "öğrenme"],
            "sağlık": ["sağlık", "hastane", "doktor", "ilaç", "tedavi", "hastalık", "semptom", "kontrol", "muayene", "sağlıklı"],
            "iş": ["iş", "çalışma", "ofis", "toplantı", "proje", "müşteri", "satış", "pazarlama", "yönetim", "kariyer"],
            "spor": ["spor", "futbol", "basketbol", "koşu", "egzersiz", "antrenman", "oyun", "takım", "maç", "fitness"],
            "sanat": ["sanat", "müzik", "resim", "film", "kitap", "şiir", "tarih", "kültür", "yaratıcılık", "tasarım"],
            "bilim": ["bilim", "araştırma", "deney", "laboratuvar", "keşif", "teori", "hipotez", "veri", "analiz", "sonuç"],
            "günlük": ["günlük", "rutin", "alışkanlık", "yaşam", "gün", "hafta", "ay", "zaman", "plan", "program"]
        }
        
        self.english_topic_keywords = {
            "technology": ["computer", "software", "programming", "code", "internet", "web", "app", "phone", "technology", "ai", "artificial intelligence"],
            "education": ["learn", "education", "course", "lesson", "school", "university", "teacher", "student", "book", "study", "learning"],
            "health": ["health", "hospital", "doctor", "medicine", "treatment", "disease", "symptom", "check", "examination", "healthy"],
            "business": ["work", "job", "office", "meeting", "project", "client", "sales", "marketing", "management", "career"],
            "sports": ["sport", "football", "basketball", "running", "exercise", "training", "game", "team", "match", "fitness"],
            "arts": ["art", "music", "painting", "film", "book", "poetry", "history", "culture", "creativity", "design"],
            "science": ["science", "research", "experiment", "laboratory", "discovery", "theory", "hypothesis", "data", "analysis", "result"],
            "daily": ["daily", "routine", "habit", "life", "day", "week", "month", "time", "plan", "schedule"]
        }
    
    def detect_topic(self, text: str, context: Optional[Dict] = None) -> Tuple[str, float]:
        """Detect topic with confidence score"""
        text_lower = text.lower()
        scores = {}
        
        # Check Turkish topics
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[topic] = score / len(keywords)
        
        # Check English topics
        for topic, keywords in self.english_topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[topic] = score / len(keywords)
        
        # Context-aware scoring
        if context and context.get('current_topic') != "genel":
            current_topic = context['current_topic']
            if current_topic in scores:
                scores[current_topic] *= 1.5  # Boost current topic
        
        if scores:
            best_topic = max(scores, key=scores.get)
            confidence = min(1.0, scores[best_topic])
            return best_topic, confidence
        
        return "genel", 0.0
    
    def get_related_topics(self, topic: str) -> List[str]:
        """Get related topics for better context switching"""
        related_topics = {
            "teknoloji": ["eğitim", "iş", "bilim"],
            "eğitim": ["teknoloji", "bilim", "sanat"],
            "sağlık": ["bilim", "günlük"],
            "iş": ["teknoloji", "eğitim", "günlük"],
            "spor": ["sağlık", "günlük"],
            "sanat": ["eğitim", "kültür", "günlük"],
            "bilim": ["teknoloji", "eğitim", "sağlık"],
            "günlük": ["sağlık", "iş", "spor"]
        }
        return related_topics.get(topic, [])

class IntelligentLogicReasoner:
    """Advanced logic reasoning with causal analysis and inference"""
    
    def __init__(self):
        self.causal_words = {
            "türkçe": ["çünkü", "çünkü ki", "zira", "nitekim", "şöyle ki", "şu sebepten", "bu yüzden", "bu nedenle", "bu sebeple"],
            "english": ["because", "since", "as", "for", "therefore", "thus", "hence", "consequently", "as a result"]
        }
        
        self.question_words = {
            "türkçe": ["ne", "kim", "nerede", "ne zaman", "nasıl", "neden", "niçin", "hangi", "kaç", "nereden"],
            "english": ["what", "who", "where", "when", "how", "why", "which", "how many", "how much", "where from"]
        }
        
        self.assumption_words = {
            "türkçe": ["sanırım", "galiba", "muhtemelen", "belki", "olabilir", "eğer", "varsayalım ki", "farz edelim ki"],
            "english": ["i think", "maybe", "probably", "perhaps", "possibly", "if", "suppose", "assume", "let's say"]
        }
    
    def analyze_logic(self, text: str) -> Dict[str, Any]:
        """Analyze logical structure of the text"""
        text_lower = text.lower()
        
        # Detect causal relations
        causal_relations = []
        for lang, words in self.causal_words.items():
            for word in words:
                if word in text_lower:
                    causal_relations.append(f"Causal relation detected with '{word}'")
        
        # Detect questions
        questions = []
        for lang, words in self.question_words.items():
            for word in words:
                if word in text_lower:
                    questions.append(f"Question detected with '{word}'")
        
        # Detect assumptions
        assumptions = []
        for lang, words in self.assumption_words.items():
            for word in words:
                if word in text_lower:
                    assumptions.append(f"Assumption detected with '{word}'")
        
        # Detect logical operators
        logical_operators = []
        logical_words = ["ve", "veya", "ama", "fakat", "ancak", "and", "or", "but", "however", "although"]
        for word in logical_words:
            if word in text_lower:
                logical_operators.append(f"Logical operator: '{word}'")
        
        return {
            'causal_relations': causal_relations,
            'questions': questions,
            'assumptions': assumptions,
            'logical_operators': logical_operators,
            'complexity_score': len(causal_relations) + len(questions) + len(assumptions) + len(logical_operators)
        }
    
    def infer_context(self, current_context: Dict, new_message: str) -> Dict[str, Any]:
        """Infer new context based on current context and new message"""
        inferences = {
            'topic_continuation': False,
            'context_switch': False,
            'new_entities': [],
            'repeated_concepts': [],
            'logical_connections': []
        }
        
        # Check for topic continuation
        if current_context.get('current_topic') != "genel":
            current_topic = current_context['current_topic']
            if current_topic.lower() in new_message.lower():
                inferences['topic_continuation'] = True
        
        # Check for context switches
        if current_context.get('topics'):
            recent_topics = current_context['topics'][-3:]
            for topic in recent_topics:
                if topic.lower() in new_message.lower():
                    inferences['repeated_concepts'].append(topic)
        
        # Check for new entities
        if current_context.get('entities'):
            for entity_type, entities in current_context['entities'].items():
                for entity in entities:
                    if entity.lower() in new_message.lower():
                        inferences['logical_connections'].append(f"Entity reference: {entity}")
        
        return inferences

class ContextManager:
    """Enhanced context manager with advanced features"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.contexts: Dict[str, ConversationContext] = {}
        self.entity_extractor = EnhancedEntityExtractor()
        self.topic_analyzer = AdvancedTopicAnalyzer()
        self.logic_reasoner = IntelligentLogicReasoner()
        self.context_file = Path("data/conversation_contexts.json")
        self._load_contexts()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for context manager"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def get_context(self, session_id: str) -> ConversationContext:
        """Get or create context for a session"""
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext(
                session_id=session_id,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
        return self.contexts[session_id]
    
    def add_message(self, session_id: str, message_type: str, content: str):
        """Add a simple message to context (for backward compatibility)"""
        context = self.get_context(session_id)
        
        if message_type == 'user':
            # Add user message
            context.add_message(content, "", "general", 0.5, "neutral")
        elif message_type == 'bot':
            # Update last bot response if exists
            if context.messages:
                context.messages[-1]['bot_response'] = content
            else:
                # Create a dummy user message if none exists
                context.add_message("", content, "general", 0.5, "neutral")
        
        self._save_contexts()
    
    def update_context(self, session_id: str, user_message: str, bot_response: str, intent: str, confidence: float, sentiment: str = "neutral"):
        """Update context with new conversation data"""
        context = self.get_context(session_id)
        
        # Add message to context
        context.add_message(user_message, bot_response, intent, confidence, sentiment)
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(user_message)
        for entity_type, entity_list in entities.items():
            context.entities[entity_type].extend(entity_list)
            context.entities[entity_type] = list(set(context.entities[entity_type]))  # Remove duplicates
        
        # Detect topic
        topic, confidence_score = self.topic_analyzer.detect_topic(user_message, context.to_dict())
        if topic != "genel":
            if topic not in context.topics:
                context.topics.append(topic)
            context.current_topic = topic
            context.topic_confidence = confidence_score
        
        # Analyze logic
        logic_analysis = self.logic_reasoner.analyze_logic(user_message)
        context.logical_connections.extend(logic_analysis['causal_relations'])
        context.questions_asked.extend(logic_analysis['questions'])
        context.assumptions.extend(logic_analysis['assumptions'])
        
        # Infer new context
        inferences = self.logic_reasoner.infer_context(context.to_dict(), user_message)
        if inferences['topic_continuation']:
            context.context_switches.append({
                'type': 'topic_continuation',
                'topic': context.current_topic,
                'timestamp': datetime.now().isoformat()
            })
        
        # Add memory anchors for important information
        if intent in ['chatbot_identity', 'user_preference']:
            context.add_memory_anchor(f"intent_{intent}", user_message, importance=0.8)
        
        # Save context
        self._save_contexts()
        self.logger.info(f"🔄 Enhanced context updated: {session_id}")
    
    def get_contextual_response(self, session_id: str, base_response: str, user_message: str) -> str:
        """Generate context-aware response"""
        context = self.get_context(session_id)
        
        # Check for repeated questions
        if self._is_repeated_question(user_message, context.get_recent_context(5)):
            return "Bu soruyu daha önce sormuştunuz. Başka bir konuda yardımcı olabilir miyim? 🤔"
        
        # Check for entity references
        entities = self.entity_extractor.extract_entities(user_message)
        for entity_type, entity_list in entities.items():
            if entity_list:
                for entity in entity_list:
                    entity_context = context.get_entity_context(entity_type, entity)
                    if entity_context:
                        return f"{base_response} {entity} hakkında daha önce konuşmuştuk. Size nasıl yardımcı olabilirim? 🤔"
        
        # Check for topic continuation
        if context.current_topic != "genel":
            related_topics = self.topic_analyzer.get_related_topics(context.current_topic)
            if any(topic in user_message.lower() for topic in related_topics):
                return f"{base_response} {context.current_topic} konusundan bahsederken ilgili bir konuya geçiş yaptınız. Devam etmek ister misiniz? 🎯"
        
        # Check memory anchors
        for key, anchor_data in context.memory_anchors.items():
            if anchor_data['importance'] > 0.7 and anchor_data['value'].lower() in user_message.lower():
                return f"{base_response} Daha önce bu konudan bahsetmiştik. Size daha spesifik yardım edebilir miyim? 💡"
        
        return base_response
    
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive context summary"""
        context = self.get_context(session_id)
        
        # Calculate context statistics
        total_messages = len(context.messages)
        unique_topics = len(set(context.topics))
        total_entities = sum(len(entities) for entities in context.entities.values())
        
        # Get recent activity
        recent_activity = []
        for msg in context.messages[-5:]:
            recent_activity.append({
                'intent': msg.get('intent', 'unknown'),
                'sentiment': msg.get('sentiment', 'neutral'),
                'timestamp': msg.get('timestamp', '')
            })
        
        # Get topic distribution
        topic_distribution = {}
        if context.topics:
            topic_counter = Counter(context.topics)
            topic_distribution = dict(topic_counter)
        
        return {
            'session_id': session_id,
            'message_count': total_messages,
            'current_topic': context.current_topic,
            'topic_confidence': context.topic_confidence,
            'topics_discussed': context.topics,
            'topic_distribution': topic_distribution,
            'entities_found': context.entities,
            'total_entities': total_entities,
            'logical_connections': context.logical_connections[-10:],  # Last 10
            'questions_asked': context.questions_asked[-10:],  # Last 10
            'assumptions': context.assumptions[-10:],  # Last 10
            'recent_activity': recent_activity,
            'memory_anchors_count': len(context.memory_anchors),
            'context_switches_count': len(context.context_switches),
            'created_at': context.created_at.isoformat(),
            'last_updated': context.last_updated.isoformat()
        }
    
    def clear_context(self, session_id: str):
        """Clear context for a session"""
        if session_id in self.contexts:
            del self.contexts[session_id]
            self._save_contexts()
            self.logger.info(f"🗑️ Context cleared: {session_id}")
    
    def _is_repeated_question(self, current_message: str, recent_messages: List[Dict]) -> bool:
        """Check if current message is similar to recent messages"""
        current_lower = current_message.lower()
        
        for msg in recent_messages:
            user_msg = msg.get('user_message', '').lower()
            if user_msg and self._calculate_similarity(current_lower, user_msg) > 0.7:
                return True
        
        return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _load_contexts(self):
        """Load contexts from file"""
        try:
            if self.context_file.exists():
                with open(self.context_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for session_id, context_data in data.items():
                        # Convert string timestamps back to datetime
                        if 'created_at' in context_data:
                            context_data['created_at'] = datetime.fromisoformat(context_data['created_at'])
                        if 'last_updated' in context_data:
                            context_data['last_updated'] = datetime.fromisoformat(context_data['last_updated'])
                        
                        context = ConversationContext(**context_data)
                        self.contexts[session_id] = context
                
                self.logger.info(f"📂 Loaded {len(self.contexts)} contexts from file")
        except Exception as e:
            self.logger.error(f"❌ Error loading contexts: {e}")
    
    def _save_contexts(self):
        """Save contexts to file"""
        try:
            data = {}
            for session_id, context in self.contexts.items():
                data[session_id] = context.to_dict()
            
            with open(self.context_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 Saved {len(self.contexts)} contexts to file")
        except Exception as e:
            self.logger.error(f"❌ Error saving contexts: {e}")
