#!/usr/bin/env python3
"""
Model Trainer - Gelişmiş Model Eğitim Sistemi
Bu modül farklı model türlerini eğitir ve optimize eder
"""

import json
import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import random

# ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier, RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class ModelTrainer:
    """Gelişmiş model eğitim sistemi"""
    
    def __init__(self, data_path: str = 'data/training_datasets.json', 
                 model_path: str = 'data/models'):
        self.data_path = data_path
        self.model_path = model_path
        self.setup_logging()
        
        # Model türleri
        self.model_types = {
            'naive_bayes': MultinomialNB,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'random_forest': RandomForestClassifier,
            'ensemble': VotingClassifier
        }
        
        # Model konfigürasyonları
        self.model_configs = {
            'small': {
                'max_features': 1000,
                'ngram_range': (1, 2),
                'ensemble_count': 2,
                'cv_folds': 3
            },
            'medium': {
                'max_features': 2000,
                'ngram_range': (1, 3),
                'ensemble_count': 3,
                'cv_folds': 5
            },
            'large': {
                'max_features': 5000,
                'ngram_range': (1, 4),
                'ensemble_count': 5,
                'cv_folds': 5
            },
            'enterprise': {
                'max_features': 10000,
                'ngram_range': (1, 5),
                'ensemble_count': 7,
                'cv_folds': 10
            }
        }
    
    def setup_logging(self):
        """Logging ayarlarını yapılandır"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('ModelTrainer')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            file_handler = logging.FileHandler(f'{log_dir}/model_trainer.log', encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def load_training_data(self) -> Tuple[List[str], List[str]]:
        """Eğitim verilerini yükle"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            patterns = []
            labels = []
            
            for intent, intent_data in data.items():
                if 'patterns' in intent_data:
                    patterns.extend(intent_data['patterns'])
                    labels.extend([intent] * len(intent_data['patterns']))
            
            self.logger.info(f"✅ {len(patterns)} eğitim örneği yüklendi")
            return patterns, labels
            
        except Exception as e:
            self.logger.error(f"Veri yükleme hatası: {e}")
            return [], []
    
    def preprocess_text(self, texts: List[str]) -> List[str]:
        """Metinleri ön işle"""
        if not ML_AVAILABLE:
            return texts
        
        try:
            # NLTK verilerini indir
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Basit stop words listesi (NLTK yüklenemezse)
            basic_stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            try:
                stop_words = set(stopwords.words('english') + stopwords.words('turkish'))
            except:
                stop_words = basic_stop_words
            
            try:
                lemmatizer = WordNetLemmatizer()
            except:
                lemmatizer = None
            
            processed_texts = []
            
            for text in texts:
                try:
                    # Tokenize
                    tokens = word_tokenize(text.lower())
                    
                    # Stop words'leri kaldır ve lemmatize et
                    if lemmatizer:
                        tokens = [lemmatizer.lemmatize(token) for token in tokens 
                                 if token.isalnum() and token not in stop_words]
                    else:
                        tokens = [token for token in tokens 
                                 if token.isalnum() and token not in stop_words]
                    
                    processed_texts.append(' '.join(tokens))
                except:
                    # Hata durumunda orijinal metni kullan
                    processed_texts.append(text.lower())
            
            self.logger.info("✅ Metin ön işleme tamamlandı")
            return processed_texts
            
        except Exception as e:
            self.logger.error(f"Metin ön işleme hatası: {e}")
            return texts
    
    def create_vectorizer(self, config: Dict) -> TfidfVectorizer:
        """TF-IDF vektörizer oluştur"""
        return TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=tuple(config['ngram_range']),
            stop_words='english',
            lowercase=True,
            analyzer='word'
        )
    
    def create_ensemble_classifier(self, config: Dict) -> VotingClassifier:
        """Ensemble sınıflandırıcı oluştur"""
        estimators = []
        
        # Naive Bayes
        estimators.append(('nb', MultinomialNB(alpha=0.1)))
        
        # Logistic Regression
        estimators.append(('lr', LogisticRegression(max_iter=1000, random_state=42)))
        
        # SVM (sadece büyük modeller için)
        if config['ensemble_count'] >= 3:
            estimators.append(('svm', SVC(kernel='linear', probability=True, random_state=42)))
        
        # Random Forest (sadece büyük modeller için)
        if config['ensemble_count'] >= 4:
            estimators.append(('rf', RandomForestClassifier(n_estimators=100, random_state=42)))
        
        return VotingClassifier(
            estimators=estimators[:config['ensemble_count']],
            voting='soft'
        )
    
    def train_model(self, model_size: str = 'medium') -> Dict[str, Any]:
        """Model eğit"""
        if not ML_AVAILABLE:
            self.logger.error("❌ ML kütüphaneleri bulunamadı!")
            return {}
        
        self.logger.info(f"🚀 {model_size} boyutunda model eğitiliyor...")
        
        # Konfigürasyonu al
        config = self.model_configs.get(model_size, self.model_configs['medium'])
        
        # Verileri yükle
        patterns, labels = self.load_training_data()
        if not patterns:
            self.logger.error("❌ Eğitim verisi bulunamadı!")
            return {}
        
        # Metinleri ön işle
        processed_patterns = self.preprocess_text(patterns)
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            processed_patterns, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # TF-IDF vektörizer oluştur ve eğit
        vectorizer = self.create_vectorizer(config)
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        
        # Label encoder
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Ensemble sınıflandırıcı oluştur ve eğit
        classifier = self.create_ensemble_classifier(config)
        
        # Cross-validation
        cv_scores = cross_val_score(classifier, X_train_vectorized, y_train_encoded, 
                                  cv=config['cv_folds'])
        
        # Model eğit
        classifier.fit(X_train_vectorized, y_train_encoded)
        
        # Test tahminleri
        y_pred = classifier.predict(X_test_vectorized)
        y_pred_proba = classifier.predict_proba(X_test_vectorized)
        
        # Performans değerlendirmesi
        accuracy = accuracy_score(y_test_encoded, y_pred)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Detaylı rapor
        report = classification_report(y_test_encoded, y_pred, 
                                    target_names=label_encoder.classes_, output_dict=True)
        
        # Model bilgileri
        model_info = {
            'model_size': model_size,
            'config': config,
            'vectorizer': vectorizer,
            'label_encoder': label_encoder,
            'classifier': classifier,
            'performance': {
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'classification_report': report
            },
            'data_info': {
                'total_samples': len(patterns),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'unique_intents': len(label_encoder.classes_)
            },
            'training_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"✅ Model eğitimi tamamlandı!")
        self.logger.info(f"📊 Doğruluk: {accuracy:.4f}")
        self.logger.info(f"📊 CV Ortalama: {cv_mean:.4f} ± {cv_std:.4f}")
        
        return model_info
    
    def save_model(self, model_info: Dict[str, Any], model_name: str = None):
        """Modeli kaydet"""
        if not model_info:
            self.logger.error("❌ Kaydedilecek model bulunamadı!")
            return
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_size = model_info['model_size']
            model_name = f"advanced_{model_size}_{timestamp}"
        
        model_dir = os.path.join(self.model_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Model bileşenlerini kaydet
            model_file = os.path.join(model_dir, 'model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model_info, f)
            
            # Performans raporunu kaydet
            report_file = os.path.join(model_dir, 'performance_report.json')
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(model_info['performance'], f, ensure_ascii=False, indent=2)
            
            # Model bilgilerini kaydet
            info_file = os.path.join(model_dir, 'model_info.json')
            info_to_save = {
                'model_size': model_info['model_size'],
                'config': model_info['config'],
                'data_info': model_info['data_info'],
                'training_timestamp': model_info['training_timestamp']
            }
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info_to_save, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ Model kaydedildi: {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Model kaydetme hatası: {e}")
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Modeli yükle"""
        try:
            model_file = os.path.join(model_path, 'model.pkl')
            with open(model_file, 'rb') as f:
                model_info = pickle.load(f)
            
            self.logger.info(f"✅ Model yüklendi: {model_path}")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {e}")
            return {}
    
    def evaluate_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Model performansını değerlendir"""
        if not model_info:
            return {}
        
        performance = model_info['performance']
        data_info = model_info['data_info']
        
        evaluation = {
            'accuracy': performance['accuracy'],
            'cv_mean': performance['cv_mean'],
            'cv_std': performance['cv_std'],
            'total_samples': data_info['total_samples'],
            'unique_intents': data_info['unique_intents'],
            'model_size': model_info['model_size'],
            'training_timestamp': model_info['training_timestamp']
        }
        
        # Performans seviyesi belirle
        if performance['accuracy'] >= 0.9:
            evaluation['performance_level'] = 'Excellent'
        elif performance['accuracy'] >= 0.8:
            evaluation['performance_level'] = 'Good'
        elif performance['accuracy'] >= 0.7:
            evaluation['performance_level'] = 'Fair'
        else:
            evaluation['performance_level'] = 'Poor'
        
        return evaluation
    
    def hyperparameter_tuning(self, model_size: str = 'medium') -> Dict[str, Any]:
        """Hiperparametre optimizasyonu"""
        self.logger.info(f"🔧 {model_size} modeli için hiperparametre optimizasyonu...")
        
        # Verileri yükle
        patterns, labels = self.load_training_data()
        if not patterns:
            return {}
        
        # Metinleri ön işle
        processed_patterns = self.preprocess_text(patterns)
        
        # TF-IDF vektörizer
        config = self.model_configs.get(model_size, self.model_configs['medium'])
        vectorizer = self.create_vectorizer(config)
        X_vectorized = vectorizer.fit_transform(processed_patterns)
        
        # Label encoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        
        # Grid search parametreleri
        param_grid = {
            'C': [0.1, 1, 10],
            'max_iter': [500, 1000],
            'random_state': [42]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            LogisticRegression(),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_vectorized, y_encoded)
        
        self.logger.info(f"✅ En iyi parametreler: {grid_search.best_params_}")
        self.logger.info(f"📊 En iyi skor: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def train_all_models(self):
        """Tüm model boyutlarını eğit"""
        self.logger.info("🚀 Tüm model boyutları eğitiliyor...")
        
        results = {}
        
        for model_size in self.model_configs.keys():
            self.logger.info(f"\n📦 {model_size.upper()} modeli eğitiliyor...")
            
            try:
                # Model eğit
                model_info = self.train_model(model_size)
                
                if model_info:
                    # Modeli kaydet
                    self.save_model(model_info)
                    
                    # Performansı değerlendir
                    evaluation = self.evaluate_model(model_info)
                    results[model_size] = evaluation
                    
                    self.logger.info(f"✅ {model_size} modeli tamamlandı!")
                else:
                    self.logger.error(f"❌ {model_size} modeli eğitilemedi!")
                    
            except Exception as e:
                self.logger.error(f"❌ {model_size} modeli hatası: {e}")
        
        # Sonuçları kaydet
        results_file = os.path.join(self.model_path, 'training_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📊 Tüm sonuçlar kaydedildi: {results_file}")
        
        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_all_models()
