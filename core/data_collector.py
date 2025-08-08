#!/usr/bin/env python3
"""
Data Collector - Gelişmiş Veri Toplama ve Eğitim Sistemi
Bu modül farklı kaynaklardan veri toplar ve chatbot'u eğitir
"""

import json
import requests
import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import random

class DataCollector:
    """Gelişmiş veri toplama ve eğitim sistemi"""
    
    def __init__(self, data_path: str = 'data/training_datasets.json'):
        self.data_path = data_path
        self.setup_logging()
        
        # API kaynakları
        self.api_sources = {
            'intents': [
                'https://api.github.com/search/repositories?q=chatbot+intent',
                'https://api.github.com/search/repositories?q=nlp+training+data'
            ],
            'conversations': [
                'https://api.github.com/search/repositories?q=conversation+dataset',
                'https://api.github.com/search/repositories?q=dialogue+corpus'
            ]
        }
        
        # Önceden eğitilmiş model kaynakları
        self.pretrained_sources = {
            'bert': 'https://huggingface.co/bert-base-multilingual-cased',
            'gpt': 'https://huggingface.co/gpt2',
            't5': 'https://huggingface.co/t5-base'
        }
    
    def setup_logging(self):
        """Logging ayarlarını yapılandır"""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('DataCollector')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            file_handler = logging.FileHandler(f'{log_dir}/data_collector.log', encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def collect_from_api(self, source_type: str) -> List[Dict]:
        """API'lerden veri topla"""
        collected_data = []
        
        try:
            for api_url in self.api_sources.get(source_type, []):
                self.logger.info(f"API'den veri toplanıyor: {api_url}")
                
                response = requests.get(api_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    collected_data.extend(self._process_api_data(data))
                    
        except Exception as e:
            self.logger.error(f"API veri toplama hatası: {e}")
        
        return collected_data
    
    def _process_api_data(self, data: Dict) -> List[Dict]:
        """API verilerini işle"""
        processed_data = []
        
        try:
            if 'items' in data:
                for item in data['items']:
                    if 'description' in item and item['description']:
                        processed_data.append({
                            'text': item['description'],
                            'source': 'github_api',
                            'timestamp': datetime.now().isoformat()
                        })
        except Exception as e:
            self.logger.error(f"API veri işleme hatası: {e}")
        
        return processed_data
    
    def collect_from_file(self, file_path: str) -> List[Dict]:
        """Dosyadan veri topla"""
        collected_data = []
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    collected_data = self._process_file_data(data, file_path)
                    
        except Exception as e:
            self.logger.error(f"Dosya veri toplama hatası: {e}")
        
        return collected_data
    
    def _process_file_data(self, data: Dict, source: str) -> List[Dict]:
        """Dosya verilerini işle"""
        processed_data = []
        
        try:
            if isinstance(data, dict):
                for intent, intent_data in data.items():
                    if 'patterns' in intent_data:
                        for pattern in intent_data['patterns']:
                            processed_data.append({
                                'text': pattern,
                                'intent': intent,
                                'source': source,
                                'timestamp': datetime.now().isoformat()
                            })
        except Exception as e:
            self.logger.error(f"Dosya veri işleme hatası: {e}")
        
        return processed_data
    
    def collect_from_web(self, urls: List[str]) -> List[Dict]:
        """Web'den veri topla"""
        collected_data = []
        
        try:
            for url in urls:
                self.logger.info(f"Web'den veri toplanıyor: {url}")
                
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Basit HTML parsing
                    text_content = self._extract_text_from_html(response.text)
                    if text_content:
                        collected_data.append({
                            'text': text_content,
                            'source': url,
                            'timestamp': datetime.now().isoformat()
                        })
                        
        except Exception as e:
            self.logger.error(f"Web veri toplama hatası: {e}")
        
        return collected_data
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """HTML'den metin çıkar"""
        import re
        
        # Basit HTML tag temizleme
        clean_text = re.sub(r'<[^>]+>', '', html_content)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text.strip()
    
    def merge_training_data(self, new_data: List[Dict]) -> Dict:
        """Yeni verileri mevcut eğitim verileriyle birleştir"""
        try:
            # Mevcut verileri yükle
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}
            
            # Yeni verileri işle
            for item in new_data:
                intent = item.get('intent', 'general')
                
                if intent not in existing_data:
                    existing_data[intent] = {
                        'patterns': [],
                        'responses': []
                    }
                
                if 'text' in item:
                    existing_data[intent]['patterns'].append(item['text'])
                    
                    # Otomatik yanıt oluştur
                    response = self._generate_response(item['text'], intent)
                    if response not in existing_data[intent]['responses']:
                        existing_data[intent]['responses'].append(response)
            
            return existing_data
            
        except Exception as e:
            self.logger.error(f"Veri birleştirme hatası: {e}")
            return {}
    
    def _generate_response(self, text: str, intent: str) -> str:
        """Metin için otomatik yanıt oluştur"""
        responses = {
            'greeting': [
                'Merhaba! Size nasıl yardımcı olabilirim?',
                'Selam! Bugün nasılsınız?',
                'Merhaba! Hoş geldiniz!'
            ],
            'farewell': [
                'Görüşürüz! İyi günler!',
                'Hoşça kalın!',
                'Tekrar görüşmek üzere!'
            ],
            'help': [
                'Size nasıl yardımcı olabilirim?',
                'Hangi konuda yardıma ihtiyacınız var?',
                'Lütfen sorunuzu belirtin.'
            ],
            'general': [
                'Anlıyorum.',
                'İlginç bir konu.',
                'Devam edin, dinliyorum.'
            ]
        }
        
        return random.choice(responses.get(intent, responses['general']))
    
    def save_enhanced_data(self, data: Dict, output_path: str = None):
        """Geliştirilmiş verileri kaydet"""
        if output_path is None:
            output_path = self.data_path
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Geliştirilmiş veriler kaydedildi: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Veri kaydetme hatası: {e}")
    
    def collect_all_sources(self) -> Dict:
        """Tüm kaynaklardan veri topla"""
        self.logger.info("🔍 Tüm kaynaklardan veri toplanıyor...")
        
        all_data = []
        
        # 1. Mevcut dosyadan veri topla
        existing_data = self.collect_from_file(self.data_path)
        all_data.extend(existing_data)
        
        # 2. API'lerden veri topla
        api_data = self.collect_from_api('intents')
        all_data.extend(api_data)
        
        # 3. Web'den veri topla (örnek URL'ler)
        web_urls = [
            'https://raw.githubusercontent.com/chatbot-datasets/chatbot-datasets/master/data/chatbot-datasets.json'
        ]
        web_data = self.collect_from_web(web_urls)
        all_data.extend(web_data)
        
        # 4. Verileri birleştir
        merged_data = self.merge_training_data(all_data)
        
        self.logger.info(f"✅ Toplam {len(all_data)} veri öğesi toplandı")
        
        return merged_data
    
    def enhance_with_pretrained_data(self, model_name: str = 'bert') -> Dict:
        """Önceden eğitilmiş model verileriyle geliştir"""
        self.logger.info(f"🧠 {model_name} modeli ile veri geliştiriliyor...")
        
        # Örnek gelişmiş veriler
        enhanced_data = {
            'advanced_greeting': {
                'patterns': [
                    'Merhaba, nasılsınız?',
                    'Selam, bugün nasıl gidiyor?',
                    'Günaydın, size nasıl yardımcı olabilirim?',
                    'İyi günler, hoş geldiniz!',
                    'Merhaba, ne yapıyorsunuz?'
                ],
                'responses': [
                    'Merhaba! Ben gelişmiş AI asistanınız. Size nasıl yardımcı olabilirim?',
                    'Selam! Bugün size nasıl yardımcı olabilirim?',
                    'Günaydın! Size nasıl hizmet verebilirim?',
                    'Merhaba! Hoş geldiniz. Size nasıl yardımcı olabilirim?'
                ]
            },
            'technical_support': {
                'patterns': [
                    'Teknik sorun yaşıyorum',
                    'Sistem hatası alıyorum',
                    'Yazılım problemi var',
                    'Kod hatası',
                    'Debug yapmam gerekiyor'
                ],
                'responses': [
                    'Teknik sorununuzu detaylandırabilir misiniz? Size yardımcı olmaya çalışacağım.',
                    'Hangi teknoloji ile ilgili sorun yaşıyorsunuz?',
                    'Hata mesajını paylaşabilir misiniz?',
                    'Sorununuzu adım adım açıklayabilir misiniz?'
                ]
            },
            'learning_request': {
                'patterns': [
                    'Yeni bir şey öğrenmek istiyorum',
                    'Eğitim almak istiyorum',
                    'Kurs arıyorum',
                    'Öğrenme kaynakları',
                    'Tutorial istiyorum'
                ],
                'responses': [
                    'Hangi konuda öğrenmek istiyorsunuz? Size uygun kaynaklar önerebilirim.',
                    'Hangi seviyede bilginiz var? Başlangıç, orta, ileri?',
                    'Öğrenme hedefiniz nedir? Size uygun eğitim programı önerebilirim.',
                    'Hangi alanda kendinizi geliştirmek istiyorsunuz?'
                ]
            }
        }
        
        return enhanced_data
    
    def create_training_report(self, data: Dict) -> Dict:
        """Eğitim raporu oluştur"""
        total_intents = len(data)
        total_patterns = sum(len(intent_data.get('patterns', [])) for intent_data in data.values())
        total_responses = sum(len(intent_data.get('responses', [])) for intent_data in data.values())
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_intents': total_intents,
            'total_patterns': total_patterns,
            'total_responses': total_responses,
            'intent_details': {}
        }
        
        for intent, intent_data in data.items():
            report['intent_details'][intent] = {
                'patterns_count': len(intent_data.get('patterns', [])),
                'responses_count': len(intent_data.get('responses', []))
            }
        
        return report
    
    def run_full_collection(self):
        """Tam veri toplama işlemini çalıştır"""
        self.logger.info("🚀 Tam veri toplama işlemi başlatılıyor...")
        
        # 1. Tüm kaynaklardan veri topla
        collected_data = self.collect_all_sources()
        
        # 2. Önceden eğitilmiş model verileriyle geliştir
        enhanced_data = self.enhance_with_pretrained_data()
        
        # 3. Verileri birleştir
        final_data = self.merge_training_data([
            {'intent': k, 'text': v['patterns'][0]} for k, v in enhanced_data.items()
        ])
        
        # 4. Geliştirilmiş verileri ekle
        for intent, intent_data in enhanced_data.items():
            if intent not in final_data:
                final_data[intent] = intent_data
        
        # 5. Verileri kaydet
        self.save_enhanced_data(final_data)
        
        # 6. Rapor oluştur
        report = self.create_training_report(final_data)
        
        # Raporu kaydet
        report_path = 'data/training_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📊 Eğitim raporu kaydedildi: {report_path}")
        self.logger.info(f"✅ Toplam {report['total_intents']} intent, {report['total_patterns']} pattern")
        
        return final_data

if __name__ == "__main__":
    collector = DataCollector()
    collector.run_full_collection()
