#!/usr/bin/env python3
"""
Advanced Chatbot Test Suite
Gelişmiş chatbot'un tüm model boyutlarını test eder
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.advanced_chatbot import AdvancedChatbot
import time
import os

def test_model_sizes():
    """Farklı model boyutlarını test et"""
    model_sizes = ["small", "medium", "large"]
    
    print("🧪 Gelişmiş Chatbot - Model Boyutu Testleri")
    print("=" * 60)
    
    results = {}
    
    for model_size in model_sizes:
        print(f"\n🔍 {model_size.upper()} Model Test Ediliyor...")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Chatbot'u başlat
            chatbot = AdvancedChatbot(model_size=model_size)
            
            # İstatistikleri al
            stats = chatbot.get_stats()
            
            # Performans testi
            evaluation = chatbot.evaluate_model()
            
            load_time = time.time() - start_time
            
            results[model_size] = {
                'load_time': load_time,
                'stats': stats,
                'evaluation': evaluation,
                'success': True
            }
            
            print(f"✅ Model başarıyla yüklendi ({load_time:.2f}s)")
            print(f"   - Toplam Intent: {stats['total_intents']}")
            print(f"   - ML Aktif: {stats['ml_available']}")
            print(f"   - Güven Eşiği: {stats['confidence_threshold']}")
            
            if 'accuracy' in evaluation:
                print(f"   - Doğruluk: {evaluation['accuracy']:.3f}")
            
        except Exception as e:
            results[model_size] = {
                'error': str(e),
                'success': False
            }
            print(f"❌ Model yüklenemedi: {e}")
    
    # Sonuçları karşılaştır
    print("\n" + "=" * 60)
    print("📊 Model Karşılaştırması:")
    
    for model_size, result in results.items():
        if result['success']:
            stats = result['stats']
            eval_result = result['evaluation']
            
            print(f"\n🏷️ {model_size.upper()} Model:")
            print(f"   Yükleme Süresi: {result['load_time']:.2f}s")
            print(f"   Intent Sayısı: {stats['total_intents']}")
            print(f"   Güven Eşiği: {stats['confidence_threshold']}")
            
            if 'accuracy' in eval_result:
                print(f"   Doğruluk: {eval_result['accuracy']:.3f}")
        else:
            print(f"\n❌ {model_size.upper()} Model: {result['error']}")

def test_conversation_flow():
    """Konuşma akışını test et"""
    print("\n🎯 Konuşma Akışı Testi")
    print("=" * 40)
    
    chatbot = AdvancedChatbot(model_size="medium")
    
    test_conversations = [
        "Merhaba",
        "Nasılsın?",
        "5 + 3 kaç eder?",
        "Şaka yap",
        "Müzik öner",
        "Teşekkürler",
        "Görüşürüz"
    ]
    
    for i, message in enumerate(test_conversations, 1):
        print(f"\n{i}. 👤 Kullanıcı: {message}")
        
        response = chatbot.get_response(message)
        intent, confidence = chatbot.predict_intent(message)
        
        print(f"   🤖 Bot: {response}")
        print(f"   🎯 Intent: {intent} (Güven: {confidence:.3f})")

def test_data_sources():
    """Veri kaynaklarını test et"""
    print("\n📊 Veri Kaynakları Testi")
    print("=" * 40)
    
    # Veri dosyalarını kontrol et
    data_files = [
        "data/training_datasets.json",
        "config/model_config.json"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - Mevcut")
        else:
            print(f"❌ {file_path} - Bulunamadı")

def interactive_advanced_test():
    """İnteraktif gelişmiş test"""
    print("\n🎮 İnteraktif Gelişmiş Test Modu")
    print("Komutlar:")
    print("  - 'switch <size>': Model boyutunu değiştir (small/medium/large)")
    print("  - 'stats': İstatistikleri göster")
    print("  - 'eval': Model değerlendirmesi")
    print("  - 'retrain': Modeli yeniden eğit")
    print("  - 'quit': Çıkış")
    print("-" * 40)
    
    current_size = "medium"
    chatbot = AdvancedChatbot(model_size=current_size)
    
    while True:
        user_input = input(f"\n👤 [{current_size}] Siz: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'çık']:
            print("👋 Görüşürüz!")
            break
        
        if user_input.lower().startswith('switch '):
            new_size = user_input.split(' ', 1)[1].strip()
            if new_size in ['small', 'medium', 'large', 'enterprise']:
                try:
                    print(f"🔄 {new_size} modeline geçiliyor...")
                    chatbot = AdvancedChatbot(model_size=new_size)
                    current_size = new_size
                    print(f"✅ {new_size} modeli yüklendi!")
                except Exception as e:
                    print(f"❌ Model değiştirilemedi: {e}")
            else:
                print("❌ Geçersiz model boyutu! (small/medium/large/enterprise)")
            continue
        
        if user_input.lower() == 'stats':
            stats = chatbot.get_stats()
            print("📊 İstatistikler:")
            for key, value in stats.items():
                if key != 'model_metadata':
                    print(f"   {key}: {value}")
            continue
        
        if user_input.lower() == 'eval':
            evaluation = chatbot.evaluate_model()
            if 'error' not in evaluation:
                print(f"📊 Doğruluk: {evaluation['accuracy']:.3f}")
                print(f"   Doğru: {evaluation['correct']}/{evaluation['total']}")
            else:
                print(f"❌ {evaluation['error']}")
            continue
        
        if user_input.lower() == 'retrain':
            print("🔄 Model yeniden eğitiliyor...")
            try:
                chatbot.retrain_model()
                print("✅ Model yeniden eğitildi!")
            except Exception as e:
                print(f"❌ Yeniden eğitim hatası: {e}")
            continue
        
        if not user_input:
            continue
        
        # Normal yanıt
        response = chatbot.get_response(user_input)
        print(f"🤖 Bot: {response}")
        
        # ML bilgilerini göster
        if chatbot.classifier:
            intent, confidence = chatbot.predict_intent(user_input)
            print(f"🎯 Intent: {intent} (Güven: {confidence:.3f})")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            interactive_advanced_test()
        elif sys.argv[1] == "conversation":
            test_conversation_flow()
        elif sys.argv[1] == "data":
            test_data_sources()
        elif sys.argv[1] == "models":
            test_model_sizes()
        else:
            print("Kullanım: python test_advanced.py [interactive|conversation|data|models]")
    else:
        # Tüm testleri çalıştır
        test_data_sources()
        test_model_sizes()
        test_conversation_flow()