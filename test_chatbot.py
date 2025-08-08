"""
AI Asistan Test Scripti
Chatbot'u test etmek için basit script
"""

import requests
import json
import time

def test_chatbot():
    """Chatbot'u test et"""
    base_url = "http://localhost:5000"
    
    # Test mesajları
    test_messages = [
        "Merhaba",
        "Sen kimsin?",
        "Adın ne?",
        "5 + 3",
        "Yardım",
        "Teknik sorun yaşıyorum",
        "Görüşürüz"
    ]
    
    print("🤖 AI Asistan Test Ediliyor...")
    print("=" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n📝 Test {i}: '{message}'")
        
        try:
            # Chat endpoint'ine istek gönder
            response = requests.post(
                f"{base_url}/chat",
                json={"message": message},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ Yanıt: {data['response']}")
                    
                    # Kişilik bilgilerini göster
                    if 'personality_info' in data:
                        personality = data['personality_info']
                        print(f"   😊 Ruh Hali: {personality.get('mood', 'N/A')}")
                        print(f"   ⚡ Enerji: {personality.get('energy_level', 'N/A')}/10")
                    
                    # İstatistikleri göster
                    if 'stats' in data:
                        stats = data['stats']
                        print(f"   📊 Toplam İstek: {stats.get('total_requests', 'N/A')}")
                        print(f"   ⏱️ Ortalama Süre: {stats.get('average_response_time', 'N/A')}s")
                        
                else:
                    print(f"❌ Hata: {data.get('response', 'Bilinmeyen hata')}")
            else:
                print(f"❌ HTTP Hatası: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Bağlantı hatası! Uygulama çalışıyor mu?")
            return
        except Exception as e:
            print(f"❌ Genel hata: {e}")
        
        time.sleep(1)  # Kısa bekleme
    
    print("\n" + "=" * 50)
    print("🎉 Test tamamlandı!")
    
    # API endpoint'lerini test et
    test_api_endpoints(base_url)

def test_api_endpoints(base_url):
    """API endpoint'lerini test et"""
    print("\n🔧 API Endpoint Testleri:")
    print("-" * 30)
    
    endpoints = [
        ("/api/personality/info", "Kişilik Bilgileri"),
        ("/api/stats", "İstatistikler"),
        ("/health", "Sağlık Kontrolü")
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            if response.status_code == 200:
                print(f"✅ {name}: Başarılı")
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: Hata - {e}")

if __name__ == "__main__":
    print("🚀 AI Asistan Test Scripti")
    print("Uygulamanın çalıştığından emin olun: python app.py")
    print()
    
    test_chatbot()
