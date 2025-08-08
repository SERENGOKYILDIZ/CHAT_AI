# 🤖 AI Asistan - Gelişmiş Yapay Zeka Chatbot

Kişilik sistemi ile entegre, modern ve akıllı AI asistan.

## ✨ Özellikler

- **🎭 Kişilik Sistemi**: Dinamik ruh hali, enerji seviyesi ve iletişim tarzı
- **🧠 Makine Öğrenmesi**: Intent classification ve sentiment analysis
- **💬 Doğal Dil İşleme**: Türkçe dil desteği
- **🧮 Matematik Hesaplamaları**: Basit matematik işlemleri
- **📊 Gerçek Zamanlı İstatistikler**: Performans takibi
- **🎨 Modern Web Arayüzü**: Responsive ve kullanıcı dostu
- **💾 Veri Persistansı**: Konuşma geçmişi ve kişilik kaydetme

## 🚀 Kurulum

### Gereksinimler

```bash
Python 3.8+
```

### Kurulum Adımları

1. **Bağımlılıkları yükle:**
```bash
pip install -r requirements.txt
```

2. **NLTK verilerini indir:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

3. **Model eğitimi (opsiyonel):**
```bash
python train_model.py
```

4. **Uygulamayı başlat:**
```bash
python app.py
```

5. **Tarayıcıda aç:**
```
http://localhost:5000
```

## 📁 Proje Yapısı

```
ChatBot/
├── app.py                 # Ana Flask uygulaması
├── train_model.py         # Model eğitimi scripti
├── requirements.txt       # Python bağımlılıkları
├── README.md             # Bu dosya
├── core/                 # Çekirdek modüller
│   ├── __init__.py
│   ├── personality.py    # Kişilik sistemi
│   ├── smart_chatbot.py # Ana chatbot sınıfı
│   ├── model_trainer.py # Model eğitimi
│   ├── data_collector.py # Veri toplama
│   └── web_design.py    # Web tasarım yöneticisi
├── data/                 # Veri dosyaları
│   └── training_datasets.json
├── models/               # Eğitilmiş modeller
├── templates/            # HTML şablonları
│   └── index.html
├── tests/                # Test dosyaları
└── config/               # Konfigürasyon dosyaları
```

## 🎭 Kişilik Sistemi

AI Asistan'ın güçlü bir kişilik sistemi vardır:

### Kişilik Özellikleri
- **Ad**: AI Asistan
- **Versiyon**: 2.0
- **Kişilik Tipi**: Yardımsever ve Bilgili
- **İletişim Tarzı**: Sıcak ve profesyonel

### Dinamik Durumlar
- **Ruh Hali**: mutlu, heyecanlı, düşünceli
- **Enerji Seviyesi**: 1-10 arası
- **Heyecan Seviyesi**: 1-10 arası

### Uzmanlık Alanları
- Python Programlama
- Makine Öğrenmesi
- Web Geliştirme
- Veri Analizi
- Teknoloji Danışmanlığı
- Eğitim ve Öğretim

## 🧠 Makine Öğrenmesi

### Intent Classification
- Gelişmiş selamlaşma
- Chatbot kimliği
- Teknik destek
- Öğrenme istekleri
- Yardım talepleri
- Matematik hesaplamaları
- Vedalaşma

### Sentiment Analysis
- Pozitif duygu analizi
- Negatif duygu analizi
- Nötr duygu analizi

## 💬 Kullanım

### Web Arayüzü
1. Tarayıcıda `http://localhost:5000` adresini aç
2. Mesaj kutusuna yaz ve gönder
3. Kişilik bilgilerini sağ panelden takip et

### API Endpoints

#### Chat
```bash
POST /chat
{
    "message": "Merhaba"
}
```

#### Kişilik Bilgileri
```bash
GET /api/personality/info
```

#### İstatistikler
```bash
GET /api/stats
```

#### Konuşma Geçmişi
```bash
GET /api/conversation/history?limit=10
```

## 🎯 Örnek Kullanımlar

### Temel Sohbet
```
Kullanıcı: "Merhaba"
AI Asistan: "Merhaba! Ben AI Asistan 👋 Size nasıl yardımcı olabilirim?"

Kullanıcı: "Sen kimsin?"
AI Asistan: "Ben AI Asistan, gelişmiş bir yapay zeka chatbot'uyum. Python ve makine öğrenmesi teknolojileri ile geliştirildim. Size günlük konuşmalar, teknik destek, öğrenme yardımı ve daha birçok konuda destek olabilirim!"
```

### Matematik İşlemleri
```
Kullanıcı: "5 + 3"
AI Asistan: "5 + 3 = 8 ✅"

Kullanıcı: "10 * 5"
AI Asistan: "10 * 5 = 50 ✅"
```

### Teknik Destek
```
Kullanıcı: "Teknik sorun yaşıyorum"
AI Asistan: "Teknik sorununuzu detaylandırabilir misiniz? Size yardımcı olmaya çalışacağım."
```

## 🔧 Geliştirme

### Yeni Intent Ekleme
`data/training_datasets.json` dosyasına yeni intent ekleyin:

```json
{
  "yeni_intent": {
    "patterns": [
      "Örnek kalıp 1",
      "Örnek kalıp 2"
    ],
    "responses": [
      "Örnek yanıt 1",
      "Örnek yanıt 2"
    ]
  }
}
```

### Model Yeniden Eğitimi
```bash
python train_model.py
```

### Kişilik Özelleştirme
`core/personality.py` dosyasında kişilik özelliklerini değiştirin.

## 📊 Performans

### Model Boyutları
- **Small**: Hızlı, az bellek kullanımı
- **Medium**: Dengeli performans (varsayılan)
- **Large**: Yüksek doğruluk, daha fazla bellek
- **Enterprise**: En yüksek doğruluk

### İstatistikler
- Ortalama yanıt süresi: < 1 saniye
- Intent doğruluk oranı: %85+
- Desteklenen intent sayısı: 7+

## 🛠️ Teknolojiler

- **Backend**: Python, Flask
- **ML**: scikit-learn, NLTK
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Veri**: JSON, Pickle
- **Logging**: Python logging

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Commit yapın (`git commit -am 'Yeni özellik eklendi'`)
4. Push yapın (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🆘 Destek

Sorunlarınız için:
- Issue açın
- Email gönderin
- Dokümantasyonu inceleyin

---

**AI Asistan v2.0** - Gelişmiş Yapay Zeka Chatbot 🤖✨