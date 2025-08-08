"""
AI Asistan Model Eğitimi
Basit ve hızlı model eğitimi
"""

import logging
from pathlib import Path
from core.model_trainer import ModelTrainer

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Ana eğitim fonksiyonu"""
    try:
        logger.info("🚀 AI Asistan Model Eğitimi Başlatılıyor...")
        
        # Gerekli dizinleri oluştur
        Path("models").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        
        # Model trainer'ı başlat
        trainer = ModelTrainer()
        
        # Model boyutları
        model_sizes = ["small", "medium"]
        
        for size in model_sizes:
            logger.info(f"📊 {size.upper()} model eğitiliyor...")
            
            try:
                # Model eğit
                trainer.train_model(size)
                logger.info(f"✅ {size.upper()} model eğitimi tamamlandı")
                
            except Exception as e:
                logger.error(f"❌ {size.upper()} model eğitimi hatası: {e}")
        
        logger.info("🎉 Model eğitimi tamamlandı!")
        logger.info("🤖 Chatbot'u başlatmak için: python app.py")
        
    except Exception as e:
        logger.error(f"❌ Genel eğitim hatası: {e}")


if __name__ == "__main__":
    main()
