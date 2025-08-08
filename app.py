"""
AI Asistan - Ana Uygulama
Kişilik sistemi ile entegre gelişmiş chatbot
"""

import uuid
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session
from pathlib import Path

from core.smart_chatbot import SmartChatbot
from core.web_design import WebDesignManager


# Flask uygulaması
app = Flask(__name__)
app.secret_key = 'ai_asistan_secret_key_2024'

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global chatbot instance
chatbot = None
web_design = WebDesignManager()


def initialize_chatbot():
    """Chatbot'u başlat"""
    global chatbot
    try:
        chatbot = SmartChatbot(model_size="medium")
        logger.info("✅ Chatbot başarıyla başlatıldı")
    except Exception as e:
        logger.error(f"❌ Chatbot başlatma hatası: {e}")
        chatbot = None


@app.route('/')
def home():
    """Ana sayfa"""
    try:
        return web_design.render_main_page()
    except Exception as e:
        logger.error(f"❌ Ana sayfa hatası: {e}")
        return "Hata oluştu", 500


@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id', None)

        # Session ID kontrolü
        if not session_id:
            if 'session_id' not in session:
                session['session_id'] = str(uuid.uuid4())
            session_id = session['session_id']

        # Chatbot kontrolü
        if not chatbot:
            initialize_chatbot()
            if not chatbot:
                return jsonify({
                    'success': False,
                    'response': 'Chatbot başlatılamadı. Lütfen tekrar deneyin.',
                    'error': 'Chatbot initialization failed'
                }), 500

        # Mesajı işle
        response = chatbot.get_response(user_message, session_id)
        
        # Kişilik bilgilerini al
        personality_info = chatbot.get_personality_info()
        
        # İstatistikleri al
        stats = chatbot.get_stats()

        return jsonify({
            'success': True,
            'response': response,
            'session_id': session_id,
            'personality_info': personality_info,
            'stats': {
                'total_requests': stats['total_requests'],
                'successful_requests': stats['successful_requests'],
                'average_response_time': round(stats['average_response_time'], 3)
            },
            'timestamp': chatbot.get_timestamp()
        })

    except Exception as e:
        logger.error(f"❌ Chat endpoint hatası: {e}")
        return jsonify({
            'success': False,
            'response': 'Bir hata oluştu. Lütfen tekrar deneyin.',
            'error': str(e)
        }), 500


@app.route('/api/personality/info', methods=['GET'])
def get_personality_info():
    """Kişilik bilgilerini getir"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            return jsonify(chatbot.get_personality_info())
        else:
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        logger.error(f"❌ Kişilik bilgisi hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/personality/summary', methods=['GET'])
def get_personality_summary():
    """Kişilik özetini getir"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            return jsonify({
                'summary': chatbot.get_personality_summary(),
                'timestamp': chatbot.get_timestamp()
            })
        else:
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        logger.error(f"❌ Kişilik özeti hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """İstatistikleri getir"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            return jsonify(chatbot.get_stats())
        else:
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        logger.error(f"❌ İstatistik hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/conversation/history', methods=['GET'])
def get_conversation_history():
    """Konuşma geçmişini getir"""
    try:
        limit = request.args.get('limit', 10, type=int)
        
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            history = chatbot.get_conversation_history(limit)
            return jsonify({
                'history': history,
                'count': len(history),
                'timestamp': chatbot.get_timestamp()
            })
        else:
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        logger.error(f"❌ Konuşma geçmişi hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/conversation/clear', methods=['POST'])
def clear_conversation():
    """Konuşma geçmişini temizle"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            chatbot.clear_conversation()
            return jsonify({
                'success': True,
                'message': 'Konuşma geçmişi temizlendi',
                'timestamp': chatbot.get_timestamp()
            })
        else:
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        logger.error(f"❌ Konuşma temizleme hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/help', methods=['GET'])
def get_help():
    """Yardım bilgilerini getir"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            return jsonify({
                'help_info': chatbot.get_help_info(),
                'timestamp': chatbot.get_timestamp()
            })
        else:
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        logger.error(f"❌ Yardım hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_model():
    """Model eğitimi"""
    try:
        data = request.get_json()
        model_size = data.get('model_size', 'medium')
        
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            chatbot.train_model(model_size)
            return jsonify({
                'success': True,
                'message': f'Model eğitimi tamamlandı: {model_size}',
                'timestamp': chatbot.get_timestamp()
            })
        else:
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        logger.error(f"❌ Model eğitimi hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/save', methods=['POST'])
def save_data():
    """Verileri kaydet"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            chatbot.save_conversation()
            return jsonify({
                'success': True,
                'message': 'Veriler kaydedildi',
                'timestamp': chatbot.get_timestamp()
            })
        else:
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        logger.error(f"❌ Veri kaydetme hatası: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Sağlık kontrolü"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        return jsonify({
            'status': 'healthy',
            'chatbot_loaded': chatbot is not None,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Sağlık kontrolü hatası: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.errorhandler(404)
def not_found(error):
    """404 hatası"""
    return jsonify({
        'error': 'Sayfa bulunamadı',
        'status': 404
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 hatası"""
    return jsonify({
        'error': 'Sunucu hatası',
        'status': 500
    }), 500


if __name__ == '__main__':
    # Gerekli dizinleri oluştur
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Chatbot'u başlat
    initialize_chatbot()
    
    # Uygulamayı çalıştır
    logger.info("🚀 AI Asistan başlatılıyor...")
    logger.info("📱 Web arayüzü: http://localhost:5000")
    logger.info("🔧 API endpoint: http://localhost:5000/chat")
    
    app.run(debug=True, host='0.0.0.0', port=5000)