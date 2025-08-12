#!/usr/bin/env python3
"""
Epsilon AI - Advanced AI Chatbot
Main Flask application - Central control point
"""

from flask import Flask, request, jsonify, render_template, session
from pathlib import Path
from core.smart_chatbot import SmartChatbot
from core.web_design import WebDesignManager
from core.logger import logger, log_info, log_error, log_performance, log_api_call, log_chat_interaction, log_chat_conversation, log_chat_session_summary
from core.multi_chat_system import create_session, add_message, update_title, get_session_data, delete_session, get_all_sessions, get_session_stats
import uuid
import time
import psutil
import os
import json
from datetime import datetime
import atexit
import signal
import sys

# Flask uygulaması
app = Flask(__name__)
app.secret_key = 'ai_asistan_secret_key_2024'

# Global chatbot instance
chatbot = None
web_design = WebDesignManager()

def initialize_chatbot():
    """Chatbot'u başlat"""
    global chatbot
    try:
        chatbot = SmartChatbot(model_size="medium")
        log_info("✅ Chatbot başarıyla başlatıldı", model_size="medium")
        
        # AI Engine capabilities log
        if hasattr(chatbot, 'ai_engine'):
            capabilities = chatbot.ai_engine.get_ai_capabilities()
            log_info("🚀 AI Engine yetenekleri yüklendi", capabilities=capabilities)
        
    except Exception as e:
        log_error(e, context={"operation": "chatbot_initialization"}, user_id="system")
        chatbot = None

def cleanup_on_exit():
    """Cleanup function called when application exits"""
    try:
        log_info("🔄 Uygulama kapatılıyor...")
        log_info("📊 Son sistem durumu kaydediliyor...")
        
        # Force flush all loggers
        import logging
        for logger_name in logging.root.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers:
                handler.flush()
                handler.close()
        
        log_info("✅ Uygulama güvenli şekilde kapatıldı")
        
    except Exception as e:
        print(f"❌ Cleanup hatası: {e}")

# Register cleanup function
atexit.register(cleanup_on_exit)

# Handle Windows signals
if os.name == 'nt':  # Windows
    def signal_handler(signum, frame):
        log_info(f"📡 Sinyal alındı: {signum}")
        cleanup_on_exit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

@app.route('/')
def home():
    """Ana sayfa"""
    try:
        log_info("📱 Ana sayfa ziyaret edildi")
        return web_design.render_main_page()
    except Exception as e:
        log_error(e, context={"operation": "home_page_render"}, user_id="system")
        return "Hata oluştu", 500

@app.route('/test')
def test_page():
    """Test sayfası"""
    try:
        log_info("🧪 Test sayfası ziyaret edildi")
        with open('tests/test_multi_chat.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        log_error(e, context={"operation": "test_page_render"}, user_id="system")
        return "Test sayfası yüklenemedi", 500

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint - Ana sohbet fonksiyonu"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        session_id = data.get('session_id', None)
        debug_mode = data.get('debug_mode', False)
        
        # Input validation
        if not user_message or len(user_message.strip()) == 0:
            return jsonify({
                'success': False,
                'response': 'Lütfen bir mesaj yazın.',
                'error': 'Empty message'
            }), 400
        
        if len(user_message) > 5000:  # Max message length
            return jsonify({
                'success': False,
                'response': 'Mesaj çok uzun. Maksimum 5000 karakter kullanabilirsiniz.',
                'error': 'Message too long'
            }), 400
        
        # Session ID kontrolü
        if not session_id:
            if 'session_id' not in session:
                session['session_id'] = str(uuid.uuid4())
            session_id = session['session_id']
        
        # Chatbot kontrolü
        if not chatbot:
            initialize_chatbot()
            if not chatbot:
                log_error(Exception("Chatbot initialization failed"), 
                         context={"operation": "chat_request"}, 
                         user_id=session_id)
                return jsonify({
                    'success': False,
                    'response': 'Chatbot başlatılamadı. Lütfen tekrar deneyin.',
                    'error': 'Chatbot initialization failed'
                }), 500
        
        # Multi chat system ile mesaj ekleme
        try:
            # User mesajını session'a ekle
            add_message(session_id, "user", user_message)
            
            # Mesaj işleme
            response = chatbot.get_response(user_message, session_id)
            personality_info = chatbot.get_personality_info()
            stats = chatbot.get_stats()
            
            # Bot cevabını session'a ekle
            add_message(session_id, "bot", response)
            
        except Exception as e:
            log_error(e, context={"operation": "multi_message_management"}, user_id=session_id)
            # Continue with response even if message management fails
        
        # Response time hesaplama
        response_time = (time.time() - start_time) * 1000  # milliseconds
        
        # Log chat interaction
        intent, confidence = chatbot.predict_intent(user_message)
        log_chat_interaction(
            user_message=user_message,
            bot_response=response,
            user_id=session_id,
            intent=intent,
            confidence=confidence,
            response_time=response_time/1000  # Convert to seconds
        )
        
        # Log complete chat conversation for analysis
        session_info = {
            'intent': intent,
            'confidence': confidence,
            'response_time_ms': response_time,
            'personality': personality_info,
            'timestamp': datetime.now().isoformat()
        }
        log_chat_conversation(
            session_id=session_id,
            user_message=user_message,
            bot_response=response,
            session_info=session_info
        )
        
        # Base response
        response_data = {
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
        }
        
        # Debug mode için ek bilgiler
        if debug_mode:
            # Intent ve confidence bilgileri
            sentiment = chatbot._analyze_sentiment(user_message)
            
            # Context analizi
            context_info = chatbot.get_context_info(session_id)
            
            # Model performans bilgileri
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            
            # AI Engine bilgileri
            ai_engine_info = {}
            if hasattr(chatbot, 'ai_engine'):
                ai_engine_info = {
                    'ai_capabilities': chatbot.ai_engine.get_ai_capabilities(),
                    'ai_stats': chatbot.ai_engine.get_performance_stats()
                }
            
            response_data.update({
                'debug_mode': True,
                'intent': intent,
                'confidence': round(confidence, 3),
                'sentiment': sentiment,
                'response_time_ms': round(response_time, 2),
                'context_analysis': {
                    'topic': context_info.get('current_topic', 'genel'),
                    'entities': context_info.get('entities_found', {}),
                    'logic': {
                        'logical_connections': context_info.get('logical_connections', [])[-3:],  # Son 3
                        'questions_asked': context_info.get('questions_asked', [])[-3:],  # Son 3
                        'assumptions': context_info.get('assumptions', [])[-3:]  # Son 3
                    }
                },
                'model_performance': {
                    'model_size': chatbot.model_size,
                    'response_time': round(response_time, 2),
                    'memory_usage': f"{memory_usage:.1f} MB",
                    'session_memory': len(context_info.get('messages', [])),
                    'context_switches': context_info.get('context_switches_count', 0)
                },
                'ai_engine_info': ai_engine_info
            })
        
        # Log performance
        log_performance("chat_response", response_time/1000, {
            "session_id": session_id,
            "message_length": len(user_message),
            "debug_mode": debug_mode,
            "intent": intent,
            "confidence": confidence
        })
        
        log_info(f"💬 Chat response completed", 
                session_id=session_id, 
                response_time_ms=round(response_time, 2),
                debug_mode=debug_mode,
                intent=intent,
                confidence=confidence)
        
        return jsonify(response_data)
        
    except Exception as e:
        log_error(e, context={
            "operation": "chat_request",
            "user_message": user_message,
            "session_id": session_id,
            "debug_mode": debug_mode
        }, user_id=session_id)
        
        return jsonify({
            'success': False,
            'response': 'Bir hata oluştu. Lütfen tekrar deneyin.',
            'error': str(e)
        }), 500

@app.route('/api/personality/info', methods=['GET'])
def get_personality_info():
    """Kişilik bilgileri endpoint"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            personality_info = chatbot.get_personality_info()
            log_info("👤 Kişilik bilgileri alındı")
            return jsonify(personality_info)
        else:
            log_error(Exception("Chatbot not available"), 
                     context={"operation": "personality_info"}, 
                     user_id="system")
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        log_error(e, context={"operation": "personality_info"}, user_id="system")
        return jsonify({'error': str(e)}), 500

@app.route('/api/personality/summary', methods=['GET'])
def get_personality_summary():
    """Kişilik özeti endpoint"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            summary = chatbot.personality_manager.get_personality_summary()
            log_info("📋 Kişilik özeti alındı")
            return jsonify(summary)
        else:
            log_error(Exception("Chatbot not available"), 
                     context={"operation": "personality_summary"}, 
                     user_id="system")
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        log_error(e, context={"operation": "personality_summary"}, user_id="system")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """İstatistikler endpoint"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            stats = chatbot.get_stats()
            log_info("📊 İstatistikler alındı")
            return jsonify(stats)
        else:
            log_error(Exception("Chatbot not available"), 
                     context={"operation": "get_stats"}, 
                     user_id="system")
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        log_error(e, context={"operation": "get_stats"}, user_id="system")
        return jsonify({'error': str(e)}), 500

@app.route('/api/context/info', methods=['GET'])
def get_context_info():
    """Bağlam bilgileri endpoint"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            session_id = request.args.get('session_id', 'default')
            context_info = chatbot.get_context_info(session_id)
            log_info("🧠 Bağlam bilgileri alındı", session_id=session_id)
            return jsonify(context_info)
        else:
            log_error(Exception("Chatbot not available"), 
                     context={"operation": "context_info"}, 
                     user_id="system")
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        log_error(e, context={"operation": "context_info"}, user_id="system")
        return jsonify({'error': str(e)}), 500

@app.route('/api/context/clear', methods=['POST'])
def clear_context():
    """Bağlam temizleme endpoint"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            data = request.get_json()
            session_id = data.get('session_id', 'default')
            chatbot.context_manager.clear_context(session_id)
            log_info("🗑️ Bağlam temizlendi", session_id=session_id)
            return jsonify({
                'success': True,
                'message': 'Bağlam temizlendi',
                'timestamp': chatbot.get_timestamp()
            })
        else:
            log_error(Exception("Chatbot not available"), 
                     context={"operation": "clear_context"}, 
                     user_id="system")
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        log_error(e, context={"operation": "clear_context"}, user_id="system")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Sağlık kontrolü endpoint"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Chatbot status
        chatbot_status = 'running' if chatbot else 'stopped'
        ai_engine_status = 'running' if (chatbot and hasattr(chatbot, 'ai_engine')) else 'not_available'
        
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'chatbot_status': chatbot_status,
            'ai_engine_status': ai_engine_status,
            'model_size': chatbot.model_size if chatbot else 'unknown',
            'system_metrics': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_free_gb': round(disk.free / (1024**3), 2)
            },
            'process_metrics': {
                'memory_usage_mb': round(process_memory, 1),
                'cpu_percent': process.cpu_percent(),
                'threads': process.num_threads()
            }
        }
        
        log_info("💚 Health check completed", status=status)
        return jsonify(status)
        
    except Exception as e:
        log_error(e, context={"operation": "health_check"}, user_id="system")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/api/debug/info', methods=['GET'])
def get_debug_info():
    """Debug bilgileri endpoint"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot:
            # AI Engine info
            ai_engine_info = {}
            if hasattr(chatbot, 'ai_engine'):
                ai_engine_info = {
                    'ai_capabilities': chatbot.ai_engine.get_ai_capabilities(),
                    'ai_stats': chatbot.ai_engine.get_performance_stats()
                }
            
            debug_info = {
                'chatbot_status': 'running',
                'model_size': chatbot.model_size,
                'model_loaded': chatbot.classifier is not None,
                'vectorizer_loaded': chatbot.vectorizer is not None,
                'personality_loaded': chatbot.personality_manager is not None,
                'context_loaded': chatbot.context_manager is not None,
                'ai_engine_loaded': hasattr(chatbot, 'ai_engine'),
                'ai_engine_info': ai_engine_info,
                'system_metrics': {
                    'memory_usage': f"{psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB",
                    'cpu_percent': psutil.cpu_percent(),
                    'active_sessions': len(chatbot.context_manager.contexts) if hasattr(chatbot, 'context_manager') else 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
            log_info("🔍 Debug bilgileri alındı", debug_info=debug_info)
            return jsonify(debug_info)
        else:
            log_error(Exception("Chatbot not available"), 
                     context={"operation": "debug_info"}, 
                     user_id="system")
            return jsonify({'error': 'Chatbot başlatılamadı'}), 500
            
    except Exception as e:
        log_error(e, context={"operation": "debug_info"}, user_id="system")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/summary', methods=['GET'])
def get_logs_summary():
    """Log özeti endpoint"""
    try:
        summary = logger.get_log_summary()
        log_info("📋 Log özeti alındı")
        return jsonify(summary)
    except Exception as e:
        log_error(e, context={"operation": "logs_summary"}, user_id="system")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/cleanup', methods=['POST'])
def cleanup_logs():
    """Clean up old log files"""
    try:
        # Clean up logs older than 30 days
        logger.cleanup_old_logs(days_to_keep=30)
        return {"status": "success", "message": "Log temizleme tamamlandı"}
    except Exception as e:
        log_error(e, context={"operation": "log_cleanup"})
        return {"status": "error", "message": f"Log temizleme hatası: {str(e)}"}

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Sistem durumu endpoint"""
    try:
        logger.log_system_status()
        log_info("📊 Sistem durumu loglandı")
        return jsonify({'success': True, 'message': 'System status logged'})
    except Exception as e:
        log_error(e, context={"operation": "system_status"}, user_id="system")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/frontend', methods=['POST'])
def receive_frontend_logs():
    """Receive logs from frontend"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Log frontend message with readable timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_info(f"Frontend Log - {timestamp}: {data.get('message', 'No message')}", 
                context=data)
        
        return jsonify({"status": "success"})
    except Exception as e:
        log_error(e, context={"operation": "frontend_log_reception"})
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/capabilities', methods=['GET'])
def get_ai_capabilities():
    """AI yetenekleri endpoint"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot and hasattr(chatbot, 'ai_engine'):
            capabilities = chatbot.ai_engine.get_ai_capabilities()
            log_info("🧠 AI yetenekleri alındı")
            return jsonify(capabilities)
        else:
            return jsonify({'error': 'AI Engine mevcut değil'}), 404
            
    except Exception as e:
        log_error(e, context={"operation": "ai_capabilities"}, user_id="system")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/performance', methods=['GET'])
def get_ai_performance():
    """AI performans endpoint"""
    try:
        if not chatbot:
            initialize_chatbot()
        
        if chatbot and hasattr(chatbot, 'ai_engine'):
            performance = chatbot.ai_engine.get_performance_stats()
            log_info("📈 AI performans bilgileri alındı")
            return jsonify(performance)
        else:
            return jsonify({'error': 'AI Engine mevcut değil'}), 404
            
    except Exception as e:
        log_error(e, context={"operation": "ai_performance"}, user_id="system")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/latest-errors', methods=['GET'])
def get_latest_errors():
    """Get the latest error logs and error details"""
    try:
        errors_dir = os.path.join("logs", "errors")
        latest_errors = {}
        
        if os.path.exists(errors_dir):
            # Get latest error log
            latest_error_log = os.path.join(errors_dir, "latest.txt")
            if os.path.exists(latest_error_log):
                try:
                    with open(latest_error_log, 'r', encoding='utf-8') as f:
                        latest_errors['latest_error_log'] = f.read()
                except Exception as e:
                    latest_errors['latest_error_log_error'] = str(e)
            
            # Get latest error details
            latest_error_details = os.path.join(errors_dir, "latest_error_details.txt")
            if os.path.exists(latest_error_details):
                try:
                    with open(latest_error_details, 'r', encoding='utf-8') as f:
                        latest_errors['latest_error_details'] = f.read()
                except Exception as e:
                    latest_errors['latest_error_details_error'] = str(e)
            
            # Get list of all error files
            error_files = []
            for file in os.listdir(errors_dir):
                if file.endswith('.txt') and not file.startswith('latest'):
                    file_path = os.path.join(errors_dir, file)
                    file_stat = os.stat(file_path)
                    error_files.append({
                        'filename': file,
                        'size_mb': round(file_stat.st_size / (1024 * 1024), 2),
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    })
            
            # Sort by modification time (newest first)
            error_files.sort(key=lambda x: x['modified'], reverse=True)
            latest_errors['all_error_files'] = error_files[:10]  # Show last 10
            
        return jsonify({
            'status': 'success',
            'errors_directory': errors_dir,
            'latest_errors': latest_errors
        })
        
    except Exception as e:
        log_error(e, context={"operation": "latest_errors_retrieval"})
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/session-summary', methods=['POST'])
def log_session_summary():
    """Log chat session summary when session ends"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        messages = data.get('messages', [])
        session_duration = data.get('session_duration')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Log session summary for analysis
        log_chat_session_summary(
            session_id=session_id,
            messages=messages,
            session_duration=session_duration
        )
        
        return jsonify({'success': True, 'message': 'Session summary logged'})
        
    except Exception as e:
        log_error(e, context={"operation": "session_summary_logging"}, user_id="system")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/sessions', methods=['GET'])
def get_chat_sessions():
    """Tüm chat session'ları döner"""
    try:
        sessions = get_all_sessions()
        return jsonify({'success': True, 'sessions': sessions})
    except Exception as e:
        log_error(e, context={"operation": "get_chat_sessions"}, user_id="system")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    """Belirli bir session'daki mesajları döner"""
    try:
        session_data = get_session_data(session_id)
        if session_data:
            return jsonify({'success': True, 'messages': session_data['messages']})
        else:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
    except Exception as e:
        log_error(e, context={"operation": "get_session_messages"}, user_id="system")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/sessions', methods=['POST'])
def create_chat_session():
    """Yeni chat session oluşturur"""
    try:
        data = request.get_json()
        title = data.get('title', 'Yeni Sohbet')
        
        session_id = create_session(title)
        
        return jsonify({'success': True, 'session_id': session_id})
    except Exception as e:
        log_error(e, context={"operation": "create_chat_session"}, user_id="system")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/sessions/<session_id>/title', methods=['PUT'])
def update_session_title(session_id):
    """Session başlığını günceller"""
    try:
        data = request.get_json()
        title = data.get('title')
        
        if not title:
            return jsonify({'success': False, 'error': 'Title required'}), 400
        
        success = update_title(session_id, title)
        if success:
            return jsonify({'success': True, 'message': 'Title updated'})
        else:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
            
    except Exception as e:
        log_error(e, context={"operation": "update_session_title"}, user_id="system")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/sessions/<session_id>', methods=['DELETE'])
def delete_chat_session(session_id):
    """Chat session'ı siler"""
    try:
        success = delete_session(session_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Session deleted'})
        else:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
            
    except Exception as e:
        log_error(e, context={"operation": "delete_chat_session"}, user_id="system")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/sessions/<session_id>/full', methods=['GET'])
def get_full_session(session_id):
    """Session ve tüm mesajlarını birlikte döner"""
    try:
        session_data = get_session_data(session_id)
        if session_data:
            return jsonify({'success': True, 'data': session_data})
        else:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
    except Exception as e:
        log_error(e, context={"operation": "get_full_session"}, user_id="system")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/stats', methods=['GET'])
def get_chat_stats():
    """Chat istatistiklerini döner"""
    try:
        stats = get_session_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        log_error(e, context={"operation": "get_chat_stats"}, user_id="system")
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    """404 hatası"""
    log_error(error, context={"operation": "404_error", "path": request.path}, user_id="system")
    return jsonify({'error': 'Sayfa bulunamadı', 'status': 404}), 404

@app.errorhandler(500)
def internal_error(error):
    """500 hatası"""
    log_error(error, context={"operation": "500_error", "path": request.path}, user_id="system")
    return jsonify({'error': 'Sunucu hatası', 'status': 500}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Genel exception handler"""
    log_error(e, context={"operation": "unhandled_exception", "path": request.path}, user_id="system")
    return jsonify({'error': 'Beklenmeyen hata oluştu', 'status': 500}), 500

if __name__ == '__main__':
    # Gerekli dizinleri oluştur
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("config").mkdir(exist_ok=True)
    
    # Chatbot'u başlat
    initialize_chatbot()
    
    # Log system status
    logger.log_system_status()
    
    log_info("🚀 Epsilon AI başlatılıyor...")
    log_info("📱 Web arayüzü: http://localhost:5000")
    log_info("🔧 API endpoint: http://localhost:5000/chat")
    log_info("🔍 Debug endpoint: http://localhost:5000/api/debug/info")
    log_info("💚 Health check: http://localhost:5000/health")
    log_info("📋 Log özeti: http://localhost:5000/api/logs/summary")
    log_info("🧠 AI yetenekleri: http://localhost:5000/api/ai/capabilities")
    
    app.run(debug=True, host='0.0.0.0', port=5000)