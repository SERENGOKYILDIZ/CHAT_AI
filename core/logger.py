#!/usr/bin/env python3
"""
Simplified Logging System for AI Chatbot
Creates a single log file with timestamp-based naming
"""

import logging
import os
import sys
import json
import traceback
import atexit
import signal
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import psutil
import platform

class TimestampFileHandler(logging.FileHandler):
    """File handler that creates a single log file with timestamp naming"""
    
    def __init__(self, base_log_dir: str = "logs", max_file_size_mb: int = 50):
        self.base_log_dir = base_log_dir
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        
        # Create logs directory
        os.makedirs(base_log_dir, exist_ok=True)
        
        # Generate filename with current timestamp (YYYY_MM_DD-HH-MM-SS.txt)
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime("%Y_%m_%d-%H-%M-%S")
        self.filename = os.path.join(base_log_dir, f"{timestamp}.txt")
        
        # Check if file exists and is too large
        if os.path.exists(self.filename) and os.path.getsize(self.filename) > self.max_file_size:
            # Create numbered backup
            backup_num = 1
            while os.path.exists(f"{self.filename}.{backup_num}"):
                backup_num += 1
            os.rename(self.filename, f"{self.filename}.{backup_num}")
        
        super().__init__(self.filename, mode='a', encoding='utf-8')
        
        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.setFormatter(formatter)

class ChatbotLogger:
    """Main logger class with single file logging"""
    
    def __init__(self, name: str = "ChatbotLogger", base_log_dir: str = "logs"):
        self.name = name
        self.base_log_dir = base_log_dir
        self.start_time = datetime.now()
        
        # Create base log directory
        os.makedirs(base_log_dir, exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Register shutdown handlers
        atexit.register(self._shutdown_logging)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"🚀 Logger başlatıldı: {name}")
        self.logger.info(f"📁 Log dosyası: {os.path.abspath(self.filename)}")
    
    def _setup_handlers(self):
        """Setup single file log handler"""
        try:
            # Single timestamp-based file handler
            self.file_handler = TimestampFileHandler(self.base_log_dir)
            self.filename = self.file_handler.filename
            self.logger.addHandler(self.file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            self.logger.info(f"✅ Logger handler başarıyla kuruldu")
            self.logger.info(f"📂 Tek log dosyası: {os.path.basename(self.filename)}")
            
        except Exception as e:
            print(f"❌ Logger handler kurulumunda hata: {e}")
            # Fallback to basic logging
            basic_handler = logging.StreamHandler()
            basic_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(basic_handler)
    
    def _shutdown_logging(self):
        """Log shutdown information"""
        try:
            uptime = datetime.now() - self.start_time
            self.logger.info(f"🔄 Uygulama kapatılıyor...")
            self.logger.info(f"⏱️ Toplam çalışma süresi: {uptime}")
            self.logger.info(f"📊 Son sistem durumu kaydediliyor...")
            
            # Final system status log
            self.log_system_status()
            
            # Create shutdown summary
            self._create_shutdown_summary()
            
            self.logger.info(f"✅ Logger kapatıldı")
            
        except Exception as e:
            print(f"❌ Shutdown logging hatası: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"📡 Sinyal alındı: {signum}")
        sys.exit(0)
    
    def _create_shutdown_summary(self):
        """Create a summary of the session"""
        try:
            summary_file = os.path.join(self.base_log_dir, "session_summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=== OTURUM ÖZETİ ===\n")
                f.write(f"Başlangıç: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Çalışma süresi: {datetime.now() - self.start_time}\n")
                f.write(f"Log dosyası: {os.path.basename(self.filename)}\n")
                f.write(f"Log boyutu: {os.path.getsize(self.filename) / (1024*1024):.2f} MB\n")
                
        except Exception as e:
            print(f"❌ Session summary oluşturulamadı: {e}")
    
    def log_performance(self, operation: str, duration: float, details: Optional[Dict] = None):
        """Log performance metrics"""
        message = f"⏱️ Performance: {operation} - {duration:.3f}s"
        if details:
            message += f" - Details: {details}"
        
        self.logger.info(message)
    
    def log_error(self, error: Exception, context: Optional[Dict] = None, user_id: Optional[str] = None):
        """Log errors with context"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        message = f"❌ Error: {error_info['error_type']} - {error_info['error_message']}"
        if context:
            message += f" - Context: {context}"
        if user_id:
            message += f" - User: {user_id}"
        
        self.logger.error(message, exc_info=True)
        
        # Save detailed error info
        self._save_error_details(error, context)
    
    def log_user_action(self, action: str, user_id: str, details: Optional[Dict] = None):
        """Log user actions"""
        message = f"👤 User Action: {action} - User: {user_id}"
        if details:
            message += f" - Details: {details}"
        
        self.logger.info(message)
    
    def log_security_event(self, event_type: str, details: Dict, severity: str = "INFO"):
        """Log security-related events"""
        message = f"🔒 Security: {event_type} - {details}"
        
        if severity == "ERROR":
            self.logger.error(message)
        elif severity == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def log_api_call(self, endpoint: str, method: str, user_id: Optional[str] = None, 
                     duration: Optional[float] = None, status_code: Optional[int] = None):
        """Log API calls"""
        message = f"🌐 API: {method} {endpoint}"
        if user_id:
            message += f" - User: {user_id}"
        if duration:
            message += f" - Duration: {duration:.3f}s"
        if status_code:
            message += f" - Status: {status_code}"
        
        self.logger.info(message)
    
    def log_chat_interaction(self, user_message: str, bot_response: str, user_id: str, 
                           intent: str, confidence: float, response_time: float):
        """Log chat interactions"""
        # Truncate long messages for logging
        user_msg_truncated = user_message[:100] + "..." if len(user_message) > 100 else user_message
        bot_resp_truncated = bot_response[:100] + "..." if len(bot_response) > 100 else bot_response
        
        message = (f"💬 Chat: User({user_id}) -> Bot - "
                  f"Intent: {intent}({confidence:.2f}) - "
                  f"Response Time: {response_time:.3f}s")
        
        details = {
            'user_message': user_msg_truncated,
            'bot_response': bot_resp_truncated,
            'intent': intent,
            'confidence': confidence,
            'response_time': response_time
        }
        
        self.logger.info(message, extra={'extra_fields': details})
    
    def log_ai_activity(self, activity: str, details: Optional[Dict] = None):
        """Log AI-specific activities"""
        message = f"🤖 AI: {activity}"
        if details:
            message += f" - Details: {details}"
        
        self.logger.info(message)
    
    def log_system_status(self):
        """Log system status information"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used / (1024**3)  # GB
            memory_total = memory.total / (1024**3)  # GB
            
            # Disk usage
            try:
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                disk_free = disk.free / (1024**3)  # GB
            except:
                disk_percent = 0
                disk_free = 0
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            
            status_info = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': round(memory_used, 2),
                'memory_total_gb': round(memory_total, 2),
                'disk_percent': disk_percent,
                'disk_free_gb': round(disk_free, 2),
                'process_memory_mb': round(process_memory, 2)
            }
            
            message = (f"📊 System Status: CPU: {cpu_percent}% | "
                      f"Memory: {memory_percent}% ({memory_used:.1f}GB/{memory_total:.1f}GB) | "
                      f"Disk: {disk_percent}% | Process: {process_memory:.1f}MB")
            
            self.logger.info(message, extra={'extra_fields': status_info})
            
        except Exception as e:
            self.logger.error(f"❌ System status logging failed: {e}")
    
    def _save_error_details(self, error: Exception, context: Dict[str, Any] = None):
        """Save detailed error information"""
        try:
            timestamp = datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
            error_filename = f"error_details_{timestamp}.txt"
            error_filepath = os.path.join(self.base_log_dir, error_filename)
            
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'context': context or {},
                'system_info': {
                    'platform': platform.platform(),
                    'python_version': platform.python_version(),
                    'memory_usage': psutil.virtual_memory()._asdict() if psutil else 'N/A'
                }
            }
            
            with open(error_filepath, 'w', encoding='utf-8') as f:
                f.write("=== DETAYLI HATA RAPORU ===\n")
                f.write(f"Zaman: {error_info['timestamp']}\n")
                f.write(f"Hata Türü: {error_info['error_type']}\n")
                f.write(f"Hata Mesajı: {error_info['error_message']}\n")
                f.write(f"Bağlam: {json.dumps(error_info['context'], indent=2, ensure_ascii=False)}\n")
                f.write(f"Sistem Bilgisi: {json.dumps(error_info['system_info'], indent=2, ensure_ascii=False)}\n")
                f.write("\n=== TRACEBACK ===\n")
                f.write(error_info['traceback'])
            
            # Create latest error details file
            latest_error_path = os.path.join(self.base_log_dir, "latest_error_details.txt")
            try:
                with open(error_filepath, 'r', encoding='utf-8') as src:
                    with open(latest_error_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
            except Exception:
                pass
            
            self.logger.error(f"Detaylı hata kaydedildi: {error_filename}")
            
        except Exception as e:
            print(f"❌ Hata detayları kaydedilemedi: {e}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of log files"""
        try:
            summary = {
                'total_files': 0,
                'total_size_mb': 0.0,
                'files': [],
                'session_info': {
                    'start_time': self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    'uptime': str(datetime.now() - self.start_time)
                }
            }
            
            # Get all log files
            files = [f for f in os.listdir(self.base_log_dir) if f.endswith('.txt')]
            summary['total_files'] = len(files)
            
            for filename in files:
                file_path = os.path.join(self.base_log_dir, filename)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                summary['total_size_mb'] += file_size
                
                file_info = {
                    'filename': filename,
                    'size_mb': round(file_size, 2),
                    'path': file_path
                }
                summary['files'].append(file_info)
            
            summary['total_size_mb'] = round(summary['total_size_mb'], 2)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Log özeti alınamadı: {e}")
            return {'error': str(e)}
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            files = [f for f in os.listdir(self.base_log_dir) if f.endswith('.txt')]
            
            for filename in files:
                try:
                    # Extract date from filename (e.g., "2025_08_11-16-52-11.txt")
                    if filename.endswith('.txt'):
                        # Remove .txt extension and split by dash
                        name_without_ext = filename.replace('.txt', '')
                        parts = name_without_ext.split('-')
                        
                        if len(parts) >= 3:  # Should have: YYYY_MM_DD, HH, MM, SS
                            date_part = parts[0]  # YYYY_MM_DD
                            try:
                                file_date = datetime.strptime(date_part, "%Y_%m_%d")
                                
                                if file_date < cutoff_date:
                                    file_path = os.path.join(self.base_log_dir, filename)
                                    os.remove(file_path)
                                    deleted_count += 1
                                    self.logger.info(f"Eski log dosyası silindi: {filename}")
                            except ValueError:
                                continue
                
                except Exception as e:
                    self.logger.warning(f"Log dosyası işlenirken hata: {filename}, {e}")
            
            if deleted_count > 0:
                self.logger.info(f"✅ {deleted_count} eski log dosyası temizlendi")
            else:
                self.logger.info("✅ Eski log dosyası bulunamadı")
                
        except Exception as e:
            self.logger.error(f"❌ Log temizleme hatası: {e}")

# Global logger instance
logger = ChatbotLogger("AI_Chatbot")

# Convenience functions
def log_info(message: str, **kwargs):
    """Log info message"""
    logger.logger.info(message, extra={'extra_fields': kwargs})

def log_error(error: Exception, context: Optional[Dict] = None, user_id: Optional[str] = None):
    """Log error"""
    logger.log_error(error, context, user_id)

def log_performance(operation: str, duration: float, details: Optional[Dict] = None):
    """Log performance metric"""
    logger.log_performance(operation, duration, details)

def log_user_action(action: str, user_id: str, details: Optional[Dict] = None):
    """Log user action"""
    logger.log_user_action(action, user_id, details)

def log_api_call(endpoint: str, method: str, user_id: Optional[str] = None, 
                duration: Optional[float] = None, status_code: Optional[int] = None):
    """Log API call"""
    logger.log_api_call(endpoint, method, user_id, duration, status_code)

def log_chat_interaction(user_message: str, bot_response: str, user_id: str, 
                        intent: str, confidence: float, response_time: float):
    """Log chat interaction"""
    logger.log_chat_interaction(user_message, bot_response, user_id, intent, confidence, response_time)

def log_ai_activity(activity: str, details: Optional[Dict] = None):
    """Log AI activity"""
    logger.log_ai_activity(activity, details)
