#!/usr/bin/env python3
"""
Epsilon AI Web Design Manager
Modern and user-friendly interface
"""

from flask import render_template
import logging

logger = logging.getLogger(__name__)


class WebDesignManager:
    """Web tasarım yöneticisi"""
    
    def __init__(self):
        self.title = "Epsilon AI - Advanced AI Assistant"
        self.description = "Advanced AI assistant with integrated personality system"
        self.version = "2.0"
    
    def render_main_page(self):
        """Ana sayfayı render et"""
        try:
            return render_template('index.html', 
                                title=self.title,
                                description=self.description,
                                version=self.version)
        except Exception as e:
            logger.error(f"❌ Ana sayfa render hatası: {e}")
            return self._get_fallback_html()
    
    def _get_fallback_html(self):
        """Fallback HTML"""
        return f"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .chat-container {{ background: rgba(255,255,255,0.95); border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }}
        .message {{ margin: 10px; padding: 15px; border-radius: 15px; max-width: 80%; }}
        .user-message {{ background: #007bff; color: white; margin-left: auto; }}
        .bot-message {{ background: #f8f9fa; color: #333; }}
        .personality-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
    </style>
</head>
<body>
    <div class="container-fluid py-4">
            <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="chat-container p-4">
                    <div class="text-center mb-4">
                        <h1><i class="fas fa-rocket text-primary"></i> Epsilon AI</h1>
                        <p class="text-muted">Advanced AI Assistant v{self.version}</p>
                    </div>

                    <div id="chatMessages" class="mb-3" style="height: 400px; overflow-y: auto;">
                                <div class="message bot-message">
                            <i class="fas fa-rocket"></i> Merhaba! Ben Epsilon AI. Size nasıl yardımcı olabilirim? 🚀
                        </div>
                    </div>

                    <div class="input-group">
                        <input type="text" id="messageInput" class="form-control" placeholder="Mesajınızı yazın...">
                        <button class="btn btn-primary" onclick="sendMessage()">
                            <i class="fas fa-paper-plane"></i> Gönder
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function sendMessage() {{
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            fetch('/chat', {{
                        method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ message: message }})
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    addMessage(data.response, 'bot');
                }} else {{
                    addMessage('Üzgünüm, bir hata oluştu.', 'bot');
                }}
            }})
            .catch(error => {{
                addMessage('Bağlantı hatası oluştu.', 'bot');
            }});
        }}
        
        function addMessage(text, sender) {{
            const messages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${{sender}}-message`;
            messageDiv.innerHTML = `${{sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>'}} ${{text}}`;
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }}
        
        document.getElementById('messageInput').addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                sendMessage();
            }}
        }});
    </script>
</body>
</html>
        """