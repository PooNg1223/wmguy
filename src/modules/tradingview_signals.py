from flask import Flask, request
from typing import Dict, Any, Callable
import threading
import logging

class TradingViewSignals:
    """트레이딩뷰 시그널 처리"""
    
    def __init__(self, signal_handler: Callable):
        self.logger = logging.getLogger('tradingview')
        self.app = Flask(__name__)
        self.signal_handler = signal_handler
        
        @self.app.route('/webhook', methods=['POST'])
        def webhook():
            data = request.json
            self.signal_handler(data)
            return 'OK'
    
    def start(self, port: int = 5000):
        """웹훅 서버 시작"""
        threading.Thread(
            target=self.app.run,
            kwargs={'port': port},
            daemon=True
        ).start() 