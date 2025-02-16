from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from .bybit_client import BybitClient
from .trading_strategy import TradingStrategy

class TradingModule:
    """트레이딩 모듈 - 트레이딩 전략 및 포지션 관리"""
    
    def __init__(self, config: Dict[str, Any], bybit_client: BybitClient):
        """
        트레이딩 모듈 초기화
        
        Args:
            config: 트레이딩 설정
            bybit_client: 바이비트 API 클라이언트
        """
        self.logger = logging.getLogger('trading')
        self.config = config
        
        # 바이비트 클라이언트 설정
        self.bybit = bybit_client
        
        # 상태 저장소 초기화
        self.positions = {}
        self.trades_history = []
        self.current_prices = {}
        
        # 트레이딩 전략 초기화
        self.strategy = TradingStrategy()
        
        self.logger.info("Trading module initialized")
