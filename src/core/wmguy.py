import logging
from typing import Dict, Any, Optional
from datetime import datetime
from src.modules.trading_module import TradingModule
from src.modules.data_collector import DataCollector
from src.modules.market_learner import MarketLearner
from src.utils.config_manager import ConfigManager
from src.modules.bybit_client import BybitClient

class WMGuy:
    """
    개인 AI 어시스턴트 - 트레이딩 자동화 중심
    """
    def __init__(self):
        self.logger = logging.getLogger('wmguy')
        self.logger.info("Initializing Trading Assistant")
        
        try:
            # 1. 설정 관리자 초기화
            self._config_manager = ConfigManager()
            
            # 2. 트레이딩 설정 가져오기
            trading_config = {
                'exchange': {
                    'name': 'bybit',
                    'api_key': self._config_manager.get('trading.api_key'),
                    'api_secret': self._config_manager.get('trading.api_secret'),
                    'testnet': self._config_manager.get('trading.testnet', True)
                },
                'risk': {
                    'max_position_size': self._config_manager.get('trading.risk.max_position_size', 1000),
                    'risk_level': self._config_manager.get('trading.risk.risk_level', 'moderate')
                }
            }
            
            # 3. 바이비트 클라이언트 초기화
            self.bybit = BybitClient(trading_config['exchange'])
            if not self.bybit.test_connection():
                raise ConnectionError("Failed to connect to Bybit API")
            self.logger.info("Bybit client initialized and tested")
            
            # 4. 데이터 수집기 초기화
            self.collector = DataCollector(self.bybit)
            self.logger.info("Data collector initialized")
            
            # 5. 트레이딩 모듈 초기화
            self.trading = TradingModule(trading_config, self.bybit)
            self.logger.info("Trading module initialized")
            
            # 6. 학습 모듈 초기화
            self.learner = MarketLearner(self.collector)
            self.logger.info("Market learner initialized")
            
            self.logger.info("Trading Assistant initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise
    
    @property
    def config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        return self._config_manager.get_all()
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """설정 업데이트"""
        try:
            self._config_manager.update(new_config)
            return True
        except Exception as e:
            self.logger.error(f"Config update failed: {e}")
            return False
    
    def start_trading(self, symbols: list[str]):
        """자동 트레이딩 시작"""
        try:
            # 실시간 가격 스트림 대신 현재 가격만 확인
            for symbol in symbols:
                price = self.trading.get_current_price(symbol)
                if price:
                    self.logger.info(f"Current price for {symbol}: {price}")
            
            self.logger.info(f"Trading initialized for symbols: {symbols}")
            
        except Exception as e:
            self.logger.error(f"Failed to start trading: {e}")
            raise
    
    def get_trading_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """현재 트레이딩 상태 조회"""
        if symbol:
            return {
                'positions': self.trading.get_position_status(symbol),
                'daily_summary': self.trading.get_daily_summary()
            }
        else:
            # 모든 거래 내역에서 유효한 심볼 목록만 추출
            symbols = {
                trade['trade_info']['symbol'] 
                for trade in self.trading.trades_history
                if trade.get('trade_info', {}).get('symbol')  # None이 아닌 경우만
            }
            
            positions = {}
            for symbol in symbols:
                if isinstance(symbol, str):  # 문자열인 경우만 처리
                    positions[symbol] = self.trading.get_position_status(symbol)
            
            return {
                'positions': positions,
                'daily_summary': self.trading.get_daily_summary()
            }
    
    def learn_market(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """시장 학습 및 분석"""
        try:
            # 학습 실행
            self.logger.info(f"Starting market learning for {symbol}")
            result = self.learner.learn(symbol, days=days)
            
            if not result:
                raise ValueError("Learning failed")
            
            # 현재 시장 분석
            analysis = self.learner.analyze(symbol)
            
            # 성과 기록 가져오기
            performance = self.learner.performance_history[-1]['performance']
            
            report = {
                'symbol': symbol,
                'signal': analysis['signal'],
                'strength': analysis['strength'],
                'confidence': analysis['confidence'],
                'performance': performance
            }
            
            self.logger.info(f"Analysis completed for {symbol}: {report}")
            return report
            
        except Exception as e:
            self.logger.error(f"Market learning failed: {e}")
            return {
                'symbol': symbol,
                'error': str(e)
            } 