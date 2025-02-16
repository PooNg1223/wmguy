from typing import Dict, Any, Optional
import logging
from pybit.unified_trading import HTTP

class BybitClient:
    """바이비트 API 클라이언트"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger('bybit')
        
        # API 클라이언트 초기화
        self.client = HTTP(
            testnet=config.get('testnet', True),
            api_key=config.get('api_key'),
            api_secret=config.get('api_secret')
        )
        
        self.logger.info("Bybit client initialized")
    
    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            # API 키가 없어도 되는 공개 엔드포인트 사용
            response = self.client.get_tickers(
                category="spot",
                symbol="BTCUSDT"
            )
            self.logger.debug(f"Test connection response: {response}")
            return isinstance(response, dict) and 'result' in response
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """포지션 조회"""
        try:
            response = self.client.get_positions(
                category="linear",
                symbol=symbol
            )
            
            if isinstance(response, dict) and 'result' in response:
                result = response['result']
                if isinstance(result, dict) and 'list' in result:
                    positions = result['list']
                    if positions and len(positions) > 0:
                        return positions[0]
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get position: {e}")
            return None
    
    def get_kline(self, symbol: str, interval: str = "15", limit: int = 100) -> Optional[Dict[str, Any]]:
        """K라인 데이터 조회"""
        try:
            # API 요청 로깅 추가
            self.logger.debug(f"Requesting kline data: symbol={symbol}, interval={interval}, limit={limit}")
            
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # 응답 상세 로깅
            self.logger.debug(f"Raw kline response: {response}")
            if isinstance(response, dict):
                self.logger.debug(f"Response keys: {response.keys()}")
                if 'result' in response:
                    self.logger.debug(f"Result keys: {response['result'].keys()}")
                    if 'list' in response['result']:
                        self.logger.debug(f"Got {len(response['result']['list'])} data points")
            
            if isinstance(response, dict) and 'result' in response:
                return response
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get kline data: {e}")
            return None
    
    def place_order(self, category: str, symbol: str, side: str, orderType: str, qty: str, timeInForce: str) -> Optional[Dict[str, Any]]:
        """주문 실행"""
        try:
            response = self.client.place_order(
                category=category,
                symbol=symbol,
                side=side,
                orderType=orderType,
                qty=qty,
                timeInForce=timeInForce
            )
            
            if isinstance(response, dict) and 'result' in response:
                return response
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return None 