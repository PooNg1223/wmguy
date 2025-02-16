from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .bybit_client import BybitClient
from ta.momentum import RSIIndicator

class DataCollector:
    """시장 데이터 수집 및 전처리"""
    
    def __init__(self, client: BybitClient):
        self.logger = logging.getLogger('collector')
        self.client = client
        
    def get_historical_data(self, symbol: str, interval: str = '15', days: int = 30) -> pd.DataFrame:
        """과거 데이터 수집"""
        try:
            # 바이비트에서 K라인 데이터 조회
            kline = self.client.get_kline(
                symbol=symbol,
                interval=interval,
                limit=days * 24 * 4  # 15분봉 기준
            )
            
            if not kline or 'result' not in kline or not kline['result']['list']:
                self.logger.error(f"No data received for {symbol}")
                return pd.DataFrame()
            
            # 데이터프레임 생성
            df = pd.DataFrame(kline['result']['list'], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # timestamp를 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
            
            # 숫자형 데이터 변환
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 기술적 지표 추가
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        try:
            # 이동평균
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA60'] = df['close'].rolling(window=60).mean()
            
            # 볼린저 밴드
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
            
            # RSI (ta 라이브러리 사용)
            rsi_indicator = RSIIndicator(close=df['close'], window=14)
            df['RSI'] = rsi_indicator.rsi()
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to add indicators: {e}")
            return df

# 클래스 export
__all__ = ['DataCollector'] 