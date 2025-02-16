from typing import Dict, Any, List
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .bybit_client import BybitClient
from ta.momentum import RSIIndicator

class DataCollector:
    """시장 데이터 수집기"""
    
    def __init__(self, bybit_client: BybitClient):
        self.bybit = bybit_client
        self.logger = logging.getLogger('collector')
        self.logger.setLevel(logging.DEBUG)  # 로깅 레벨 변경
        
    def get_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """과거 데이터 수집"""
        try:
            self.logger.debug(f"Fetching data for {symbol}, days={days}")
            
            klines = self.bybit.get_kline(
                symbol=symbol,
                interval="15",
                limit=days * 24 * 4  # 15분봉 기준
            )
            
            if not klines or 'result' not in klines:
                self.logger.error(f"No data received for {symbol}")
                return pd.DataFrame()
            
            data = []
            for k in klines['result']['list']:
                data.append({
                    'timestamp': pd.to_datetime(int(k[0]), unit='ms'),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            # 데이터프레임 상태 로깅
            self.logger.debug(f"DataFrame shape before preprocessing: {df.shape}")
            self.logger.debug(f"DataFrame head:\n{df.head()}")
            self.logger.debug(f"DataFrame info:\n{df.info()}")
            
            # 숫자형 데이터 변환
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    self.logger.warning(f"Column {col} has {null_count} null values")
            
            # 결측치 제거
            df = df.dropna()
            
            # 전처리 후 상태 로깅
            self.logger.debug(f"DataFrame shape after preprocessing: {df.shape}")
            if not df.empty:
                self.logger.debug(f"Processed DataFrame head:\n{df.head()}")
            
            if len(df) == 0:
                self.logger.error("No valid data after preprocessing")
                return pd.DataFrame()
            
            # 기술적 지표 추가
            df = self._add_technical_indicators(df)
            
            # 최종 상태 로깅
            self.logger.debug(f"Final DataFrame shape: {df.shape}")
            self.logger.debug(f"Final columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            self.logger.exception("Detailed error:")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        try:
            # 기본 이동평균
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA60'] = df['close'].rolling(window=60).mean()
            
            # 볼린저 밴드
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_std'] = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
            df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
            
            # RSI (ta 라이브러리 사용)
            rsi_indicator = RSIIndicator(close=df['close'], window=14)
            df['RSI'] = rsi_indicator.rsi()
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
            
            # 거래량 분석
            df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA20']
            
            # OBV (On-Balance Volume)
            df['Price_Change'] = df['close'].diff()
            df['OBV'] = (df['volume'] * df['Price_Change'].apply(np.sign)).cumsum()
            
            # 변동성 지표
            df['ATR'] = self._calculate_atr(df)
            df['Volatility'] = df['ATR'] / df['close']
            
            # 추세 강도
            df['Trend_Strength'] = abs(df['MA5'] - df['MA20']) / df['MA20']
            
            # 가격 모멘텀
            df['ROC'] = df['close'].pct_change(periods=10)
            df['Momentum'] = df['close'] / df['close'].shift(10)
            
            # 거래량 급증 감지
            df['Volume_Surge'] = df['Volume_Ratio'] > 2.0
            
            # 가격 패턴
            df['Higher_High'] = df['high'] > df['high'].shift(1)
            df['Lower_Low'] = df['low'] < df['low'].shift(1)
            df['Pattern'] = self._calculate_pattern(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to add indicators: {e}")
            return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR(Average True Range) 계산"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_pattern(self, df: pd.DataFrame) -> pd.Series:
        """가격 패턴 식별"""
        pattern = pd.Series(0, index=df.index)
        
        # 상승 추세
        uptrend = (df['MA5'] > df['MA20']) & (df['MA20'] > df['MA60'])
        # 하락 추세
        downtrend = (df['MA5'] < df['MA20']) & (df['MA20'] < df['MA60'])
        
        # 과매수/과매도
        overbought = df['RSI'] > 70
        oversold = df['RSI'] < 30
        
        # 거래량 급증
        volume_surge = df['Volume_Surge']
        
        # 패턴 점수 계산
        pattern = pattern.mask(uptrend, 1)
        pattern = pattern.mask(downtrend, -1)
        pattern = pattern.mask(uptrend & volume_surge, 2)
        pattern = pattern.mask(downtrend & volume_surge, -2)
        pattern = pattern.mask(oversold & volume_surge, 3)
        pattern = pattern.mask(overbought & volume_surge, -3)
        
        return pattern

# 클래스 export
__all__ = ['DataCollector'] 