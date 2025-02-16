from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

class TechnicalAnalyzer:
    """
    기술적 분석을 수행하는 클래스
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.indicators_config = config['indicators']

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        여러 기술적 지표를 계산
        
        Args:
            df: OHLCV 데이터가 있는 DataFrame
            
        Returns:
            Dict[str, Any]: 계산된 기술적 지표들
        """
        if df is None or df.empty:
            return {}

        # RSI
        rsi = RSIIndicator(
            close=df['close'],
            window=self.indicators_config['rsi']['period']
        )
        
        # MACD
        macd = MACD(
            close=df['close'],
            window_fast=self.indicators_config['macd']['fast_period'],
            window_slow=self.indicators_config['macd']['slow_period'],
            window_sign=self.indicators_config['macd']['signal_period']
        )
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        
        # Stochastic
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )

        return {
            'rsi': {
                'value': rsi.rsi(),
                'overbought': self.indicators_config['rsi']['overbought'],
                'oversold': self.indicators_config['rsi']['oversold']
            },
            'macd': {
                'macd': macd.macd(),
                'signal': macd.macd_signal(),
                'histogram': macd.macd_diff()
            },
            'bollinger': {
                'upper': bb.bollinger_hband(),
                'middle': bb.bollinger_mavg(),
                'lower': bb.bollinger_lband()
            },
            'stochastic': {
                'k': stoch.stoch(),
                'd': stoch.stoch_signal()
            },
            'vwap': vwap.volume_weighted_average_price()
        }

    def analyze_signals(self, indicators: Dict[str, Any]) -> Dict[str, str]:
        """
        기술적 지표를 기반으로 매매 신호 분석
        
        Args:
            indicators: 계산된 기술적 지표들
            
        Returns:
            Dict[str, str]: 각 지표별 매매 신호 ('buy', 'sell', 'neutral')
        """
        signals = {}
        
        # RSI 분석
        if 'rsi' in indicators:
            rsi_value = indicators['rsi']['value'].iloc[-1]
            if rsi_value < indicators['rsi']['oversold']:
                signals['rsi'] = 'buy'
            elif rsi_value > indicators['rsi']['overbought']:
                signals['rsi'] = 'sell'
            else:
                signals['rsi'] = 'neutral'
        
        # MACD 분석
        if 'macd' in indicators:
            macd_hist = indicators['macd']['histogram'].iloc[-1]
            macd_prev = indicators['macd']['histogram'].iloc[-2]
            
            if macd_hist > 0 and macd_prev < 0:
                signals['macd'] = 'buy'
            elif macd_hist < 0 and macd_prev > 0:
                signals['macd'] = 'sell'
            else:
                signals['macd'] = 'neutral'
        
        # Bollinger Bands 분석
        if 'bollinger' in indicators:
            close = indicators['bollinger']['middle'].index[-1]
            upper = indicators['bollinger']['upper'].iloc[-1]
            lower = indicators['bollinger']['lower'].iloc[-1]
            
            if close < lower:
                signals['bollinger'] = 'buy'
            elif close > upper:
                signals['bollinger'] = 'sell'
            else:
                signals['bollinger'] = 'neutral'
                
        return signals

    def get_combined_signal(self, signals: Dict[str, str]) -> str:
        """
        여러 지표의 신호를 종합하여 최종 매매 신호 도출
        
        Args:
            signals: 각 지표별 매매 신호
            
        Returns:
            str: 최종 매매 신호 ('buy', 'sell', 'neutral')
        """
        buy_count = sum(1 for signal in signals.values() if signal == 'buy')
        sell_count = sum(1 for signal in signals.values() if signal == 'sell')
        
        # 60% 이상의 지표가 같은 신호를 보낼 때 매매 신호 생성
        total_indicators = len(signals)
        if buy_count / total_indicators >= 0.6:
            return 'buy'
        elif sell_count / total_indicators >= 0.6:
            return 'sell'
        return 'neutral' 