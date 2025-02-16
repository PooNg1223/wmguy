from typing import Dict, Any, List
import pandas as pd
import numpy as np

class MarketSentimentAnalyzer:
    """
    시장 심리를 분석하는 클래스
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def analyze_volume_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        거래량 트렌드 분석
        """
        if df is None or df.empty:
            return {}

        # 이동평균 거래량 계산
        volume_ma = df['volume'].rolling(window=20).mean()
        current_volume = df['volume'].iloc[-1]
        
        volume_trend = {
            'current_volume': current_volume,
            'volume_ma': volume_ma.iloc[-1],
            'is_volume_increasing': current_volume > volume_ma.iloc[-1],
            'volume_strength': current_volume / volume_ma.iloc[-1]
        }
        
        return volume_trend

    def analyze_price_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        가격 모멘텀 분석
        """
        if df is None or df.empty:
            return {}

        # 최근 가격 변동성 계산
        returns = df['close'].pct_change()
        volatility = returns.std()
        
        # 추세 강도 계산
        price_ma = df['close'].rolling(window=20).mean()
        trend_strength = (df['close'].iloc[-1] - price_ma.iloc[-1]) / price_ma.iloc[-1]
        
        momentum = {
            'volatility': volatility,
            'trend_strength': trend_strength,
            'is_trending_up': trend_strength > 0,
            'momentum_strength': abs(trend_strength)
        }
        
        return momentum

    def get_market_sentiment(self, volume_trend: Dict[str, Any], 
                           momentum: Dict[str, Any]) -> str:
        """
        거래량과 모멘텀을 기반으로 시장 심리 판단
        """
        # 거래량 신호
        volume_signal = 1 if volume_trend['is_volume_increasing'] else -1
        
        # 모멘텀 신호
        momentum_signal = 1 if momentum['is_trending_up'] else -1
        
        # 강도 계산
        strength = (volume_trend['volume_strength'] * momentum['momentum_strength'])
        
        # 종합 신호
        combined_signal = volume_signal * momentum_signal * strength
        
        if combined_signal > 0.5:
            return 'bullish'
        elif combined_signal < -0.5:
            return 'bearish'
        return 'neutral' 