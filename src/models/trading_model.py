from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from src.analysis.technical_indicators import TechnicalAnalyzer
from src.analysis.market_sentiment import MarketSentimentAnalyzer

class TradingModel:
    """
    트레이딩 결정을 내리는 메인 모델
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tech_analyzer = TechnicalAnalyzer(config)
        self.sentiment_analyzer = MarketSentimentAnalyzer(config)
        self.position = None
        self.risk_per_trade = config['trading']['risk_per_trade']
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              account_balance: float) -> float:
        """
        리스크 기반 포지션 사이즈 계산
        """
        if stop_loss >= entry_price:
            return 0.0
            
        risk_amount = account_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        position_size = risk_amount / price_risk
        
        return position_size
        
    def find_support_resistance(self, df: pd.DataFrame, 
                              window: int = 20) -> Dict[str, float]:
        """
        지지/저항 레벨 찾기
        """
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        recent_high = highs.iloc[-window:].max()
        recent_low = lows.iloc[-window:].min()
        
        return {
            'support': recent_low,
            'resistance': recent_high
        }
        
    def calculate_stop_loss(self, df: pd.DataFrame, 
                          position_type: str) -> float:
        """
        ATR 기반 스탑로스 계산
        """
        # ATR 계산 (20일)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=20).mean().iloc[-1]
        
        current_price = df['close'].iloc[-1]
        
        if position_type == 'long':
            return current_price - (atr * 2)  # 2 ATR for stop loss
        else:
            return current_price + (atr * 2)
            
    def analyze_trade_opportunity(self, 
                                kline_data: pd.DataFrame,
                                timeframe: str) -> Dict[str, Any]:
        """
        거래 기회 분석
        """
        if kline_data is None or kline_data.empty:
            return {'signal': 'neutral'}
            
        # 기술적 지표 계산
        indicators = self.tech_analyzer.calculate_indicators(kline_data)
        tech_signals = self.tech_analyzer.analyze_signals(indicators)
        tech_signal = self.tech_analyzer.get_combined_signal(tech_signals)
        
        # 시장 심리 분석
        volume_trend = self.sentiment_analyzer.analyze_volume_trend(kline_data)
        momentum = self.sentiment_analyzer.analyze_price_momentum(kline_data)
        sentiment = self.sentiment_analyzer.get_market_sentiment(volume_trend, momentum)
        
        # 지지/저항 레벨
        levels = self.find_support_resistance(kline_data)
        current_price = kline_data['close'].iloc[-1]
        
        trade_signal = {
            'signal': 'neutral',
            'entry_price': current_price,
            'stop_loss': None,
            'timeframe': timeframe,
            'confidence': 0.0
        }
        
        # 매수 신호 조건
        if (tech_signal == 'buy' and sentiment == 'bullish' and
            current_price > levels['support']):
            trade_signal.update({
                'signal': 'buy',
                'stop_loss': self.calculate_stop_loss(kline_data, 'long'),
                'confidence': self._calculate_confidence(indicators, volume_trend, momentum)
            })
            
        # 매도 신호 조건
        elif (tech_signal == 'sell' and sentiment == 'bearish' and
              current_price < levels['resistance']):
            trade_signal.update({
                'signal': 'sell',
                'stop_loss': self.calculate_stop_loss(kline_data, 'short'),
                'confidence': self._calculate_confidence(indicators, volume_trend, momentum)
            })
            
        return trade_signal
        
    def _calculate_confidence(self, indicators: Dict[str, Any],
                            volume_trend: Dict[str, Any],
                            momentum: Dict[str, Any]) -> float:
        """
        거래 신호의 신뢰도 계산
        """
        confidence_factors = []
        
        # RSI 신뢰도
        rsi = indicators['rsi']['value'].iloc[-1]
        if rsi < 30:
            confidence_factors.append(0.8)  # 과매도
        elif rsi > 70:
            confidence_factors.append(0.8)  # 과매수
        else:
            confidence_factors.append(0.5)
            
        # 거래량 신뢰도
        volume_strength = volume_trend['volume_strength']
        confidence_factors.append(min(volume_strength / 2, 1.0))
        
        # 모멘텀 신뢰도
        momentum_strength = momentum['momentum_strength']
        confidence_factors.append(min(momentum_strength, 1.0))
        
        return sum(confidence_factors) / len(confidence_factors) 