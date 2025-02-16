from typing import Dict, Any, List, Optional, Tuple, Mapping
import numpy as np
from datetime import datetime
import logging
import talib

class TradingStrategy:
    """자가 학습형 하이브리드 트레이딩 전략"""
    
    def __init__(self):
        self.logger = logging.getLogger('strategy')
        self.signals_history: List[Dict[str, Any]] = []
        self.trades_history: List[Dict[str, Any]] = []
        self.performance: Dict[str, Dict[str, float]] = {}
        
        # 전략 가중치 (학습을 통해 조정됨)
        self.weights = {
            'tradingview': 0.4,    # 트레이딩뷰 시그널
            'technical': 0.3,      # 기술적 지표
            'momentum': 0.3        # 모멘텀/추세
        }
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        여러 전략을 결합하여 최종 매매 신호 생성
        
        Args:
            data: {
                'symbol': str,
                'price': float,
                'tradingview_signals': List[Dict],
                'technical_indicators': Dict,
                'market_data': Dict
            }
        """
        try:
            signals = []
            
            # 1. 트레이딩뷰 시그널 분석
            tv_signal = self._analyze_tradingview(data['tradingview_signals'])
            signals.append(('tradingview', tv_signal))
            
            # 2. 기술적 지표 분석
            tech_signal = self._analyze_technical(data['technical_indicators'])
            signals.append(('technical', tech_signal))
            
            # 3. 모멘텀/추세 분석
            mom_signal = self._analyze_momentum(data['market_data'])
            signals.append(('momentum', mom_signal))
            
            # 가중치 적용하여 최종 신호 생성
            final_signal = self._combine_signals(signals)
            
            # 신호 기록
            self.signals_history.append({
                'timestamp': datetime.now(),
                'data': data,
                'signals': signals,
                'final_signal': final_signal
            })
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Strategy analysis failed: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0.0}
    
    def learn(self):
        """과거 데이터를 기반으로 전략 가중치 최적화"""
        try:
            if len(self.trades_history) < 10:  # 충분한 데이터가 쌓일 때까지 대기
                return
            
            # 각 전략별 성과 계산
            strategy_performance = self._calculate_performance()
            
            # 가중치 최적화
            total_pnl = sum(p['pnl'] for p in strategy_performance.values())
            if total_pnl > 0:
                for strategy, perf in strategy_performance.items():
                    self.weights[strategy] = perf['pnl'] / total_pnl
            
            self.logger.info(f"Strategy weights updated: {self.weights}")
            
        except Exception as e:
            self.logger.error(f"Strategy learning failed: {e}")
    
    def _calculate_performance(self) -> Mapping[str, Dict[str, float]]:
        """각 전략별 성과 계산"""
        performance: Dict[str, Dict[str, float]] = {
            'tradingview': {'wins': 0.0, 'losses': 0.0, 'pnl': 0.0},
            'technical': {'wins': 0.0, 'losses': 0.0, 'pnl': 0.0},
            'momentum': {'wins': 0.0, 'losses': 0.0, 'pnl': 0.0}
        }
        
        for trade in self.trades_history[-100:]:  # 최근 100개 거래만 분석
            for strategy, signal in trade['signals']:
                if trade['pnl'] > 0:
                    performance[strategy]['wins'] += 1
                else:
                    performance[strategy]['losses'] += 1
                performance[strategy]['pnl'] += trade['pnl']
        
        return performance 

    def _analyze_tradingview(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """트레이딩뷰 시그널 분석"""
        if not signals:
            return {'signal': 'NEUTRAL', 'strength': 0.0}
            
        # 시그널 강도 계산
        buy_signals = sum(1 for s in signals if s.get('action') == 'BUY')
        sell_signals = sum(1 for s in signals if s.get('action') == 'SELL')
        
        total_signals = len(signals)
        if total_signals == 0:
            return {'signal': 'NEUTRAL', 'strength': 0.0}
            
        strength = (buy_signals - sell_signals) / total_signals
        
        if strength > 0.3:
            return {'signal': 'BUY', 'strength': strength}
        elif strength < -0.3:
            return {'signal': 'SELL', 'strength': abs(strength)}
        return {'signal': 'NEUTRAL', 'strength': 0.0}
    
    def _analyze_technical(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """기술적 지표 분석"""
        try:
            close_prices = np.array(indicators['close_prices'])
            
            # RSI
            rsi = talib.RSI(close_prices)[-1]
            
            # MACD
            macd, signal, _ = talib.MACD(close_prices)
            macd_latest = macd[-1] - signal[-1]
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close_prices)
            bb_position = (close_prices[-1] - lower[-1]) / (upper[-1] - lower[-1])
            
            # 종합 신호 생성
            signal_strength = 0.0
            if rsi < 30 and macd_latest > 0:
                signal_strength = 0.8  # 강한 매수
            elif rsi > 70 and macd_latest < 0:
                signal_strength = -0.8  # 강한 매도
            else:
                signal_strength = (50 - rsi) / 50  # 중립~약한 신호
            
            return {
                'signal': 'BUY' if signal_strength > 0 else 'SELL',
                'strength': abs(signal_strength)
            }
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0.0}
    
    def _analyze_momentum(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """모멘텀/추세 분석"""
        try:
            volumes = np.array(market_data['volumes'])
            prices = np.array(market_data['prices'])
            
            # 거래량 증가율
            volume_change = volumes[-1] / volumes[-20:].mean()
            
            # 가격 모멘텀
            momentum = (prices[-1] / prices[-20:].mean()) - 1
            
            # 변동성
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            
            # 신호 강도 계산
            strength = (
                0.4 * momentum +
                0.3 * (volume_change - 1) +
                0.3 * (1 - volatility)
            )
            
            return {
                'signal': 'BUY' if strength > 0 else 'SELL',
                'strength': abs(strength)
            }
            
        except Exception as e:
            self.logger.error(f"Momentum analysis failed: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0.0}
    
    def _combine_signals(self, signals: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """여러 전략의 신호를 결합"""
        try:
            weighted_strength = 0.0
            
            for strategy, signal in signals:
                if strategy in self.weights:
                    signal_value = 1 if signal['signal'] == 'BUY' else -1
                    weighted_strength += (
                        signal_value *
                        signal['strength'] *
                        self.weights[strategy]
                    )
            
            return {
                'signal': 'BUY' if weighted_strength > 0.1 else 'SELL' if weighted_strength < -0.1 else 'NEUTRAL',
                'strength': abs(weighted_strength),
                'weighted_strength': weighted_strength
            }
            
        except Exception as e:
            self.logger.error(f"Signal combination failed: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'weighted_strength': 0.0} 