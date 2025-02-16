from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .data_collector import DataCollector

class MarketLearner:
    """시장 학습 및 분석 모델"""
    
    def __init__(self, collector: DataCollector):
        self.logger = logging.getLogger('learner')
        self.collector = collector
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.performance_history: List[Dict[str, Any]] = []
        
    def learn(self, symbol: str, days: int = 30) -> bool:
        """과거 데이터로 학습"""
        try:
            # 데이터 수집
            df = self.collector.get_historical_data(symbol, days=days)
            if df.empty:
                raise ValueError("No data available for learning")
            
            # 학습 데이터 준비
            X, y = self._prepare_features(df)
            
            # 학습 실행
            self.model.fit(X, y)
            
            # 성능 평가
            performance = self._evaluate_performance(df, X, y)
            self.performance_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'performance': performance
            })
            
            self.logger.info(f"Learning completed for {symbol}: {performance}")
            return True
            
        except Exception as e:
            self.logger.error(f"Learning failed: {e}")
            return False
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """현재 시장 상태 분석"""
        try:
            # 최근 데이터 수집
            df = self.collector.get_historical_data(symbol, days=5)
            if df.empty:
                raise ValueError("No data available for analysis")
            
            # 특성 추출
            X, _ = self._prepare_features(df)
            
            # 예측 확률
            proba = self.model.predict_proba(X)[-1]
            
            # 신호 강도 계산
            signal_strength = abs(proba[1] - proba[0])
            
            return {
                'signal': 'BUY' if proba[1] > proba[0] else 'SELL',
                'strength': float(signal_strength),
                'confidence': float(max(proba)),
                'features': self._get_current_features(df)
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                'signal': 'NEUTRAL',
                'strength': 0.0,
                'confidence': 0.0,
                'features': {}
            }
    
    def _prepare_features(self, df: pd.DataFrame) -> tuple:
        """학습 특성 준비"""
        # 기술적 지표 기반 특성
        features = pd.DataFrame()
        
        # 추세 특성
        features['trend'] = (df['close'] > df['MA20']).astype(int)
        features['trend_strength'] = (df['close'] - df['MA20']) / df['MA20']
        
        # 모멘텀 특성
        features['rsi'] = df['RSI'] / 100
        features['macd'] = df['MACD'] / df['close']
        
        # 변동성 특성
        features['bb_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        features['volatility'] = df['close'].rolling(20).std() / df['close']
        
        # 거래량 특성
        features['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # 레이블 생성 (다음 봉의 상승/하락)
        labels = (df['close'].shift(-1) > df['close']).astype(int)
        
        # NaN 제거
        features = features.dropna()
        labels = labels[features.index]
        
        # 스케일링
        X = self.scaler.fit_transform(features)
        
        return X, labels
    
    def _evaluate_performance(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """모델 성능 평가"""
        # 예측
        y_pred = self.model.predict(X)
        
        # 정확도
        accuracy = (y == y_pred).mean()
        
        # 수익성 평가
        returns = []
        position = None
        entry_price = None
        
        for i, pred in enumerate(y_pred):
            price = df['close'].iloc[i]
            
            if pred == 1 and position != 'long':  # 매수 신호
                position = 'long'
                entry_price = price
            elif pred == 0 and position == 'long':  # 매도 신호
                if entry_price:
                    returns.append((price - entry_price) / entry_price)
                position = None
                entry_price = None
        
        return {
            'accuracy': float(accuracy),
            'avg_return': float(np.mean(returns)) if returns else 0.0,
            'win_rate': float(sum(r > 0 for r in returns) / len(returns)) if returns else 0.0
        }
    
    def _get_current_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """현재 시장 특성 추출"""
        last_row = df.iloc[-1]
        return {
            'trend': float(last_row['close'] > last_row['MA20']),
            'trend_strength': float((last_row['close'] - last_row['MA20']) / last_row['MA20']),
            'rsi': float(last_row['RSI']),
            'macd': float(last_row['MACD']),
            'bb_position': float((last_row['close'] - last_row['BB_lower']) / 
                               (last_row['BB_upper'] - last_row['BB_lower'])),
            'volume_trend': float(df['volume'].rolling(5).mean().iloc[-1] / 
                                df['volume'].rolling(20).mean().iloc[-1])
        } 