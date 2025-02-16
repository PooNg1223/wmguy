from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .data_collector import DataCollector
from sklearn.metrics import accuracy_score, precision_score, recall_score
from .market_analysis import MarketAnalysis

class MarketLearner:
    """시장 학습 및 분석 모델"""
    
    # 특징 그룹 정의
    feature_groups = {
        'trend': [
            'MA5', 'MA20', 'MA60',
            'BB_middle', 'BB_upper', 'BB_lower',
            'Trend_Strength', 'Pattern'
        ],
        'momentum': [
            'RSI', 'MACD', 'MACD_Hist',
            'ROC', 'Momentum', 'Volume_Ratio'
        ],
        'volatility': [
            'ATR', 'Volatility', 'BB_width',
            'Higher_High', 'Lower_Low', 'Volume_Surge'
        ]
    }
    
    def __init__(self, collector: DataCollector):
        self.logger = logging.getLogger('learner')
        self.collector = collector
        self.market_analyzer = MarketAnalysis()
        
        # 앙상블 모델 구성
        self.models = {
            'trend': RandomForestClassifier(n_estimators=100),
            'momentum': RandomForestClassifier(n_estimators=100),
            'volatility': RandomForestClassifier(n_estimators=100)
        }
        
        self.scalers = {
            'trend': StandardScaler(),
            'momentum': StandardScaler(),
            'volatility': StandardScaler()
        }
        
        self.performance_history = []
        
    def learn(self, symbol: str) -> bool:
        """시장 학습"""
        try:
            df = self.collector.get_historical_data(symbol)
            if df.empty:
                return False
            
            # 특성 생성
            features = self._create_features(df)
            labels = self._create_labels(df)
            
            # 모델별 학습
            for model_type in self.models:
                X = features[model_type]
                y = labels[model_type]
                
                if len(X) == 0 or len(y) == 0:
                    continue
                
                # 스케일링
                X_scaled = self.scalers[model_type].fit_transform(X)
                
                # 학습
                self.models[model_type].fit(X_scaled, y)
            
            # 성과 평가
            performance = self._evaluate_performance(df)
            self.performance_history.append({
                'timestamp': pd.Timestamp.now(),
                'symbol': symbol,
                'performance': performance
            })
            
            return True
            
        except Exception:
            return False
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """시장 분석"""
        try:
            df = self.collector.get_historical_data(symbol)
            if df.empty:
                return {
                    'signal': 'NEUTRAL',
                    'strength': 0.0,
                    'confidence': 0.0
                }
            
            # 특성 생성
            features = self._create_features(df)
            
            # 모델별 예측
            predictions = {}
            confidences = {}
            
            for model_type in self.models:
                X = features[model_type]
                if len(X) == 0:
                    continue
                
                X_scaled = self.scalers[model_type].transform(X)
                pred_proba = self.models[model_type].predict_proba(X_scaled)
                
                predictions[model_type] = pred_proba[-1, 1]  # 상승 확률
                confidences[model_type] = max(pred_proba[-1])
            
            # 앙상블 예측
            signal_strength = np.mean(list(predictions.values()))
            confidence = np.mean(list(confidences.values()))
            
            return {
                'signal': 'BUY' if signal_strength > 0.6 else 'SELL' if signal_strength < 0.4 else 'NEUTRAL',
                'strength': abs(signal_strength - 0.5) * 2,
                'confidence': confidence,
                'predictions': predictions
            }
            
        except Exception:
            return {
                'signal': 'NEUTRAL',
                'strength': 0.0,
                'confidence': 0.0
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
        X = self.scalers['trend'].fit_transform(features)
        
        return X, labels
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """샤프 비율 계산"""
        if not returns:
            return 0.0
        std = float(np.std(returns))
        if std == 0:
            return 0.0
        return float(np.mean(returns) / std)
    
    def _evaluate_performance(self, df: pd.DataFrame) -> float:
        """모델 성과 평가"""
        try:
            # 특성 생성
            features = self._create_features(df)
            labels = self._create_labels(df)
            
            # 모델별 예측
            predictions = {}
            confidences = {}
            
            for model_type in self.models:
                X = features[model_type]
                y = labels[model_type]
                
                if len(X) == 0 or len(y) == 0:
                    continue
                
                X_scaled = self.scalers[model_type].transform(X)
                pred_proba = self.models[model_type].predict_proba(X_scaled)
                
                predictions[model_type] = pred_proba[-1, 1]  # 상승 확률
                confidences[model_type] = max(pred_proba[-1])
            
            # 앙상블 예측
            signal_strength = np.mean(list(predictions.values()))
            confidence = float(np.mean(list(confidences.values())))
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
            return 0.0
    
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

    def _create_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """모델별 학습 특성 생성"""
        features = {}
        
        # 각 모델 타입별로 특성 생성
        for model_type, feature_list in self.feature_groups.items():
            try:
                # List comprehension 대신 명시적 리스트 생성
                feature_arrays = []
                for feature in feature_list:
                    # numpy array로 명시적 변환
                    feature_array = np.array(df[feature].values, dtype=np.float64)
                    feature_arrays.append(feature_array)
                
                # column_stack에 시퀀스 전달
                features[model_type] = np.column_stack(tuple(feature_arrays))
                
            except Exception as e:
                self.logger.error(f"Failed to create {model_type} features: {e}")
                features[model_type] = np.array([], dtype=np.float64)
        
        return features

    def _create_labels(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """모델별 학습 레이블 생성"""
        labels = {}
        
        try:
            # 미래 수익률 계산 (5봉 후)
            future_returns = df['close'].pct_change(5).shift(-5)
            
            # 모델별 레이블 생성
            for model_type in self.feature_groups:
                if model_type == 'trend':
                    # 추세 모델: 2% 이상 상승
                    labels[model_type] = (future_returns > 0.02).astype(int)
                elif model_type == 'momentum':
                    # 모멘텀 모델: 1% 이상 상승
                    labels[model_type] = (future_returns > 0.01).astype(int)
                elif model_type == 'volatility':
                    # 변동성 모델: ATR의 50% 이상 상승
                    volatility_threshold = df['ATR'] * 0.5
                    labels[model_type] = (future_returns > volatility_threshold).astype(int)
                
                # NaN 처리
                labels[model_type] = pd.Series(labels[model_type]).fillna(0).astype(int).values
        
        except Exception as e:
            self.logger.error(f"Failed to create labels: {e}")
            # 에러 발생 시 기본값 반환
            for model_type in self.feature_groups:
                labels[model_type] = np.zeros(len(df))
        
        return labels 