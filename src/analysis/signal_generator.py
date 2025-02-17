from typing import Dict, Any
import pandas as pd
import numpy as np
from src.analysis.technical_indicators import TechnicalIndicators

class SignalGenerator:
    """매매 신호 생성 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        매매 신호 생성기 초기화
        
        Args:
            df: OHLCV 데이터가 포함된 DataFrame
        """
        self.df = df
        self.indicators = TechnicalIndicators(df)
        
    def generate_signal(self, market_state: Any) -> Dict[str, Any]:
        """매매 신호 생성"""
        try:
            # 데이터 길이 체크
            if len(self.df) < 20:
                print("신호 생성을 위한 데이터가 부족합니다")
                return self._create_empty_signal()
                
            # 현재 가격
            current_price = self.df['close'].iloc[-1]
            if pd.isna(current_price) or current_price == 0:
                print("현재 가격이 유효하지 않습니다")
                return self._create_empty_signal()
                
            # 기술적 지표 계산
            rsi = self.indicators._calculate_rsi()
            macd = self.indicators._calculate_macd()
            
            # 지표 유효성 검사
            if pd.isna(rsi.iloc[-1]) or pd.isna(macd['macd'].iloc[-1]) or pd.isna(macd['signal'].iloc[-1]):
                print("기술적 지표 계산 결과가 유효하지 않습니다")
                return self._create_empty_signal()
                
            # 매매 신호 판단
            signal_type = 'none'
            confidence = 0.0
            
            # RSI 기반 신호
            if rsi.iloc[-1] < 30:  # 과매도
                signal_type = 'buy'
                confidence += 30
            elif rsi.iloc[-1] > 70:  # 과매수
                signal_type = 'sell'
                confidence += 30
                
            # MACD 기반 신호
            if macd['macd'].iloc[-1] > macd['signal'].iloc[-1]:  # 골든크로스
                if signal_type == 'buy':
                    confidence += 30
                else:
                    signal_type = 'buy'
                    confidence += 20
            elif macd['macd'].iloc[-1] < macd['signal'].iloc[-1]:  # 데드크로스
                if signal_type == 'sell':
                    confidence += 30
                else:
                    signal_type = 'sell'
                    confidence += 20
                    
            # 추세 강도 반영
            trend_strength = getattr(market_state, 'trend_strength', 0)
            if trend_strength > 25:
                confidence += 10
                
            # 진입가격과 스탑로스 설정
            entry_price = current_price
            stop_loss = 0.0
            
            if signal_type == 'buy':
                stop_loss = self.df['low'].tail(20).min()
            elif signal_type == 'sell':
                stop_loss = self.df['high'].tail(20).max()
                
            return {
                'signal': signal_type,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss
            }
            
        except Exception as e:
            print(f"신호 생성 오류: {str(e)}")
            return self._create_empty_signal()
            
    def _create_empty_signal(self) -> Dict[str, Any]:
        """빈 신호 생성"""
        return {
            'signal': 'none',
            'confidence': 0.0,
            'entry_price': 0.0,
            'stop_loss': 0.0
        } 