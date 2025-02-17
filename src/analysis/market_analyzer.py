from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from src.analysis.market_state import MarketState
from src.analysis.technical_indicators import TechnicalIndicators

class MarketAnalyzer:
    """시장 분석을 수행하는 클래스"""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # 필요한 시점에 임포트
        from src.analysis.technical_indicators import TechnicalIndicators
        self.indicators = TechnicalIndicators(df)
        
    def analyze_market_state(self) -> MarketState:
        """시장 상태 분석"""
        try:
            # 시장 구조 분석
            market_structure = self.indicators.calculate_market_structure()
            
            # 추세 강도 계산
            trend_strength = self.indicators.calculate_trend_strength()
            
            # 시장 상태 생성
            market_state = MarketState(
                trend_strength=trend_strength,
                trend_direction=market_structure['trend'],
                volatility=market_structure['volatility_regime'],
                regime=self._determine_market_regime(market_structure),
                support_levels=market_structure.get('support_levels', []),
                resistance_levels=market_structure.get('resistance_levels', []),
                risk_level=self._assess_risk_level(market_structure)
            )
            
            return market_state
            
        except Exception as e:
            print(f"시장 상태 분석 오류: {str(e)}")
            return MarketState()
            
    def _determine_market_regime(self, market_structure: Dict[str, Any]) -> str:
        """시장 레짐 결정"""
        trend = market_structure.get('trend', 'ranging')
        
        if trend in ['strong_uptrend', 'weak_uptrend']:
            return 'uptrend'
        elif trend in ['strong_downtrend', 'weak_downtrend']:
            return 'downtrend'
        else:
            return 'ranging'
            
    def _assess_risk_level(self, market_structure: Dict[str, Any]) -> str:
        """리스크 레벨 평가"""
        volatility = market_structure.get('volatility_regime', 'normal_volatility')
        
        if volatility == 'high_volatility':
            return 'high'
        elif volatility == 'low_volatility':
            return 'low'
        else:
            return 'medium'

    def _calculate_atr(self, period: int = 14) -> pd.Series:
        """ATR(Average True Range) 계산"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
        
    def _calculate_adx(self, period: int = 14) -> float:
        """ADX(Average Directional Index) 계산"""
        try:
            # 데이터가 충분하지 않으면 0 반환
            if len(self.df) < period * 2:
                print("ADX 계산을 위한 데이터가 부족합니다")
                return 0.0
                
            high = self.df['high']
            low = self.df['low']
            close = self.df['close']
            
            # True Range 계산
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement 계산
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smoothed TR and DM
            tr_smooth = tr.ewm(span=period, adjust=False).mean()  # EMA 사용
            pos_dm_smooth = pd.Series(pos_dm).ewm(span=period, adjust=False).mean()
            neg_dm_smooth = pd.Series(neg_dm).ewm(span=period, adjust=False).mean()
            
            # 0으로 나누기 방지
            tr_smooth = tr_smooth.replace(0, np.nan)
            
            # Directional Indicators
            pdi = 100 * pos_dm_smooth / tr_smooth
            ndi = 100 * neg_dm_smooth / tr_smooth
            
            # 0으로 나누기 방지
            denominator = pdi + ndi
            denominator = denominator.replace(0, np.nan)
            
            # ADX
            dx = 100 * abs(pdi - ndi) / denominator
            adx = dx.ewm(span=period, adjust=False).mean()  # EMA 사용
            
            # NaN 값 처리
            if pd.isna(adx.iloc[-1]):
                print("ADX 계산 결과가 유효하지 않습니다")
                return 0.0
                
            return float(adx.iloc[-1])
            
        except Exception as e:
            print(f"ADX 계산 오류: {str(e)}")
            return 0.0
        
    def _calculate_trend_strength(self) -> float:
        """추세 강도 계산"""
        try:
            # 데이터 길이 체크
            if len(self.df) < 50:
                print("추세 강도 계산을 위한 데이터가 부족합니다")
                return 0.0
                
            # 이동평균선 계산
            sma20 = self.df['close'].rolling(20).mean()
            sma50 = self.df['close'].rolling(50).mean()
            
            # NaN 값 체크
            if pd.isna(sma20.iloc[-1]) or pd.isna(sma50.iloc[-1]):
                print("이동평균 계산 결과가 유효하지 않습니다")
                return 0.0
                
            # ADX 계산
            adx = self._calculate_adx()
            if pd.isna(adx) or adx == 0:
                print("ADX 계산 결과가 유효하지 않습니다")
                return 0.0
                
            # 현재 가격과 이동평균 비교
            current_price = self.df['close'].iloc[-1]
            ma20_last = sma20.iloc[-1]
            ma50_last = sma50.iloc[-1]
            
            # 추세 방향 결정
            if current_price > ma20_last and ma20_last > ma50_last:
                trend_direction = 1  # 상승 추세
            elif current_price < ma20_last and ma20_last < ma50_last:
                trend_direction = -1  # 하락 추세
            else:
                trend_direction = 0  # 횡보
                
            # 최종 추세 강도 계산
            trend_strength = adx * trend_direction
            
            # 결과 검증
            if pd.isna(trend_strength):
                print("추세 강도 계산 결과가 유효하지 않습니다")
                return 0.0
                
            return float(trend_strength)
            
        except Exception as e:
            print(f"추세 강도 계산 오류: {str(e)}")
            return 0.0
        
    def _find_key_levels(self) -> Tuple[List[float], List[float]]:
        """주요 지지/저항 레벨 탐색"""
        try:
            if len(self.df) < self.window:
                print("데이터가 충분하지 않습니다")
                return [], []  # 데이터가 충분하지 않으면 빈 리스트 반환
                
            # 피봇 포인트 찾기
            highs = self.df['high'].rolling(self.window).max()
            lows = self.df['low'].rolling(self.window).min()
            
            current_price = self.df['close'].iloc[-1]
            
            # NaN 값 제거
            highs = highs.dropna()
            lows = lows.dropna()
            
            if len(highs) == 0 or len(lows) == 0:
                print("유효한 고가/저가 데이터가 없습니다")
                return [], []
                
            # 지지/저항 레벨 필터링
            support_levels = []
            resistance_levels = []
            
            # 최근 고점/저점 필터링
            for level in lows.unique():
                if level < current_price:
                    support_levels.append(float(level))
                    
            for level in highs.unique():
                if level > current_price:
                    resistance_levels.append(float(level))
                    
            # 정렬 및 상위/하위 3개 선택
            support_levels = sorted(support_levels, reverse=True)[:3]
            resistance_levels = sorted(resistance_levels)[:3]
            
            # 빈 리스트 체크
            if not support_levels and not resistance_levels:
                # 대체 레벨 계산
                avg_price = self.df['close'].mean()
                std_price = self.df['close'].std()
                
                support_levels = [
                    float(current_price - std_price),
                    float(current_price - std_price * 1.5),
                    float(current_price - std_price * 2)
                ]
                
                resistance_levels = [
                    float(current_price + std_price),
                    float(current_price + std_price * 1.5),
                    float(current_price + std_price * 2)
                ]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            print(f"레벨 탐색 오류: {str(e)}")
            return [], []
        
    def _calculate_support_resistance(self) -> Tuple[List[float], List[float]]:
        """지지/저항 레벨 계산"""
        # 피봇 포인트 계산
        high = self.df['high'].iloc[-20:]  # 최근 20개 봉 사용
        low = self.df['low'].iloc[-20:]
        close = self.df['close'].iloc[-20:]
        
        pivot = (high + low + close) / 3
        
        # 지지선 계산
        s1 = pivot - (high - low)
        s2 = pivot - 2 * (high - low)
        
        # 저항선 계산
        r1 = pivot + (high - low)
        r2 = pivot + 2 * (high - low)
        
        # 현재가 기준으로 정렬
        current_price = close.iloc[-1]
        
        support_levels = []
        resistance_levels = []
        
        # 지지선 필터링 및 정렬
        s_levels = pd.concat([s1, s2]).sort_values(ascending=False)
        for level in s_levels.unique():
            if level < current_price:
                support_levels.append(float(level))
        
        # 저항선 필터링 및 정렬
        r_levels = pd.concat([r1, r2]).sort_values()
        for level in r_levels.unique():
            if level > current_price:
                resistance_levels.append(float(level))
        
        return support_levels, resistance_levels 

    def _calculate_volatility(self) -> float:
        """
        변동성 수준 계산
        Returns:
            float: 현재 변동성 수준 (ATR 기반)
        """
        try:
            atr = self._calculate_atr()
            if len(atr) < self.window:
                return 1.0  # 데이터가 충분하지 않으면 기본값 반환
                
            current_atr = atr.iloc[-1]
            avg_atr = atr.rolling(self.window).mean().iloc[-1]
            
            # 0으로 나누기 방지
            if pd.isna(avg_atr) or avg_atr == 0:
                return 1.0
                
            return current_atr / avg_atr
            
        except Exception as e:
            print(f"변동성 계산 오류: {str(e)}")
            return 1.0  # 오류 발생시 기본값 반환 