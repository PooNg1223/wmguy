import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

class MarketAnalysis:
    """시장 분석 및 세력 감지"""
    
    def __init__(self):
        self.logger = logging.getLogger('analysis')
    
    def detect_whale_activity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """세력 활동 감지"""
        try:
            # 1. 거래량 기반 세력 감지
            volume_mean = df['volume'].rolling(window=20).mean()
            volume_std = df['volume'].rolling(window=20).std()
            
            # 거래량 급증 기준 (동적 임계값)
            volume_threshold = volume_mean + (2 * volume_std)
            large_trades = df[df['volume'] > volume_threshold]
            
            # 2. 가격 임팩트 분석
            price_impacts = []
            for i in range(len(large_trades)):
                current_idx = large_trades.index[i]
                df_idx_list = df.index.tolist()
                current_loc = df_idx_list.index(current_idx)
                future_loc = min(current_loc + 5, len(df_idx_list) - 1)
                future_idx = df_idx_list[future_loc]
                
                # Pandas Series에서 단일 값 추출
                future_price = df.loc[future_idx, 'close'].item()
                current_price = df.loc[current_idx, 'close'].item()
                
                # 유효성 검사
                if future_price > 0 and current_price > 0:
                    price_change = (future_price / current_price) - 1.0
                    price_impacts.append(float(price_change))
            
            # 3. 매수/매도 세력 구분
            buying_pressure = df['close'] > df['open']  # Series 비교는 그대로 사용
            whale_buys = large_trades[buying_pressure.loc[large_trades.index]]
            whale_sells = large_trades[~buying_pressure.loc[large_trades.index]]
            
            # 결과 계산
            avg_price_impact = float(np.mean(price_impacts)) if price_impacts else 0.0
            activity_score = self._calculate_activity_score(
                whale_buys=len(whale_buys),
                whale_sells=len(whale_sells),
                price_impact=avg_price_impact,
                consecutive_pattern=self._detect_consecutive_trades(large_trades),
                liquidity=self._analyze_liquidity(df)
            )
            
            return {
                'is_active': bool(activity_score > 0.7),
                'activity_score': float(activity_score),
                'direction': 'BUY' if len(whale_buys) > len(whale_sells) else 'SELL',
                'large_trades_count': int(len(large_trades)),
                'buy_pressure': float(len(whale_buys) / max(len(large_trades), 1)),
                'avg_price_impact': float(avg_price_impact),
                'consecutive_pattern': int(self._detect_consecutive_trades(large_trades)),
                'liquidity_score': float(self._analyze_liquidity(df))
            }
            
        except Exception as e:
            self.logger.error(f"Whale detection failed: {e}")
            return {
                'is_active': False,
                'activity_score': 0.0,
                'direction': 'NEUTRAL',
                'large_trades_count': 0,
                'buy_pressure': 0.0,
                'avg_price_impact': 0.0,
                'consecutive_pattern': 0,
                'liquidity_score': 0.0
            }
    
    def _detect_consecutive_trades(self, large_trades: pd.DataFrame) -> int:
        """연속된 대규모 거래 패턴 감지"""
        if large_trades.empty:
            return 0
            
        consecutive = 0
        max_consecutive = 0
        
        for i in range(1, len(large_trades)):
            time_diff = large_trades.index[i] - large_trades.index[i-1]
            if time_diff <= pd.Timedelta(minutes=30):  # 30분 이내 연속 거래
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
                
        return max_consecutive
    
    def _analyze_liquidity(self, df: pd.DataFrame) -> float:
        """시장 유동성 분석"""
        try:
            # 스프레드 계산
            typical_spread = (df['high'] - df['low']).mean() / df['close'].mean()
            
            # 거래량 안정성
            volume_stability = 1 - (df['volume'].std() / df['volume'].mean())
            
            # 가격 변동성
            price_stability = 1 - (df['close'].pct_change().std())
            
            # 종합 점수 (0~1)
            liquidity_score = (
                (1 - typical_spread) * 0.4 +
                volume_stability * 0.3 +
                price_stability * 0.3
            )
            
            return float(np.clip(liquidity_score, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Liquidity analysis failed: {e}")
            return 0.0
    
    def _calculate_activity_score(self, whale_buys: int, whale_sells: int,
                                price_impact: float, consecutive_pattern: int,
                                liquidity: float) -> float:
        """세력 활동 점수 계산"""
        try:
            # 거래 빈도 점수
            frequency_score = min((whale_buys + whale_sells) / 10, 1.0)
            
            # 가격 영향력 점수
            impact_score = min(abs(price_impact) * 10, 1.0)
            
            # 연속성 점수
            consecutive_score = min(consecutive_pattern / 5, 1.0)
            
            # 유동성 고려
            liquidity_factor = 1 + (liquidity - 0.5)  # 0.5~1.5
            
            # 종합 점수 계산
            final_score = (
                frequency_score * 0.4 +
                impact_score * 0.3 +
                consecutive_score * 0.3
            ) * liquidity_factor
            
            return float(np.clip(final_score, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Activity score calculation failed: {e}")
            return 0.0 