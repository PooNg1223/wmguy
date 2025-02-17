from typing import Dict, Any, Optional, Tuple, List, Union, cast
import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from src.analysis.position_manager import PositionManager, PositionConfig
from pandas import Series, Index, Interval
from pandas.core.indexes import interval as pd_interval
from src.analysis.market_state import MarketState  # MarketState 클래스 임포트

# MarketAnalyzer 타입 힌트를 위한 forward reference
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.analysis.market_analyzer import MarketAnalyzer

class TechnicalAnalyzer:
    """
    기술적 분석을 수행하는 클래스
    """
    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        """
        기술적 분석을 수행하는 클래스 초기화
        
        Args:
            df: OHLCV 데이터가 포함된 DataFrame
            config: 트레이딩 설정 (None인 경우 기본값 사용)
        """
        self.df = df.copy()
        self.indicators = TechnicalIndicators(df)
        
        # 설정 로드
        self.config = config or self._load_default_config()
        
        # 시장 분석기 초기화
        from src.analysis.market_analyzer import MarketAnalyzer  # 런타임에 임포트
        self.market_analyzer = MarketAnalyzer(df)
        
        # 포지션 매니저 초기화
        position_config = PositionConfig(
            risk_per_trade=self.config['trading']['risk_per_trade'],
            max_position_size=self.config['trading']['max_position_size'],
            min_confidence=self.config['trading']['min_confidence'],
            profit_target_ratio=self.config['risk_management']['profit_target_ratio'],
            trailing_stop=self.config['risk_management']['trailing_stop']
        )
        self.position_manager = PositionManager(position_config)
        
        # 기본 자본금 설정
        self.capital = self.config['backtest']['initial_balance']
        
        # 기술적 지표 설정
        self.indicators_config = self.config['indicators']

    def _load_default_config(self) -> Dict:
        """기본 설정값 로드"""
        return {
            'trading': {
                'risk_per_trade': 0.02,  # 거래당 리스크 비율
                'max_position_size': 0.3,  # 최대 포지션 크기
                'min_confidence': 60,  # 최소 신뢰도
                'stop_loss_atr_multiplier': 2.0,  # ATR 기반 스탑로스 배수
                'take_profit_ratio': 2.0  # 이익실현 비율
            },
            'indicators': {
                'rsi': {
                    'period': 14,
                    'overbought': 70,
                    'oversold': 30
                },
                'macd': {
                    'fast_period': 12,
                    'slow_period': 26,
                    'signal_period': 9
                },
                'bollinger': {
                    'period': 20,
                    'std_dev': 2
                },
                'stochastic': {
                    'k_period': 14,
                    'd_period': 3
                }
            },
            'risk_management': {
                'profit_target_ratio': 2.0,  # 이익실현 목표 비율
                'trailing_stop': True,  # 트레일링 스탑 사용 여부
                'trailing_stop_atr_multiplier': 1.5  # 트레일링 스탑 ATR 배수
            },
            'backtest': {
                'initial_balance': 10000,  # 초기 자본금
                'commission_rate': 0.001  # 수수료율
            }
        }

    def analyze(self) -> Dict[str, Any]:
        """종합적인 기술적 분석 수행"""
        # 시장 상태 분석
        market_state = self.market_analyzer.analyze_market_state()
        
        # 기술적 지표 계산 및 신호 생성
        indicators = self.calculate_indicators(self.df)
        signals = self.analyze_signals(indicators)
        filtered_signals = self._filter_signals(signals, market_state)
        
        # 최종 신호 및 신뢰도 계산
        combined_signal = self.get_combined_signal(filtered_signals)
        confidence = self._calculate_signal_confidence(
            signals=filtered_signals,
            trend_score=market_state.trend_strength,
            volatility_regime=market_state.volatility,
            market_regime=market_state.regime
        )
        
        # 진입가격 및 스탑로스 계산
        entry_price = self._calculate_entry_price(combined_signal, market_state)
        stop_loss = self._calculate_stop_loss(
            combined_signal,
            market_state.support_levels,
            market_state.resistance_levels
        )
        
        # 포지션 관리
        position = self.position_manager.calculate_position_size(
            capital=self.capital,
            entry_price=entry_price,
            stop_loss=stop_loss,
            confidence=confidence,
            market_state=market_state.as_dict
        )
        
        return {
            'signal': combined_signal,
            'confidence': confidence,
            'position': position,
            'market_state': market_state,
            'indicators': indicators,
            'signals': filtered_signals
        }
        
    def _calculate_signal_confidence(self, signals: Dict[str, str], 
                                   trend_score: float, 
                                   volatility_regime: str,
                                   market_regime: str = 'ranging') -> float:
        """매매 신호의 신뢰도 계산 (0-100)"""
        try:
            # 시그널 일치도 계산
            signal_count = len(signals)
            if signal_count == 0:
                return 0.0
                
            buy_count = sum(1 for s in signals.values() if s == 'buy')
            sell_count = sum(1 for s in signals.values() if s == 'sell')
            
            # 가장 많은 신호의 비율 계산
            max_signal_count = max(buy_count, sell_count)
            signal_agreement = (max_signal_count / signal_count) * 100
            
            # 변동성 가중치
            volatility_weight = {
                'low': 1.2,     # 낮은 변동성: 신뢰도 증가
                'medium': 1.0,  # 중간 변동성: 기본값
                'high': 0.8     # 높은 변동성: 신뢰도 감소
            }.get(volatility_regime, 1.0)
            
            # 시장 레짐 가중치
            regime_weight = {
                'uptrend': 1.2,    # 상승 추세: 신뢰도 증가
                'downtrend': 1.2,  # 하락 추세: 신뢰도 증가
                'ranging': 0.8     # 횡보: 신뢰도 감소
            }.get(market_regime, 1.0)
            
            # 추세 강도 가중치 (0-100 범위를 0.5-1.5 범위로 변환)
            trend_weight = 0.5 + (trend_score / 100)
            
            # 최종 신뢰도 계산
            confidence = signal_agreement * volatility_weight * trend_weight * regime_weight
            
            return min(max(confidence, 0), 100)
            
        except Exception as e:
            print(f"신뢰도 계산 오류: {str(e)}")
            return 0.0
        
    def _calculate_entry_price(self, signal: str, market_state: MarketState) -> float:
        """
        최적 진입 가격 계산
        
        Args:
            signal: 매매 신호 ('buy', 'sell', 'neutral')
            market_state: 시장 상태 정보
        """
        current_price = self.df['close'].iloc[-1]
        atr = self.indicators._calculate_atr().iloc[-1]
        
        if signal == 'buy':
            # 매수 시 지지선 근처에서 진입
            if market_state.support_levels:
                nearest_support = market_state.support_levels[0]
                return max(nearest_support, current_price - (atr * 0.5))
            else:
                return current_price - (atr * 0.5)
                
        elif signal == 'sell':
            # 매도 시 저항선 근처에서 진입
            if market_state.resistance_levels:
                nearest_resistance = market_state.resistance_levels[0]
                return min(nearest_resistance, current_price + (atr * 0.5))
            else:
                return current_price + (atr * 0.5)
                
        return current_price
        
    def _calculate_stop_loss(self, signal: str, 
                           support_levels: List[float], 
                           resistance_levels: List[float]) -> Optional[float]:
        """
        스탑로스 레벨 계산
        
        Args:
            signal: 매매 신호 ('buy', 'sell', 'neutral')
            support_levels: 지지 레벨 목록
            resistance_levels: 저항 레벨 목록
            
        Returns:
            Optional[float]: 계산된 스탑로스 가격. 계산할 수 없는 경우 None
        """
        current_price = self.df['close'].iloc[-1]
        atr = self.indicators._calculate_atr().iloc[-1]
        
        if signal == 'buy':
            if support_levels:
                # 가장 가까운 지지선 아래에 스탑로스 설정
                nearest_support = support_levels[0]
                return nearest_support - (atr * 0.5)  # ATR의 절반만큼 여유
            else:
                # ATR의 2배 만큼 아래에 설정
                return current_price - (atr * 2)
                
        elif signal == 'sell':
            if resistance_levels:
                # 가장 가까운 저항선 위에 스탑로스 설정
                nearest_resistance = resistance_levels[0]
                return nearest_resistance + (atr * 0.5)  # ATR의 절반만큼 여유
            else:
                # ATR의 2배 만큼 위에 설정
                return current_price + (atr * 2)
                
        return None  # neutral 신호인 경우

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
        여러 지표의 신호를 종합하여 최종 매매 신호 생성
        """
        # 각 신호 카운트
        buy_count = sum(1 for s in signals.values() if s == 'buy')
        sell_count = sum(1 for s in signals.values() if s == 'sell')
        total_signals = len(signals)
        
        # 신호 비율 계산
        buy_ratio = buy_count / total_signals
        sell_ratio = sell_count / total_signals
        
        # 신호 임계값 (최소 40% 이상의 동일 신호)
        threshold = 0.4
        
        if buy_ratio >= threshold and buy_ratio > sell_ratio:
            return 'buy'
        elif sell_ratio >= threshold and sell_ratio > buy_ratio:
            return 'sell'
        else:
            return 'neutral'

    def calculate_trend_strength(self) -> float:
        """추세 강도 계산"""
        try:
            # 데이터 길이 체크
            if len(self.df) < 20:  # 최소 20일 데이터 필요
                print("추세 강도 계산을 위한 데이터가 부족합니다")
                return 0.0
            
            # 기술적 지표 계산
            adx = self.indicators.calculate_adx()
            rsi = self.indicators.calculate_rsi()
            macd = self.indicators.calculate_macd()
            
            # 이동평균 방향
            ma20 = self.df['close'].rolling(window=20).mean()
            ma50 = self.df['close'].rolling(window=50).mean()
            ma_trend = 1 if ma20.iloc[-1] > ma50.iloc[-1] else -1
            
            # 볼린저 밴드 위치
            bb = self.indicators._calculate_bollinger_bands()
            bb_position = (self.df['close'].iloc[-1] - bb['middle'].iloc[-1]) / (bb['upper'].iloc[-1] - bb['middle'].iloc[-1])
            
            # 종합 점수 계산 (0-100)
            adx_score = min(float(adx) / 50.0, 1.0) * 30  # ADX: 30%
            rsi_score = abs(rsi.iloc[-1] - 50) / 50 * 20  # RSI: 20%
            macd_score = (1 if macd['macd'].iloc[-1] * ma_trend > 0 else 0) * 20  # MACD: 20%
            bb_score = abs(bb_position) * 30  # BB: 30%
            
            total_score = adx_score + rsi_score + macd_score + bb_score
            
            return round(total_score, 2)
            
        except Exception as e:
            print(f"추세 강도 계산 오류: {str(e)}")
            return 0.0
        
    def calculate_volume_profile(self, bins: int = 50) -> Tuple[float, Index]:
        """거래량 프로파일 분석"""
        try:
            # 가격 범위 설정
            price_range = pd.cut(self.df['close'], bins=bins)
            
            # 거래량 프로파일 계산
            volume_profile = self.df.groupby(price_range)['volume'].sum()
            
            # POC (Point of Control) 찾기
            poc_interval = cast(pd_interval.Interval, volume_profile.idxmax())
            poc = float(poc_interval.mid)
            
            # Value Area 계산 (70% 거래량)
            sorted_volumes = volume_profile.sort_values(ascending=False)
            cumsum_volumes = sorted_volumes.cumsum()
            value_area = sorted_volumes[cumsum_volumes.le(cumsum_volumes.sum() * 0.7)].index
            
            return poc, value_area
            
        except Exception as e:
            print(f"거래량 프로파일 계산 오류: {str(e)}")
            return 0.0, pd.Index([])
        
    def calculate_market_structure(self, window: int = 20) -> Dict[str, Any]:
        """
        시장 구조 분석
        - 주요 지지/저항 레벨
        - 추세 방향
        - 변동성 체계
        """
        try:
            # 주요 피봇 포인트 찾기
            highs = self.indicators.find_peaks(self.df['high'], window)
            lows = self.indicators.find_peaks(-self.df['low'], window)
            
            # 최근 데이터에 가중치 부여
            current_price = self.df['close'].iloc[-1]
            
            # 지지/저항 레벨 필터링
            support_levels = lows[lows < current_price].sort_values(ascending=False)
            resistance_levels = highs[highs > current_price].sort_values()
            
            # 최근 추세 방향
            recent_trend = self._analyze_recent_trend()
            
            # 변동성 체계
            atr = self._calculate_atr()
            current_atr = atr.iloc[-1]
            avg_atr = atr.rolling(window=20).mean().iloc[-1]
            vol_ratio = current_atr / avg_atr
            
            if vol_ratio > 1.5:
                volatility_regime = 'high_volatility'
            elif vol_ratio < 0.7:
                volatility_regime = 'low_volatility'
            else:
                volatility_regime = 'normal_volatility'
            
            return {
                'support_levels': support_levels[:3],  # 상위 3개 지지선
                'resistance_levels': resistance_levels[:3],  # 상위 3개 저항선
                'trend': recent_trend,
                'volatility_regime': volatility_regime
            }
            
        except Exception as e:
            print(f"시장 구조 분석 오류: {str(e)}")
            return {}
        
    def _calculate_price_momentum(self) -> float:
        """가격 모멘텀 계산 (0-100)"""
        close = self.df['close']
        momentum = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100
        normalized = min(max(momentum, 0), 100)
        return normalized
        
    def _analyze_recent_trend(self) -> str:
        """
        최근 추세 분석
        Returns:
            str: 추세 방향 ('strong_uptrend', 'weak_uptrend', 'strong_downtrend', 'weak_downtrend')
        """
        close = self.df['close']
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        
        # 현재 가격과 이동평균선과의 관계
        current_price = close.iloc[-1]
        current_sma20 = sma20.iloc[-1]
        current_sma50 = sma50.iloc[-1]
        
        # 이동평균선의 기울기 계산
        sma20_slope = (current_sma20 - sma20.iloc[-5]) / 5
        sma50_slope = (current_sma50 - sma50.iloc[-5]) / 5
        
        # 추세 강도 판단
        if current_price > current_sma20 > current_sma50 and sma20_slope > 0 and sma50_slope > 0:
            return 'strong_uptrend'
        elif current_price < current_sma20 < current_sma50 and sma20_slope < 0 and sma50_slope < 0:
            return 'strong_downtrend'
        elif current_price > current_sma50 or (current_price > current_sma20 and sma20_slope > 0):
            return 'weak_uptrend'
        else:
            return 'weak_downtrend'
            
    def _get_rsi_signal(self, rsi: RSIIndicator) -> str:
        """RSI 신호 분석"""
        current_rsi = rsi.rsi().iloc[-1]
        
        if current_rsi > self.indicators_config['rsi']['overbought']:
            return 'sell'
        elif current_rsi < self.indicators_config['rsi']['oversold']:
            return 'buy'
        else:
            return 'neutral'
            
    def _get_macd_signal(self, macd: MACD) -> str:
        """MACD 신호 분석"""
        macd_line = macd.macd().iloc[-1]
        signal_line = macd.macd_signal().iloc[-1]
        
        if macd_line > signal_line:
            return 'buy'
        elif macd_line < signal_line:
            return 'sell'
        else:
            return 'neutral'
            
    def _get_bollinger_signal(self, bb: BollingerBands) -> str:
        """볼린저 밴드 신호 분석"""
        current_price = self.df['close'].iloc[-1]
        upper = bb.bollinger_hband().iloc[-1]
        lower = bb.bollinger_lband().iloc[-1]
        
        if current_price > upper:
            return 'sell'
        elif current_price < lower:
            return 'buy'
        else:
            return 'neutral'
            
    def _get_stochastic_signal(self, stoch: StochasticOscillator) -> str:
        """스토캐스틱 신호 분석"""
        k = stoch.stoch().iloc[-1]
        d = stoch.stoch_signal().iloc[-1]
        
        if k > 80 and d > 80:
            return 'sell'
        elif k < 20 and d < 20:
            return 'buy'
        else:
            return 'neutral'

    def _get_supertrend_signal(self, supertrend: pd.Series) -> str:
        """Super Trend 신호 분석"""
        current_price = self.df['close'].iloc[-1]
        current_supertrend = supertrend.iloc[-1]
        
        if current_price > current_supertrend:
            return 'buy'
        elif current_price < current_supertrend:
            return 'sell'
        else:
            return 'neutral'
            
    def _get_ichimoku_signal(self, tenkan: pd.Series, kijun: pd.Series, 
                            span_a: pd.Series, span_b: pd.Series, 
                            chikou: pd.Series) -> str:
        """일목균형표 신호 분석"""
        current_price = self.df['close'].iloc[-1]
        
        # 구름대 위치 확인
        above_cloud = current_price > span_a.iloc[-1] and current_price > span_b.iloc[-1]
        below_cloud = current_price < span_a.iloc[-1] and current_price < span_b.iloc[-1]
        
        # 전환선과 기준선 크로스 확인
        cross_above = tenkan.iloc[-1] > kijun.iloc[-1] and tenkan.iloc[-2] <= kijun.iloc[-2]
        cross_below = tenkan.iloc[-1] < kijun.iloc[-1] and tenkan.iloc[-2] >= kijun.iloc[-2]
        
        if above_cloud and cross_above:
            return 'buy'
        elif below_cloud and cross_below:
            return 'sell'
        else:
            return 'neutral'

    def _get_vwap_signal(self, vwap: VolumeWeightedAveragePrice) -> str:
        """VWAP 신호 분석"""
        current_price = self.df['close'].iloc[-1]
        current_vwap = vwap.volume_weighted_average_price().iloc[-1]
        
        if current_price > current_vwap * 1.02:  # 2% 이상 상회
            return 'sell'
        elif current_price < current_vwap * 0.98:  # 2% 이상 하회
            return 'buy'
        else:
            return 'neutral'
            
    def _get_divergence_signal(self, bullish_div: bool, bearish_div: bool) -> str:
        """RSI 다이버전스 신호 분석"""
        if bullish_div:
            return 'buy'
        elif bearish_div:
            return 'sell'
        else:
            return 'neutral'

    def _calculate_position_size(self, signal: str, confidence: float, market_state: Dict[str, Any]) -> float:
        """포지션 크기 계산"""
        # 기본 포지션 크기
        base_size = 1.0
        
        # 시장 상태에 따른 조정
        if market_state['volatility'] == 'high':
            base_size *= 0.5  # 높은 변동성에서는 포지션 크기 감소
        elif market_state['volatility'] == 'low':
            base_size *= 2.0  # 낮은 변동성에서는 포지션 크기 증가
        
        # 신뢰도에 따른 조정
        adjusted_size = base_size * (confidence / 100)
        
        return adjusted_size

    def _filter_signals(self, signals: Dict[str, str], market_state: MarketState) -> Dict[str, str]:
        """시장 상태를 고려한 신호 필터링"""
        return {
            signal: value 
            for signal, value in signals.items() 
            if value != 'neutral'
        }

    def _calculate_atr(self, period: int = 14) -> Series:
        """ATR 계산"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

class TechnicalIndicators:
    """
    기술적 분석을 수행하는 클래스
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # 데이터 타입 변환
        for col in ['open', 'high', 'low', 'close']:
            self.df[col] = self.df[col].astype(float)
            
    def calculate_adx(self, period: int = 14) -> float:
        """ADX 계산"""
        try:
            high = cast(Series, self.df['high'])
            low = cast(Series, self.df['low'])
            
            # +DM, -DM 계산
            up_move = high.diff()
            down_move = low.diff()
            
            # 조건에 따른 값 할당
            plus_dm = up_move.copy()
            minus_dm = down_move.copy()
            plus_dm = plus_dm.mask(plus_dm.lt(0), 0)  # < 연산자 대신 lt() 사용
            minus_dm = minus_dm.mask(minus_dm.gt(0), 0)  # > 연산자 대신 gt() 사용
            
            # TR 계산
            tr = self._calculate_atr(1)
            
            # Smoothed TR and DM
            tr_smooth = tr.ewm(span=period, adjust=False).mean()
            plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
            minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()
            
            # +DI, -DI 계산
            pdi = 100 * plus_dm_smooth / tr_smooth
            ndi = 100 * minus_dm_smooth / tr_smooth
            
            # ADX 계산
            dx = 100 * abs(pdi - ndi) / (pdi + ndi)
            adx = dx.ewm(span=period, adjust=False).mean()
            
            return float(adx.iloc[-1])
            
        except Exception as e:
            print(f"ADX 계산 오류: {str(e)}")
            return 0.0
            
    def calculate_rsi(self, period: int = 14) -> Series:
        """RSI 계산"""
        try:
            delta = self.df['close'].diff()
            
            # 상승/하락 구분
            gain = delta.copy()
            loss = delta.copy()
            gain = gain.mask(gain.lt(0), 0)
            loss = -loss.mask(loss.gt(0), 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            print(f"RSI 계산 오류: {str(e)}")
            return pd.Series(dtype=float)
            
    def calculate_macd(self, 
                      fast_period: int = 12,
                      slow_period: int = 26,
                      signal_period: int = 9) -> Dict[str, Series]:
        """MACD 계산"""
        try:
            # EMA 계산
            fast_ema = self.df['close'].ewm(span=fast_period, adjust=False).mean()
            slow_ema = self.df['close'].ewm(span=slow_period, adjust=False).mean()
            
            # MACD 라인
            macd_line = fast_ema - slow_ema
            
            # 시그널 라인
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            return {
                'macd': macd_line,
                'signal': signal_line
            }
            
        except Exception as e:
            print(f"MACD 계산 오류: {str(e)}")
            return {
                'macd': pd.Series(dtype=float),
                'signal': pd.Series(dtype=float)
            }
            
    def calculate_volume_profile(self, bins: int = 50) -> Tuple[float, Index]:
        """거래량 프로파일 분석"""
        try:
            # 가격 범위 설정
            price_range = pd.cut(self.df['close'], bins=bins)
            
            # 거래량 프로파일 계산
            volume_profile = self.df.groupby(price_range)['volume'].sum()
            
            # POC (Point of Control) 찾기
            poc_interval = cast(pd_interval.Interval, volume_profile.idxmax())
            poc = float(poc_interval.mid)
            
            # Value Area 계산 (70% 거래량)
            sorted_volumes = volume_profile.sort_values(ascending=False)
            cumsum_volumes = sorted_volumes.cumsum()
            value_area = sorted_volumes[cumsum_volumes.le(cumsum_volumes.sum() * 0.7)].index
            
            return poc, value_area
            
        except Exception as e:
            print(f"거래량 프로파일 계산 오류: {str(e)}")
            return 0.0, pd.Index([])

    def _calculate_atr(self, period: int = 14) -> Series:
        """ATR 계산"""
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
        
    def calculate_supertrend(self, period: int = 10, multiplier: float = 3.0) -> pd.Series:
        """
        Super Trend 지표 계산
        """
        hl2 = (self.df['high'] + self.df['low']) / 2
        atr = self._calculate_atr(period)
        
        # 상단 및 하단 밴드 계산
        upperband = hl2 + (multiplier * atr)
        lowerband = hl2 - (multiplier * atr)
        
        # Super Trend 계산
        supertrend = pd.Series(index=self.df.index, dtype=float)
        direction = pd.Series(index=self.df.index)
        
        for i in range(period, len(self.df)):
            if self.df['close'][i] > upperband[i-1]:
                direction[i] = 1
            elif self.df['close'][i] < lowerband[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                
            if direction[i] == 1:
                supertrend[i] = lowerband[i]
            else:
                supertrend[i] = upperband[i]
                
        return supertrend
        
    def calculate_ichimoku(self) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        일목균형표 계산
        Returns:
            Tuple[pd.Series]: (전환선, 기준선, 선행스팬A, 선행스팬B, 후행스팬)
        """
        high = self.df['high']
        low = self.df['low']
        
        # 전환선 (Conversion Line)
        tenkan_sen = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # 기준선 (Base Line)
        kijun_sen = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # 선행스팬 1 (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # 선행스팬 2 (Leading Span B)
        senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # 후행스팬 (Lagging Span)
        chikou_span = self.df['close'].shift(-26)
        
        return (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span)
        
    def calculate_rsi_divergence(self, period: int = 14) -> Tuple[bool, bool]:
        """RSI 다이버전스 감지"""
        rsi = self.calculate_rsi(period)
        
        # 최근 고점/저점 찾기
        price_highs = self.find_peaks(self.df['close'], 5)
        price_lows = self.find_peaks(-self.df['close'], 5)
        rsi_highs = self.find_peaks(rsi, 5)
        rsi_lows = self.find_peaks(-rsi, 5)
        
        # 강세/약세 다이버전스 확인
        bullish_div = bool(price_lows.iloc[-1] < price_lows.iloc[-2] and rsi_lows.iloc[-1] > rsi_lows.iloc[-2])
        bearish_div = bool(price_highs.iloc[-1] > price_highs.iloc[-2] and rsi_highs.iloc[-1] < rsi_highs.iloc[-2])
        
        return bullish_div, bearish_div

    def _calculate_bollinger_bands(self) -> Dict[str, Series]:
        """볼린저 밴드 계산"""
        try:
            bb = BollingerBands(close=self.df['close'].astype(float))
            return {
                'upper': bb.bollinger_hband(),
                'middle': bb.bollinger_mavg(),
                'lower': bb.bollinger_lband()
            }
        except Exception as e:
            print(f"볼린저 밴드 계산 오류: {str(e)}")
            return {
                'upper': pd.Series(dtype=float),
                'middle': pd.Series(dtype=float),
                'lower': pd.Series(dtype=float)
            }

    def find_peaks(self, series: Series, window: int = 20) -> Series:
        """피크 찾기"""
        try:
            series = series.astype(float)
            peaks = pd.Series(index=series.index, dtype=bool)
            
            for i in range(window, len(series) - window):
                if series.iloc[i] == max(series.iloc[i-window:i+window+1]):
                    peaks.iloc[i] = True
                    
            return peaks
            
        except Exception as e:
            print(f"피크 찾기 오류: {str(e)}")
            return pd.Series(index=series.index, dtype=bool) 