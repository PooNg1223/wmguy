from typing import Dict, Any, Optional
import pandas as pd
from src.analysis.market_analyzer import MarketAnalyzer
from src.analysis.signal_generator import SignalGenerator
from src.trading.position_manager import PositionManager

class TradingModel:
    """트레이딩 모델 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        트레이딩 모델 초기화
        
        Args:
            config: 트레이딩 설정
        """
        self.config = config
        self.data = {}  # 심볼별 데이터 캐시
        self.capital = config['trading']['initial_capital']
        
    def analyze_timeframe(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """특정 타임프레임 분석"""
        try:
            # 데이터 가져오기
            if symbol not in self.data or timeframe not in self.data[symbol]:
                print(f"데이터 없음: {symbol} {timeframe}")
                return {}
                
            df = self.data[symbol][timeframe]
            
            # 분석기 초기화
            analyzer = MarketAnalyzer(df)
            signal_generator = SignalGenerator(df)
            position_manager = PositionManager(self.config)
            
            # 시장 상태 분석
            market_state = analyzer.analyze_market_state()
            
            # 매매 신호 생성
            signal = signal_generator.generate_signal(market_state)
            
            # 포지션 크기 계산
            position = position_manager.calculate_position_size(
                capital=self.capital,
                entry_price=signal.get('entry_price', 0),
                stop_loss=signal.get('stop_loss', 0),
                confidence=signal.get('confidence', 0),
                market_state=market_state.__dict__
            )
            
            return {
                'market_state': market_state.__dict__,
                'signal': signal,
                'position': position
            }
            
        except Exception as e:
            print(f"분석 오류: {str(e)}")
            return {}
            
    def _manage_position(self, 
                        market_state: Dict[str, Any],
                        signal: Dict[str, Any],
                        capital: float) -> Dict[str, Any]:
        """포지션 관리"""
        try:
            position = self.position_manager.calculate_position_size(
                capital=capital,
                entry_price=signal.get('entry_price', 0),
                stop_loss=signal.get('stop_loss', 0),
                confidence=signal.get('confidence', 0),
                market_state=market_state
            )
            
            return position
            
        except Exception as e:
            print(f"포지션 관리 오류: {str(e)}")
            return {}

    def _analyze_timeframe(self, df: pd.DataFrame) -> Dict:
        """특정 타임프레임 분석"""
        try:
            # 데이터 길이 체크
            if len(df) < 50:
                print("데이터가 충분하지 않습니다")
                return self._create_empty_analysis()
                
            # 시장 분석
            analyzer = MarketAnalyzer(df)
            market_state = analyzer.analyze_market_state()
            
            # 매매 신호 생성
            signal = self.signal_generator.generate_signal(df, market_state)
            
            # 포지션 계산
            position = self.position_manager.calculate_position_size(
                capital=self.capital,
                entry_price=signal.get('entry_price', 0),
                stop_loss=signal.get('stop_loss', 0),
                confidence=signal.get('confidence', 0),
                market_state=market_state.__dict__
            )
            
            return {
                'market_state': {
                    'regime': market_state.regime,
                    'volatility': market_state.volatility,
                    'trend_strength': market_state.trend_strength,
                    'risk_level': market_state.risk_level,
                    'support_levels': market_state.support_levels,
                    'resistance_levels': market_state.resistance_levels
                },
                'signal': signal,
                'position': position
            }
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return self._create_empty_analysis()
            
    def _create_empty_analysis(self) -> Dict:
        """빈 분석 결과 생성"""
        return {
            'market_state': {
                'regime': 'ranging',
                'volatility': 'medium',
                'trend_strength': 0.0,
                'risk_level': 'low',
                'support_levels': [],
                'resistance_levels': []
            },
            'signal': {
                'type': 'none',
                'confidence': 0.0,
                'entry_price': 0.0,
                'stop_loss': 0.0
            },
            'position': {
                'size': 0.0000,
                'entry_price': 0.00,
                'stop_loss': 0.00,
                'take_profit': 0.00,
                'type': 'none'
            }
        } 