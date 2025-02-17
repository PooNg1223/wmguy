from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class MarketState:
    """시장 상태를 나타내는 데이터 클래스"""
    
    # 추세 관련
    trend_strength: float = 0.0  # 추세 강도 (0-100)
    trend_direction: str = 'ranging'  # 추세 방향 (uptrend/downtrend/ranging)
    
    # 변동성 관련
    volatility: str = 'medium'  # 변동성 수준 (low/medium/high)
    
    # 가격 레벨
    support_levels: List[float] = None  # 지지 레벨
    resistance_levels: List[float] = None  # 저항 레벨
    
    # 시장 레짐
    regime: str = 'ranging'  # 시장 레짐 (uptrend/downtrend/ranging)
    
    # 리스크 레벨
    risk_level: str = 'medium'  # 리스크 레벨 (low/medium/high)
    
    @property
    def market_regime(self) -> str:
        """시장 레짐 getter"""
        return self.regime
        
    def __post_init__(self):
        """초기화 후 처리"""
        if self.support_levels is None:
            self.support_levels = []
        if self.resistance_levels is None:
            self.resistance_levels = []
            
    @property
    def as_dict(self) -> Dict[str, Any]:
        return {
            'trend_strength': self.trend_strength,
            'trend_direction': self.trend_direction,
            'volatility': self.volatility,
            'regime': self.regime,
            'support_levels': self.support_levels,
            'resistance_levels': self.resistance_levels,
            'risk_level': self.risk_level
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketState':
        """딕셔너리에서 MarketState 객체 생성"""
        return cls(**data) 