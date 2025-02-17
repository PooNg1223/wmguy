from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class PositionConfig:
    """
    포지션 관리 설정
    
    Attributes:
        risk_per_trade: 트레이드당 리스크 비율 (0-1)
        max_position_size: 최대 포지션 크기 비율 (0-1)
        min_confidence: 최소 신뢰도 요구사항 (0-100)
        profit_target_ratio: 익절 비율 (손실 대비)
        trailing_stop: 트레일링 스탑 사용 여부
        
        # 시장 상태별 리스크 조정 설정
        market_regime_multipliers: Dict[str, float]  # 시장 레짐별 리스크 승수
        volatility_multipliers: Dict[str, float]     # 변동성별 리스크 승수
        risk_level_multipliers: Dict[str, float]     # 리스크 레벨별 승수
    """
    
    # 기본 설정
    risk_per_trade: float = 0.02
    max_position_size: float = 0.3
    min_confidence: float = 60.0
    profit_target_ratio: float = 2.0
    trailing_stop: bool = True
    
    # 시장 상태별 리스크 조정 설정
    market_regime_multipliers: Optional[Dict[str, float]] = None
    volatility_multipliers: Optional[Dict[str, float]] = None
    risk_level_multipliers: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """기본값 초기화"""
        # 시장 레짐별 리스크 승수
        if self.market_regime_multipliers is None:
            self.market_regime_multipliers = {
                'uptrend': 1.2,    # 상승 추세: 리스크 증가
                'downtrend': 1.0,  # 하락 추세: 기본 리스크
                'ranging': 0.8     # 횡보: 리스크 감소
            }
            
        # 변동성별 리스크 승수
        if self.volatility_multipliers is None:
            self.volatility_multipliers = {
                'low': 1.2,     # 낮은 변동성: 리스크 증가
                'medium': 1.0,  # 중간 변동성: 기본 리스크
                'high': 0.8     # 높은 변동성: 리스크 감소
            }
            
        # 리스크 레벨별 승수
        if self.risk_level_multipliers is None:
            self.risk_level_multipliers = {
                'low': 1.2,     # 낮은 리스크: 포지션 증가
                'medium': 1.0,  # 중간 리스크: 기본값
                'high': 0.8     # 높은 리스크: 포지션 감소
            }
            
        # 설정값 검증
        self._validate_config()
        
    def _validate_config(self):
        """설정값 유효성 검사"""
        if not 0 < self.risk_per_trade <= 0.1:
            raise ValueError("risk_per_trade는 0-0.1 사이여야 합니다")
            
        if not 0 < self.max_position_size <= 1:
            raise ValueError("max_position_size는 0-1 사이여야 합니다")
            
        if not 0 <= self.min_confidence <= 100:
            raise ValueError("min_confidence는 0-100 사이여야 합니다")
            
        if self.profit_target_ratio <= 0:
            raise ValueError("profit_target_ratio는 0보다 커야 합니다")
        
    def get_adjusted_risk(self, market_state: Dict[str, str]) -> float:
        """시장 상태에 따른 조정된 리스크 계산"""
        base_risk = self.risk_per_trade
        
        # 시장 레짐에 따른 조정
        regime_mult = self.market_regime_multipliers.get(
            market_state.get('market_regime', 'ranging'), 1.0
        )
        
        # 변동성에 따른 조정
        vol_mult = self.volatility_multipliers.get(
            market_state.get('volatility', 'medium'), 1.0
        )
        
        # 리스크 레벨에 따른 조정
        risk_mult = self.risk_level_multipliers.get(
            market_state.get('risk_level', 'medium'), 1.0
        )
        
        # 최종 리스크 계산
        adjusted_risk = base_risk * regime_mult * vol_mult * risk_mult
        
        # 리스크 범위 제한 (0.5% - 5%)
        return min(max(adjusted_risk, 0.005), 0.05) 