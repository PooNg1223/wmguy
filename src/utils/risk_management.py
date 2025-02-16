from typing import Dict, Any

class RiskManager:
    """
    리스크 관리를 담당하는 클래스
    거래 크기, 손실 한도 등을 관리합니다.
    """
    def __init__(self, config: Dict[str, Any]):
        self.max_position_size = config.get('max_position_size', 1000)
        self.max_daily_loss = config.get('max_daily_loss', 100)
        self.leverage = config.get('leverage', 1)
        
    def check_trade_risk(self, trade_params: Dict[str, Any]) -> bool:
        """
        거래 위험도 체크
        
        Args:
            trade_params: 거래 파라미터
            
        Returns:
            bool: 거래 허용 여부
        """
        # 기본적인 리스크 체크 로직
        position_size = trade_params.get('size', 0)
        if position_size > self.max_position_size:
            return False
        return True
        
    def calculate_position_size(self, capital: float, risk_per_trade: float) -> float:
        """
        적정 포지션 크기 계산
        
        Args:
            capital: 가용 자본
            risk_per_trade: 거래당 리스크 비율
            
        Returns:
            float: 적정 포지션 크기
        """
        return min(capital * risk_per_trade, self.max_position_size) 