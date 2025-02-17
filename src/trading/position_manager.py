from typing import Dict, Any

class PositionManager:
    """포지션 관리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        포지션 관리자 초기화
        
        Args:
            config: 트레이딩 설정
        """
        self.config = config
        self.risk_per_trade = config['trading']['risk_per_trade']
        self.max_position_size = config['trading']['max_position_size']
        self.min_confidence = config['trading']['min_confidence']
        
    def calculate_position_size(self,
                              capital: float,
                              entry_price: float,
                              stop_loss: float,
                              confidence: float,
                              market_state: Dict[str, Any]) -> Dict[str, Any]:
        """포지션 크기 계산"""
        try:
            # 입력값 검증
            if not all([capital, entry_price, stop_loss]):
                print("필수 입력값이 누락되었습니다")
                return self._create_empty_position()
                
            # 0 또는 음수 값 체크
            if any(x <= 0 for x in [capital, entry_price, stop_loss]):
                print("입력값에 0 또는 음수가 포함되어 있습니다")
                return self._create_empty_position()
                
            # 손실 비율 계산
            loss_ratio = abs(entry_price - stop_loss) / entry_price
            
            # 손실 비율 제한 (최대 5%)
            if loss_ratio > 0.05:
                loss_ratio = 0.05
                # 스탑로스 재계산
                if entry_price > stop_loss:  # 롱 포지션
                    stop_loss = entry_price * 0.95
                else:  # 숏 포지션
                    stop_loss = entry_price * 1.05
                    
            # 신뢰도 체크
            if confidence < self.min_confidence:
                print(f"신뢰도가 너무 낮습니다: {confidence} < {self.min_confidence}")
                return self._create_empty_position()
                
            # 리스크 금액 계산 (신뢰도에 따라 조정)
            confidence_multiplier = confidence / 100  # 0.6 ~ 1.0
            risk_amount = capital * self.risk_per_trade * confidence_multiplier
            
            # 포지션 크기 계산
            position_size = risk_amount / (entry_price * loss_ratio)
            
            # 최대 포지션 크기 제한
            max_size = capital * self.max_position_size / entry_price
            position_size = min(position_size, max_size)
            
            # 변동성에 따른 이익실현 비율 조정
            volatility = market_state.get('volatility', 'medium')
            profit_ratio = {
                'low': 1.5,    # 낮은 변동성: 1:1.5
                'medium': 2.0,  # 중간 변동성: 1:2
                'high': 2.5     # 높은 변동성: 1:2.5
            }.get(volatility, 2.0)
            
            # 이익실현 가격 계산
            take_profit = entry_price + (entry_price - stop_loss) * profit_ratio
            
            # 포지션 타입 결정
            position_type = 'long' if entry_price > stop_loss else 'short'
            
            return {
                'size': round(position_size, 4),
                'entry_price': round(entry_price, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit': round(take_profit, 2),
                'type': position_type,
                'risk_ratio': f"1:{profit_ratio}"  # 리스크:리워드 비율 추가
            }
            
        except Exception as e:
            print(f"포지션 계산 오류: {str(e)}")
            return self._create_empty_position()
            
    def _create_empty_position(self) -> Dict[str, Any]:
        """빈 포지션 생성"""
        return {
            'size': 0.0000,
            'entry_price': 0.00,
            'stop_loss': 0.00,
            'take_profit': 0.00,
            'type': 'none'
        } 