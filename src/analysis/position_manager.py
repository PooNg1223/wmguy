from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.analysis.position_config import PositionConfig

@dataclass
class PositionConfig:
    risk_per_trade: float  # 트레이드당 리스크 비율
    max_position_size: float  # 최대 포지션 크기
    min_confidence: float  # 최소 신뢰도 요구사항
    profit_target_ratio: float  # 익절 비율 (손실 대비)
    trailing_stop: bool  # 트레일링 스탑 사용 여부

class PositionManager:
    """포지션 크기와 리스크를 관리하는 클래스"""
    
    def __init__(self, config: PositionConfig):
        """
        Args:
            config: 포지션 관리 설정
        """
        self.config = config
        
    def calculate_position_size(self, 
                              capital: float,
                              entry_price: float,
                              stop_loss: Optional[float],
                              confidence: float,
                              market_state: Dict[str, Any]) -> Dict[str, float]:
        """
        포지션 크기 계산
        
        Args:
            capital: 계좌 자본금
            entry_price: 진입 가격
            stop_loss: 스탑로스 가격 (None인 경우 포지션 크기 0)
            confidence: 신호 신뢰도 (0-100)
            market_state: 시장 상태 정보
            
        Returns:
            Dict[str, float]: 포지션 정보
        """
        # 스탑로스가 없거나 신뢰도가 최소 요구사항보다 낮으면 포지션 크기 0
        if stop_loss is None or confidence < self.config.min_confidence:
            return self._create_empty_position()
            
        # 리스크 계산
        risk_ratio = self.config.get_adjusted_risk(market_state)
        risk_amount = capital * risk_ratio
        
        # 스탑로스 거리
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return self._create_empty_position()
            
        # 포지션 크기 계산
        position_size = risk_amount / stop_distance
        
        # 최대 포지션 크기 제한
        max_size = capital * self.config.max_position_size / entry_price
        position_size = min(position_size, max_size)
        
        # 익절 가격 계산
        reward_distance = stop_distance * self.config.profit_target_ratio
        take_profit = (
            entry_price + reward_distance if entry_price > stop_loss
            else entry_price - reward_distance
        )
        
        # 예상 수익 계산
        reward_amount = position_size * reward_distance
        
        return {
            'size': position_size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount
        }
        
    def _calculate_take_profit(self, 
                             entry: float, 
                             stop_loss: float, 
                             ratio: float) -> float:
        """
        이익실현 가격 계산
        
        Args:
            entry: 진입 가격
            stop_loss: 스탑로스 가격
            ratio: 이익실현 비율 (리스크 대비)
            
        Returns:
            float: 이익실현 가격
        """
        try:
            risk = abs(entry - stop_loss)
            if risk == 0 or entry == 0:
                return 0.00
                
            take_profit = entry + (risk * ratio)
            return round(take_profit, 2)
            
        except Exception as e:
            print(f"이익실현 가격 계산 오류: {str(e)}")
            return 0.00
        
    def _create_empty_position(self) -> Dict[str, float]:
        """빈 포지션 정보 생성"""
        return {
            'size': 0.0,
            'entry_price': 0.0,
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'risk_amount': 0.0,
            'reward_amount': 0.0
        } 