from typing import Dict, Any, List
import yaml
import os
from dataclasses import dataclass
from src.analysis.position_config import PositionConfig

class ConfigLoader:
    """
    YAML 설정 파일을 로드하고 관리하는 클래스
    """
    
    def __init__(self, config_path: str = "config/trading_config.yaml"):
        """
        Args:
            config_path: YAML 설정 파일 경로
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        YAML 설정 파일 로드
        
        Returns:
            Dict[str, Any]: 설정 데이터
        """
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            return config
            
        except Exception as e:
            print(f"설정 파일 로드 오류: {str(e)}")
            return {}
            
    def get_position_config(self) -> PositionConfig:
        """
        포지션 관리 설정 로드
        
        Returns:
            PositionConfig: 포지션 관리 설정
        """
        try:
            position_config = self.config.get('position', {})
            
            return PositionConfig(
                risk_per_trade=position_config.get('risk_per_trade', 0.02),
                max_position_size=position_config.get('max_position_size', 0.3),
                min_confidence=position_config.get('min_confidence', 60.0),
                profit_target_ratio=position_config.get('profit_target_ratio', 2.0),
                trailing_stop=position_config.get('trailing_stop', True),
                market_regime_multipliers=position_config.get('market_regime_multipliers'),
                volatility_multipliers=position_config.get('volatility_multipliers'),
                risk_level_multipliers=position_config.get('risk_level_multipliers')
            )
            
        except Exception as e:
            print(f"포지션 설정 로드 오류: {str(e)}")
            return PositionConfig()
            
    def get_indicator_config(self) -> Dict[str, Any]:
        """
        기술적 지표 설정 로드
        
        Returns:
            Dict[str, Any]: 지표 설정
        """
        return self.config.get('indicators', {})
        
    def get_backtest_config(self) -> Dict[str, Any]:
        """
        백테스트 설정 로드
        
        Returns:
            Dict[str, Any]: 백테스트 설정
        """
        return self.config.get('backtest', {})

    def get_trading_config(self) -> Dict[str, Any]:
        """
        거래 설정 로드
        
        Returns:
            Dict[str, Any]: 거래 설정
            {
                'symbols': List[str],       # 거래 심볼 목록
                'timeframes': List[str],    # 타임프레임 목록
                'initial_capital': float,   # 초기 자본금
                'max_positions': int        # 최대 동시 포지션 수
            }
        """
        try:
            trading_config = self.config.get('trading', {})
            
            # 필수 설정 검증
            required_fields = ['symbols', 'timeframes', 'initial_capital', 'max_positions']
            for field in required_fields:
                if field not in trading_config:
                    raise ValueError(f"필수 거래 설정이 누락되었습니다: {field}")
                    
            # 타임프레임 형식 검증
            for tf in trading_config['timeframes']:
                if not str(tf).isdigit():
                    raise ValueError(f"잘못된 타임프레임 형식: {tf}")
                    
            # 초기 자본금 검증
            if trading_config['initial_capital'] <= 0:
                raise ValueError("초기 자본금은 0보다 커야 합니다")
                
            # 최대 포지션 수 검증
            if trading_config['max_positions'] <= 0:
                raise ValueError("최대 포지션 수는 0보다 커야 합니다")
                
            return trading_config
            
        except Exception as e:
            print(f"거래 설정 로드 오류: {str(e)}")
            return {
                'symbols': ['BTCUSDT'],
                'timeframes': ['15', '60', '240'],
                'initial_capital': 10000,
                'max_positions': 3
            }
            
    def get_market_analysis_config(self) -> Dict[str, Any]:
        """
        시장 분석 설정 로드
        
        Returns:
            Dict[str, Any]: 시장 분석 설정
        """
        try:
            market_config = self.config.get('market_analysis', {})
            
            # 기본값 설정
            defaults = {
                'trend_threshold': 25,
                'volatility_multiplier': 1.5,
                'support_resistance_lookback': 20
            }
            
            # 누락된 설정은 기본값으로 대체
            for key, value in defaults.items():
                if key not in market_config:
                    market_config[key] = value
                    
            return market_config
            
        except Exception as e:
            print(f"시장 분석 설정 로드 오류: {str(e)}")
            return {
                'trend_threshold': 25,
                'volatility_multiplier': 1.5,
                'support_resistance_lookback': 20
            }

    def get_api_config(self) -> Dict[str, Any]:
        """
        API 설정 로드
        
        Returns:
            Dict[str, Any]: API 설정
            {
                'testnet': bool,        # 테스트넷 사용 여부
                'market_type': str,     # 마켓 타입 (linear/inverse)
                'leverage': int,        # 레버리지 설정
                'api_key': str,         # API 키 (환경변수에서 로드)
                'api_secret': str       # API 시크릿 (환경변수에서 로드)
            }
        """
        try:
            api_config = self.config.get('bybit', {})
            
            # 필수 설정 검증
            required_fields = ['testnet', 'market_type', 'leverage']
            for field in required_fields:
                if field not in api_config:
                    raise ValueError(f"필수 API 설정이 누락되었습니다: {field}")
                    
            # 마켓 타입 검증
            if api_config['market_type'] not in ['linear', 'inverse']:
                raise ValueError(f"잘못된 마켓 타입: {api_config['market_type']}")
                
            # 레버리지 검증
            if not isinstance(api_config['leverage'], int) or api_config['leverage'] <= 0:
                raise ValueError("레버리지는 양의 정수여야 합니다")
                
            # 환경변수에서 API 키 로드
            api_config['api_key'] = os.getenv('BYBIT_API_KEY', '')
            api_config['api_secret'] = os.getenv('BYBIT_API_SECRET', '')
            
            # API 키 검증
            if not api_config['api_key'] or not api_config['api_secret']:
                print("경고: API 키가 설정되지 않았습니다")
                
            return api_config
            
        except Exception as e:
            print(f"API 설정 로드 오류: {str(e)}")
            return {
                'testnet': True,
                'market_type': 'linear',
                'leverage': 1,
                'api_key': '',
                'api_secret': ''
            } 