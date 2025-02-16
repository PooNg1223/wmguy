from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path

class ConfigManager:
    """설정 관리자"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            if not os.path.exists(self.config_path):
                return {}
                
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정값 조회"""
        try:
            # 점으로 구분된 키를 처리 (예: 'trading.api_key')
            current = self.config
            for part in key.split('.'):
                current = current.get(part, {})
            return current if current != {} else default
        except Exception:
            return default
    
    def get_all(self) -> Dict[str, Any]:
        """전체 설정 반환"""
        return self.config
    
    def update(self, new_config: Dict[str, Any]) -> None:
        """설정 업데이트 및 저장"""
        try:
            self.config.update(new_config)
            
            # 설정 파일 저장
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                
        except Exception as e:
            raise RuntimeError(f"Failed to update config: {e}") 