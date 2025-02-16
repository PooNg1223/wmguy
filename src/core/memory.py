import json
from pathlib import Path
from typing import Dict, Any
import logging

class Memory:
    """메모리 관리 클래스"""
    def __init__(self):
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        self.memory_file = Path("data/wmguy_memory.json")
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
    def save_memory(self, data: Dict[str, Any]) -> None:
        try:
            with self.memory_file.open('w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to save memory: {e}")
            
    def load_memory(self) -> Dict[str, Any]:
        if not self.memory_file.exists():
            return {}
        try:
            with self.memory_file.open('r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load memory: {e}")
            return {}

    def save_daily_data(self, data: Dict[str, Any]):
        """일일 데이터 저장"""
        pass 