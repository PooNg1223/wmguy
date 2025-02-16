import logging
from typing import Optional, Dict, Any
from pathlib import Path

class WMGuy:
    def __init__(self):
        self.logger = self._setup_logger()
        self.logger.info("Initializing WMGuy")
        try:
            self.config = self._load_config()
            self.memory = Memory()
            self.learning = Learning()
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('wmguy')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
        handler = logging.FileHandler('logs/wmguy.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            config_path = Path('config/config.yaml')
            if not config_path.exists():
                self.logger.warning("Config file not found, using defaults")
                return {}
            # Add config loading logic here
            return {}
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {} 