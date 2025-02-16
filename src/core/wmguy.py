import logging
from typing import Optional

class WMGuy:
    def __init__(self):
        self.config = Config()
        self.memory = Memory()
        self.learning = Learning()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('wmguy')
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler('logs/wmguy.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger 