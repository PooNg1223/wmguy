import logging
from pathlib import Path

def setup_logger(name: str) -> logging.Logger:
    """
    로깅 설정
    
    Args:
        name: 로거 이름
        
    Returns:
        logging.Logger: 설정된 로거
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 로그 디렉토리 생성
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 파일 핸들러 설정
    handler = logging.FileHandler(log_dir / f'{name}.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger 