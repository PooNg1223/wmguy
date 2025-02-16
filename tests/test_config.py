import sys
from pathlib import Path
import pytest

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.core.wmguy import WMGuy

def test_config_loading():
    """설정 파일 로드 테스트"""
    wmguy = WMGuy()
    assert wmguy.config is not None
    assert 'version' in wmguy.config
    assert wmguy.config['version'] == '0.1'

def test_config_update():
    """설정 업데이트 테스트"""
    wmguy = WMGuy()
    new_config = {
        'trading': {
            'risk_level': 'low',
            'max_position_size': 500
        }
    }
    assert wmguy.update_config(new_config) is True
    assert wmguy.config['trading']['risk_level'] == 'low'
    assert wmguy.config['trading']['max_position_size'] == 500 