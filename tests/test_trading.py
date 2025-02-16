import sys
from pathlib import Path
import pytest
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.modules.trading_module import TradingModule

@pytest.fixture
def trading_config():
    """테스트용 트레이딩 설정"""
    return {
        'risk_level': 'low',
        'max_position_size': 500
    }

@pytest.fixture
def sample_trade():
    """테스트용 거래 정보"""
    return {
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'price': 50000.0,
        'size': 0.1,
        'memo': '테스트 매수'
    }

def test_trading_module_init(trading_config):
    """트레이딩 모듈 초기화 테스트"""
    module = TradingModule(trading_config)
    assert module.config == trading_config
    assert module.positions == {}
    assert module.trades_history == []

def test_add_trade(trading_config, sample_trade):
    """거래 추가 테스트"""
    module = TradingModule(trading_config)
    result = module.add_trade(sample_trade)
    assert result is True
    assert len(module.trades_history) == 1
    assert module.trades_history[0]['trade_info'] == sample_trade

def test_daily_summary(trading_config, sample_trade):
    """일일 거래 요약 테스트"""
    module = TradingModule(trading_config)
    module.add_trade(sample_trade)
    summary = module.get_daily_summary()
    assert summary['total_trades'] == 1
    assert len(summary['trades']) == 1
    assert summary['trades'][0]['trade_info'] == sample_trade
