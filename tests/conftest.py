import pytest

def pytest_collection_modifyitems(items):
    """Modify test items in place to ensure test classes are marked"""
    for item in items:
        # Trading tests
        if any(name in item.nodeid for name in [
            "test_module_init",
            "test_add_trade",
            "test_daily_summary",
            "test_position_status",
            "test_trading_execution",
            "test_risk_management"
        ]):
            item.add_marker(pytest.mark.trading)
        
        # Strategy tests
        elif any(name in item.nodeid for name in [
            "test_strategy_analysis",
            "test_strategy_learning"
        ]):
            item.add_marker(pytest.mark.strategy)
        
        # Integration tests
        elif "test_price_stream" in item.nodeid:
            item.add_marker(pytest.mark.integration)

def pytest_configure(config):
    """Register markers to avoid warnings"""
    config.addinivalue_line("markers", "trading: trading module tests")
    config.addinivalue_line("markers", "strategy: strategy module tests")
    config.addinivalue_line("markers", "integration: integration tests") 