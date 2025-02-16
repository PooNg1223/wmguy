import sys
from pathlib import Path
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.modules.trading_module import TradingModule
from src.modules.trading_strategy import TradingStrategy
from src.modules.bybit_client import BybitClient

@pytest.fixture
def trading_config():
    """테스트용 트레이딩 설정"""
    return {
        'exchange': {
            'name': 'bybit',
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'testnet': True
        },
        'strategy': {
            'position_sizing': {
                'base_size': 0.1,
                'max_size': 1.0,
                'volume_factor': 0.5
            }
        },
        'risk': {
            'max_position_size': 1000,
            'risk_level': 'moderate',
            'stop_loss': 0.02,
            'take_profit': 0.06
        }
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

@pytest.fixture
def market_data():
    """테스트용 시장 데이터"""
    return {
        'symbol': 'BTCUSDT',
        'price': 50000.0,
        'tradingview_signals': [
            {'action': 'BUY', 'price': 50000.0}
        ],
        'technical_indicators': {
            'close_prices': [49000.0, 49500.0, 50000.0]
        },
        'market_data': {
            'volumes': [100, 150, 200],
            'prices': [49000.0, 49500.0, 50000.0]
        }
    }

@pytest.fixture
def bybit_client():
    """테스트용 바이비트 클라이언트"""
    mock_client = Mock()
    mock_client.get_current_price = Mock(return_value=50000.0)
    mock_client.get_kline = Mock(return_value={
        'result': {
            'list': [
                ['1234567890000', '50000', '51000', '49000', '50500', '100', '5000000'],
                ['1234567891000', '50500', '52000', '50000', '51000', '150', '7500000']
            ]
        }
    })
    mock_client.get_position = Mock(return_value=None)
    mock_client.place_order = Mock(return_value={'result': {'orderId': '123'}})
    mock_client.get_positions = Mock(return_value={
        'result': [{
            'position_value': 1100.0,
            'symbol': 'BTCUSDT',
            'side': 'Buy',
            'size': 1.1,
            'leverage': 10,
            'position_idx': 0,
            'position_margin': 110.0,
            'unrealised_pnl': 0.0,
            'entry_price': 50000.0,
            'liq_price': 45000.0,
            'is_isolated': True,
            'auto_add_margin': 0,
            'position_status': 'Active',
            'risk_id': 1,
            'available_balance': 1000.0,
            'used_margin': 110.0,
            'order_margin': 0.0,
            'position_balance': 1100.0
        }]
    })
    
    return mock_client

@pytest.mark.trading
class TestTrading:
    """트레이딩 모듈 기본 기능 테스트"""
    
    def test_module_init(self, trading_config, bybit_client):
        """모듈 초기화"""
        module = TradingModule(trading_config, bybit_client)
        assert module.config == trading_config
        assert module.positions == {}
        assert module.trades_history == []

    def test_add_trade(self, trading_config, bybit_client, sample_trade):
        """거래 추가"""
        module = TradingModule(trading_config, bybit_client)
        result = module.add_trade(sample_trade)
        assert result is True
        assert len(module.trades_history) == 1

    def test_daily_summary(self, trading_config, bybit_client, sample_trade):
        """일일 거래 요약"""
        module = TradingModule(trading_config, bybit_client)
        module.add_trade(sample_trade)
        summary = module.get_daily_summary()
        assert summary['total_trades'] == 1

    def test_position_status(self, trading_config, bybit_client, sample_trade):
        """포지션 상태"""
        module = TradingModule(trading_config, bybit_client)
        module.add_trade(sample_trade)
        status = module.get_position_status('BTCUSDT')
        assert status['symbol'] == 'BTCUSDT'

    @patch('src.modules.bybit_client.HTTP')
    def test_trading_execution(self, mock_http, trading_config, bybit_client):
        """매매 실행"""
        mock_client = Mock()
        mock_client.place_order.return_value = {'result': {'orderId': 'test123'}}
        mock_http.return_value = mock_client
        
        module = TradingModule(trading_config, bybit_client)
        result = module.execute_signal('BTCUSDT', {
            'signal': 'BUY',
            'strength': 0.8,
            'whale_activity': {
                'is_active': False,
                'direction': 'NEUTRAL',
                'activity_score': 0.0,
                'liquidity_score': 0.0,
                'consecutive_pattern': 0
            }
        })
        assert result is True

    @patch('src.modules.bybit_client.HTTP')
    def test_risk_management(self, mock_http, trading_config, bybit_client):
        """리스크 관리"""
        # trading_config의 risk 설정 수정
        trading_config.update({
            'risk': {
                'max_position_size': 900.0,
                'risk_level': 'conservative',
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'max_leverage': 10,
                'min_volume': 0.01,
                'max_drawdown': 0.3
            }
        })
        
        # bybit_client 모킹
        position_data = {
            'result': [{
                'position_value': 1100.0,
                'symbol': 'BTCUSDT',
                'side': 'Buy',
                'size': 1.1,
                'leverage': 10,
                'position_idx': 0,
                'position_margin': 110.0,
                'unrealised_pnl': 0.0,
                'entry_price': 50000.0,
                'liq_price': 45000.0,
                'is_isolated': True,
                'auto_add_margin': 0,
                'position_status': 'Active',
                'risk_id': 1,
                'available_balance': 1000.0,
                'used_margin': 110.0,
                'order_margin': 0.0,
                'position_balance': 1100.0
            }]
        }
        bybit_client.get_positions = Mock(return_value=position_data)
        mock_client = Mock()
        mock_client.get_positions.return_value = position_data
        mock_http.return_value = mock_client
        
        module = TradingModule(trading_config, bybit_client)
        result = module.execute_signal('BTCUSDT', {
            'signal': 'BUY',
            'strength': 1.0,
            'confidence': 0.9,
            'price': 50000.0,
            'timestamp': datetime.now().timestamp(),
            'risk_score': 0.8,
            'position_size': 0.1,
            'leverage': 10,
            'order_type': 'MARKET',
            'stop_loss': 0.05,
            'take_profit': 0.15,
            'margin_type': 'isolated',
            'risk_level': 'conservative',
            'current_position': {
                'size': 1.1,
                'value': 1100.0,
                'margin': 110.0
            },
            'market_conditions': {
                'volatility': 0.2,
                'trend': 'bullish',
                'liquidity': 0.8,
                'risk_level': 'high'
            },
            'technical_indicators': {
                'rsi': 65.0,
                'macd': {'value': 0.5, 'signal': 0.3, 'hist': 0.2},
                'ema': {'short': 50100.0, 'long': 49900.0}
            },
            'risk_metrics': {
                'position_risk': 0.8,
                'market_risk': 0.6,
                'exposure_ratio': 0.9,
                'margin_ratio': 0.7
            },
            'whale_activity': {
                'is_active': False,
                'direction': 'NEUTRAL',
                'activity_score': 0.0,
                'liquidity_score': 0.0,
                'consecutive_pattern': 0,
                'volume_ratio': 1.2
            }
        })
        assert result is False

@pytest.mark.trading
class TestStrategy:
    """트레이딩 전략 테스트"""
    
    @patch('src.modules.bybit_client.HTTP')
    def test_strategy_analysis(self, mock_http, trading_config, market_data, bybit_client):
        """전략 분석"""
        strategy = TradingStrategy()
        # 시장 데이터에 필요한 필드 추가
        market_data['technical_indicators'].update({
            'rsi': 65.0,
            'macd': {
                'value': 0.5,
                'signal': 0.3,
                'hist': 0.2,
                'trend': 'bullish',
                'crossover': True,
                'divergence': None,
                'strength': 0.7
            },
            'trend_strength': 0.7,
            'bollinger_bands': {'upper': 51000.0, 'middle': 50000.0, 'lower': 49000.0},
            'ema_short': 50100.0,
            'ema_long': 49900.0,
            'volume_ma': 150.0,
            'stochastic': {'k': 70.0, 'd': 65.0},
            'adx': 25.0,
            'atr': 100.0,
            'momentum': 0.6,
            'price_action': {'trend': 'bullish', 'strength': 0.7},
            'volume_profile': {'value': 150.0, 'trend': 'increasing'}
        })
        market_data['market_sentiment'] = 0.6
        signal = strategy.analyze(market_data)
        assert isinstance(signal, dict)
        assert signal['signal'] in ['BUY', 'SELL', 'NEUTRAL']
        assert 0 <= signal['strength'] <= 1.0

    @patch('src.modules.bybit_client.HTTP')
    def test_strategy_learning(self, mock_http, trading_config, market_data, bybit_client):
        """전략 학습"""
        strategy = TradingStrategy()
        strategy.weights = {
            'momentum': 0.2,
            'technical': 0.5,
            'tradingview': 0.3
        }
        
        trades = [
            {
                'timestamp': datetime.now(),
                'signal': 'BUY',
                'strength': 0.8,
                'pnl': 100.0,
                'success': True,
                'weight_update': True,
                'learning_rate': 0.1,
                'trade_id': 'test_trade_1',
                'strategy_version': '1.0',
                'indicators': {
                    'momentum': 0.6,
                    'technical': 0.7,
                    'tradingview': 0.8
                }
            },
            {
                'timestamp': datetime.now(),
                'signal': 'SELL',
                'strength': 0.9,
                'pnl': -50.0,
                'success': True,
                'weight_update': True,
                'learning_rate': 0.1,
                'trade_id': 'test_trade_2',
                'strategy_version': '1.0',
                'indicators': {
                    'momentum': 0.7,
                    'technical': 0.6,
                    'tradingview': 0.9
                }
            }
        ]
        strategy.trades_history = trades
        original_weights = strategy.weights.copy()
        strategy.learn()
        assert strategy.weights != original_weights

@pytest.mark.trading
class TestMarketAnalysis:
    """시장 분석 테스트"""
    
    def test_market_analysis(self, trading_config, market_data, bybit_client):
        """시장 분석"""
        module = TradingModule(trading_config, bybit_client)
        
        # 분석 실행
        analysis = module.analyze_market('BTCUSDT')
        
        # 결과 검증
        assert isinstance(analysis, dict)
        assert 'signal' in analysis
        assert 'strength' in analysis
        assert 'confidence' in analysis
        assert analysis['signal'] in ['BUY', 'SELL', 'NEUTRAL']
        assert 0 <= analysis['strength'] <= 1.0
        assert 0 <= analysis['confidence'] <= 1.0
    
    def test_backtest_performance(self, trading_config, market_data, bybit_client):
        """백테스트 성능"""
        module = TradingModule(trading_config, bybit_client)
        
        # 백테스트 실행
        signal = {
            'signal': 'BUY',
            'strength': 0.8,
            'confidence': 0.9
        }
        
        result = module.backtest_signal('BTCUSDT', signal)
        
        # 결과 검증
        assert isinstance(result, dict)
        assert 'pnl' in result
        assert 'success' in result
        
        # 성과 지표 확인
        metrics = module.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'win_rate' in metrics
        assert 'avg_pnl' in metrics 

@pytest.mark.performance
class TestPerformanceMetrics:
    """성과 지표 테스트"""
    
    @pytest.fixture
    def sample_backtest_results(self):
        """테스트용 백테스트 결과"""
        return [
            {
                'timestamp': datetime.now(),
                'symbol': 'BTCUSDT',
                'signal': 'BUY',
                'entry_price': 50000.0,
                'exit_price': 51000.0,
                'pnl': 0.02,
                'success': True,
                'confidence': 0.8
            },
            {
                'timestamp': datetime.now(),
                'symbol': 'BTCUSDT',
                'signal': 'SELL',
                'entry_price': 51000.0,
                'exit_price': 50000.0,
                'pnl': -0.01,
                'success': False,
                'confidence': 0.6
            }
        ]
    
    def test_max_drawdown(self, trading_config, sample_backtest_results, bybit_client):
        """최대 낙폭 계산 테스트"""
        module = TradingModule(trading_config, bybit_client)
        module.backtest_results = sample_backtest_results
        
        metrics = module.get_performance_metrics()
        assert 'max_drawdown' in metrics
        assert isinstance(metrics['max_drawdown'], float)
        assert 0 <= metrics['max_drawdown'] <= 1.0
    
    def test_risk_adjusted_return(self, trading_config, sample_backtest_results, bybit_client):
        """위험 조정 수익률 테스트"""
        module = TradingModule(trading_config, bybit_client)
        module.backtest_results = sample_backtest_results
        
        metrics = module.get_performance_metrics()
        assert 'risk_adjusted_return' in metrics
        assert isinstance(metrics['risk_adjusted_return'], float)
    
    def test_alpha_beta(self, trading_config, sample_backtest_results, bybit_client):
        """알파/베타 테스트"""
        module = TradingModule(trading_config, bybit_client)
        module.backtest_results = sample_backtest_results
        
        metrics = module.get_performance_metrics()
        assert 'alpha' in metrics
        assert 'beta' in metrics
        assert isinstance(metrics['alpha'], float)
        assert isinstance(metrics['beta'], float)
    
    def test_information_ratio(self, trading_config, sample_backtest_results, bybit_client):
        """Information Ratio 테스트"""
        module = TradingModule(trading_config, bybit_client)
        module.backtest_results = sample_backtest_results
        
        metrics = module.get_performance_metrics()
        assert 'information_ratio' in metrics
        assert isinstance(metrics['information_ratio'], float)
    
    def test_visualization(self, trading_config, sample_backtest_results, bybit_client):
        """백테스팅 시각화 테스트"""
        module = TradingModule(trading_config, bybit_client)
        module.backtest_results = sample_backtest_results
        
        viz_data = module.visualize_performance()
        
        # 필수 데이터 확인
        assert 'cumulative_returns' in viz_data
        assert 'win_rate_trend' in viz_data
        assert 'pnl_distribution' in viz_data
        assert 'trade_timestamps' in viz_data
        assert 'metrics' in viz_data
        
        # 데이터 타입 검증
        assert isinstance(viz_data['cumulative_returns'], list)
        assert isinstance(viz_data['win_rate_trend'], list)
        assert isinstance(viz_data['pnl_distribution'], list)
        assert isinstance(viz_data['trade_timestamps'], list)
        assert isinstance(viz_data['metrics'], dict)
        
        # 데이터 길이 검증
        assert len(viz_data['cumulative_returns']) == len(sample_backtest_results)
        assert len(viz_data['win_rate_trend']) == len(sample_backtest_results)
        assert len(viz_data['trade_timestamps']) == len(sample_backtest_results) 