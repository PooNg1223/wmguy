from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from .bybit_client import BybitClient
from .market_analysis import MarketAnalysis
from .data_collector import DataCollector
from .market_learner import MarketLearner

class TradingModule:
    """트레이딩 분석 및 백테스팅 모듈"""
    
    def __init__(self, config: Dict[str, Any], bybit_client: BybitClient):
        self.logger = logging.getLogger('trading')
        self.config = config
        self.bybit = bybit_client
        
        # 데이터 수집기 초기화
        self.collector = DataCollector(bybit_client)
        
        # 분석기 초기화
        self.market_analyzer = MarketAnalysis()
        self.learner = MarketLearner(self.collector)
        
        # 분석 결과 저장소
        self.analysis_history: List[Dict[str, Any]] = []
        self.backtest_results: List[Dict[str, Any]] = []
        self.trades_history: List[Dict[str, Any]] = []
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        # 설정
        self.strategy_config = config.get('strategy', {})
        
        self.logger.info("Trading analysis module initialized")
    
    def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """시장 분석 및 예측"""
        try:
            # 데이터 수집
            market_data = self.collector.get_historical_data(symbol)
            
            # 세력 활동 감지
            whale_activity = self.market_analyzer.detect_whale_activity(market_data)
            
            # 시장 분석
            analysis = self.learner.analyze(symbol)
            
            # 결과 저장
            result = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'signal': analysis['signal'],
                'strength': analysis['strength'],
                'confidence': analysis['confidence'],
                'whale_activity': whale_activity,
                'predictions': analysis['predictions']
            }
            
            self.analysis_history.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
            return {
                'signal': 'NEUTRAL',
                'strength': 0.0,
                'confidence': 0.0
            }
    
    def backtest_signal(self, symbol: str, signal: Dict[str, Any], 
                       periods: int = 5) -> Dict[str, Any]:
        """신호 백테스팅"""
        try:
            # 현재 가격과 미래 가격 데이터 수집
            kline_data = self.collector.get_historical_data(symbol, days=1)
            if kline_data.empty:
                raise ValueError("No data available")
            
            current_price = float(kline_data['close'].iloc[-1])
            future_price = float(kline_data['close'].iloc[-periods]) if len(kline_data) > periods else current_price
            
            # 수익률 계산
            if signal['signal'] == 'BUY':
                pnl = (future_price - current_price) / current_price
            elif signal['signal'] == 'SELL':
                pnl = (current_price - future_price) / current_price
            else:
                pnl = 0.0
            
            # 결과 저장
            result = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'signal': signal['signal'],
                'entry_price': current_price,
                'exit_price': future_price,
                'pnl': pnl,
                'pnl_percentage': pnl * 100,
                'periods': periods,
                'success': (pnl > 0) if signal['signal'] != 'NEUTRAL' else True,
                'confidence': signal.get('confidence', 0.0)
            }
            
            self.backtest_results.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {}
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """최대 낙폭 계산"""
        try:
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            return float(abs(min(drawdowns)))
        except Exception as e:
            self.logger.error(f"Max drawdown calculation failed: {e}")
            return 0.0

    def _calculate_risk_adjusted_return(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """위험 조정 수익률 계산"""
        try:
            if not returns:
                return 0.0
            
            # 연간 수익률로 변환 (거래당 수익률 -> 연간)
            annual_return = np.mean(returns) * 252  # 연간 거래일 기준
            
            # 연간 변동성
            annual_volatility = np.std(returns) * np.sqrt(252)
            
            # Sortino Ratio 계산 (하방 위험만 고려)
            downside_returns = [r for r in returns if r < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0.0
            
            if downside_volatility == 0:
                return 0.0
            
            sortino_ratio = float((annual_return - risk_free_rate) / downside_volatility)
            return sortino_ratio
            
        except Exception as e:
            self.logger.error(f"Risk-adjusted return calculation failed: {e}")
            return 0.0

    def _calculate_alpha_beta(self, returns: List[float], market_returns: Optional[List[float]] = None) -> Tuple[float, float]:
        """알파와 베타 계산"""
        try:
            if not returns or not market_returns or len(returns) != len(market_returns):
                return 0.0, 0.0
            
            # 넘파이 배열로 변환
            returns_arr = np.array(returns)
            market_arr = np.array(market_returns)
            
            # 공분산과 분산 계산
            covariance = np.cov(returns_arr, market_arr)[0][1]
            market_variance = np.var(market_arr)
            
            # 베타 계산
            beta = float(covariance / market_variance) if market_variance != 0 else 0.0
            
            # 알파 계산 (연간화)
            alpha = float(np.mean(returns_arr - beta * market_arr) * 252)
            
            return alpha, beta
            
        except Exception as e:
            self.logger.error(f"Alpha/Beta calculation failed: {e}")
            return 0.0, 0.0

    def _calculate_information_ratio(self, returns: List[float], benchmark_returns: Optional[List[float]] = None) -> float:
        """Information Ratio 계산"""
        try:
            if not returns or not benchmark_returns or len(returns) != len(benchmark_returns):
                return 0.0
            
            # 초과 수익률 계산
            excess_returns = np.array(returns) - np.array(benchmark_returns)
            
            # 추적 오차 계산
            tracking_error = float(np.std(excess_returns))
            if tracking_error == 0:
                return 0.0
            
            # Information Ratio 계산
            ir = float(np.mean(excess_returns) / tracking_error)
            return ir * np.sqrt(252)  # 연간화
            
        except Exception as e:
            self.logger.error(f"Information ratio calculation failed: {e}")
            return 0.0

    def get_performance_metrics(self) -> Dict[str, Any]:
        """성과 지표 계산"""
        if not self.backtest_results:
            return {}
        
        actual_trades = [r for r in self.backtest_results if r['signal'] != 'NEUTRAL']
        if not actual_trades:
            return {}
        
        pnls = [trade['pnl'] for trade in actual_trades]
        winning_trades = [trade for trade in actual_trades if trade['success']]
        
        # 시장 수익률 가져오기 (BTC 기준)
        market_returns = self._get_market_returns('BTCUSDT', len(pnls))
        alpha, beta = self._calculate_alpha_beta(pnls, market_returns)
        
        return {
            'total_trades': len(actual_trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(actual_trades),
            'avg_pnl': np.mean(pnls),
            'max_pnl': max(pnls),
            'min_pnl': min(pnls),
            'sharpe_ratio': self._calculate_sharpe_ratio(pnls),
            'max_drawdown': self._calculate_max_drawdown(pnls),
            'risk_adjusted_return': self._calculate_risk_adjusted_return(pnls),
            'alpha': alpha,
            'beta': beta,
            'confidence_correlation': self._calculate_confidence_correlation(actual_trades),
            'information_ratio': self._calculate_information_ratio(pnls, market_returns)
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """샤프 비율 계산"""
        if not returns:
            return 0.0
        std = float(np.std(returns))
        if std == 0:
            return 0.0
        return float(np.mean(returns) / std)
    
    def _calculate_confidence_correlation(self, trades: List[Dict[str, Any]]) -> float:
        """신뢰도와 수익의 상관관계"""
        if not trades:
            return 0.0
        
        confidences = [trade['confidence'] for trade in trades]
        returns = [trade['pnl'] for trade in trades]
        
        if len(confidences) < 2:
            return 0.0
            
        return float(np.corrcoef(confidences, returns)[0, 1])

    def add_trade(self, trade_info: Dict[str, Any]) -> bool:
        """
        수동으로 거래 정보 추가
        
        Args:
            trade_info: {
                'symbol': 'BTCUSDT',
                'side': 'BUY' or 'SELL',
                'price': 50000.0,
                'size': 0.1,
                'memo': '첫 매수'
            }
        """
        try:
            trade = {
                'timestamp': datetime.now(),
                'trade_info': trade_info
            }
            self.trades_history.append(trade)
            self.logger.info(f"Trade added: {trade_info}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add trade: {e}")
            return False
    
    def get_daily_summary(self) -> Dict[str, Any]:
        """일일 거래 요약"""
        return {
            'total_trades': len(self.trades_history),
            'trades': self.trades_history
        }
    
    def get_position_status(self, symbol: str) -> Dict[str, Any]:
        """
        특정 종목의 실시간 평가 정보
        
        Args:
            symbol: 종목명 (예: 'BTCUSDT')
            
        Returns:
            Dict[str, Any]: {
                'symbol': 'BTCUSDT',
                'entry_price': 50000.0,
                'current_price': 51000.0,
                'position_size': 0.1,
                'unrealized_pnl': 100.0,
                'pnl_percentage': 2.0,
                'holding_period': '2일 13시간',
                'last_updated': datetime.now()
            }
        """
        return self.positions.get(symbol, {
            'symbol': symbol,
            'position_size': 0.0,
            'entry_price': 0.0,
            'current_price': 0.0,
            'unrealized_pnl': 0.0
        })
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """현재 가격 조회"""
        try:
            if not self.bybit:
                self.logger.error("Bybit client not initialized")
                return None
            
            kline = self.bybit.get_kline(symbol, interval="1", limit=1)
            if kline and 'result' in kline and kline['result']['list']:
                return float(kline['result']['list'][0][4])  # 종가 반환
            return None
        except Exception as e:
            self.logger.error(f"Failed to get current price: {e}")
            return None
    
    def _get_technical_data(self, symbol: str) -> Dict[str, Any]:
        """기술적 지표 계산을 위한 데이터 조회"""
        try:
            if not self.bybit:
                self.logger.error("Bybit client not initialized")
                return {'close_prices': []}
            
            kline = self.bybit.get_kline(symbol, interval="15", limit=100)
            if not kline or 'result' not in kline:
                return {'close_prices': []}
            
            close_prices = [float(k[4]) for k in kline['result']['list']]
            return {
                'close_prices': close_prices,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Failed to get technical data: {e}")
            return {'close_prices': []}
    
    def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """시장 데이터 조회"""
        try:
            if not self.bybit:
                self.logger.error("Bybit client not initialized")
                return {'volumes': [], 'prices': []}
            
            kline = self.bybit.get_kline(symbol, interval="5", limit=100)
            if not kline or 'result' not in kline:
                return {'volumes': [], 'prices': []}
            
            volumes = []
            prices = []
            for k in kline['result']['list']:
                volumes.append(float(k[5]))  # 거래량
                prices.append(float(k[4]))   # 종가
            
            return {
                'volumes': volumes,
                'prices': prices,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Failed to get market data: {e}")
            return {'volumes': [], 'prices': []}
    
    def execute_signal(self, symbol: str, signal: Dict[str, Any]) -> bool:
        """매매 신호 실행"""
        try:
            if not self.bybit:
                self.logger.error("Bybit client not initialized")
                return False
            
            current_position = self.bybit.get_position(symbol)
            whale_activity = signal.get('whale_activity', {})
            
            # 포지션 크기 계산
            base_size = self._calculate_position_size(
                symbol=symbol,
                signal_strength=signal['strength'],
                whale_activity=whale_activity
            )
            
            # 진입 조건 확인
            if signal['signal'] == 'BUY':
                if not current_position or current_position['size'] <= 0:
                    # 세력 매수 감지 시 진입 크기 증가
                    if whale_activity.get('is_active') and whale_activity.get('direction') == 'BUY':
                        base_size *= (1 + whale_activity['activity_score'])
                    
                    return self._place_order(symbol, 'BUY', base_size)
                    
            elif signal['signal'] == 'SELL':
                if not current_position or current_position['size'] >= 0:
                    # 세력 매도 감지 시 진입 크기 증가
                    if whale_activity.get('is_active') and whale_activity.get('direction') == 'SELL':
                        base_size *= (1 + whale_activity['activity_score'])
                    
                    return self._place_order(symbol, 'SELL', base_size)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Signal execution failed: {e}")
            return False
    
    def _calculate_position_size(self, symbol: str, signal_strength: float,
                               whale_activity: Dict[str, Any]) -> float:
        """포지션 크기 계산"""
        try:
            # 기본 설정 가져오기
            base_size = float(self.strategy_config['position_sizing']['base_size'])
            max_size = float(self.strategy_config['position_sizing']['max_size'])
            
            # 신호 강도에 따른 조정
            position_size = base_size * (1 + signal_strength)
            
            # 세력 활동 고려
            if whale_activity.get('is_active'):
                # 유동성 점수에 따른 조정
                liquidity_factor = 1 + whale_activity.get('liquidity_score', 0)
                position_size *= liquidity_factor
                
                # 연속 패턴에 따른 조정
                consecutive_factor = 1 + (whale_activity.get('consecutive_pattern', 0) / 10)
                position_size *= consecutive_factor
            
            # 최대 크기 제한
            position_size = min(position_size, max_size)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    def _place_order(self, symbol: str, side: str, size: float) -> bool:
        """주문 실행"""
        try:
            if not self.bybit:
                self.logger.error("Bybit client not initialized")
                return False
            
            current_price = self.get_current_price(symbol)
            if not current_price:
                self.logger.error("Current price not available")
                return False
            
            order = self.bybit.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(size),
                timeInForce="GTC"
            )
            
            if isinstance(order, dict) and order.get('result'):
                trade_info = {
                    'symbol': symbol,
                    'side': side,
                    'price': current_price,
                    'size': size,
                    'order_id': order.get('result', {}).get('orderId')
                }
                self.add_trade(trade_info)
                return True
            
            self.logger.error("Order failed")
            return False
            
        except Exception as e:
            self.logger.error(f"Order placement failed: {e}")
            return False
    
    def _get_market_returns(self, symbol: str, periods: int) -> List[float]:
        """시장 수익률 데이터 가져오기"""
        try:
            df = self.collector.get_historical_data(symbol, days=max(30, periods//48))
            if df.empty:
                return [0.0] * periods
            
            # 수익률 계산 및 numpy array로 명시적 변환
            returns = np.array(df['close'].pct_change().fillna(0.0).values, dtype=np.float64)
            
            # 기간에 맞게 데이터 자르기
            if len(returns) > periods:
                returns = returns[-periods:]
            elif len(returns) < periods:
                # 부족한 데이터는 0으로 채우기 (pad 대신 zeros 사용)
                padding = np.zeros(periods - len(returns), dtype=np.float64)
                returns = np.concatenate([returns, padding])
            
            # 명시적으로 float 리스트로 변환
            return [float(x) for x in returns]
            
        except Exception as e:
            self.logger.error(f"Failed to get market returns: {e}")
            return [0.0] * periods

    def visualize_performance(self) -> Dict[str, Any]:
        """백테스팅 결과 시각화 데이터 생성"""
        try:
            if not self.backtest_results:
                return {}
            
            # 결과를 DataFrame으로 변환
            df = pd.DataFrame(self.backtest_results)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # 누적 수익률 계산
            df['cumulative_pnl'] = (1 + df['pnl']).cumprod() - 1
            
            # 승률 추이
            df['win_count'] = df['success'].cumsum()
            df['trade_count'] = range(1, len(df) + 1)
            df['rolling_win_rate'] = df['win_count'] / df['trade_count']
            
            # 신뢰도와 수익률 관계
            confidence_pnl_corr = self._calculate_confidence_correlation(self.backtest_results)
            
            # 타임스탬프를 문자열로 변환 (strftime 에러 수정)
            timestamps = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in df.index]
            
            return {
                'cumulative_returns': df['cumulative_pnl'].tolist(),
                'win_rate_trend': df['rolling_win_rate'].tolist(),
                'pnl_distribution': df['pnl'].tolist(),
                'confidence_correlation': float(confidence_pnl_corr),
                'trade_timestamps': timestamps,
                'metrics': self.get_performance_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Performance visualization failed: {e}")
            return {} 