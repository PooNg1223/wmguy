import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from tradingview_ta import TA_Handler, Interval
from pybit import HTTP
import pandas as pd
from dotenv import load_dotenv

class DataFetcher:
    """
    Bybit과 TradingView에서 데이터를 수집하는 클래스
    """
    def __init__(self, config: Dict[str, Any]):
        load_dotenv()
        
        # Bybit API 초기화
        self.client = HTTP(
            endpoint="https://api-testnet.bybit.com" if config['bybit']['testnet'] else "https://api.bybit.com",
            api_key=os.getenv('BYBIT_API_KEY'),
            api_secret=os.getenv('BYBIT_API_SECRET')
        )
        
        self.symbols = config['trading']['symbols']
        self.timeframes = config['trading']['timeframes']
        
    def get_tradingview_analysis(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        TradingView 기술적 분석 데이터 수집
        """
        handler = TA_Handler(
            symbol=symbol,
            exchange="BYBIT",
            screener="crypto",
            interval=timeframe
        )
        
        try:
            analysis = handler.get_analysis()
            if analysis is None:
                return None
                
            return {
                'summary': analysis.summary,
                'oscillators': analysis.oscillators,
                'moving_averages': analysis.moving_averages,
                'indicators': analysis.indicators
            }
        except Exception as e:
            print(f"Error fetching TradingView data: {e}")
            return None
            
    def get_kline_data(self, symbol: str, interval: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """
        Bybit에서 캔들스틱 데이터 수집
        """
        try:
            # 현재 시간에서 3일 전으로 설정
            from_time = int((datetime.now() - timedelta(days=3)).timestamp())
            
            print(f"Requesting kline data for {symbol} with interval {interval} from {from_time}")
            
            response = self.client.query_kline(
                symbol=symbol,
                interval=interval,
                limit=limit,
                from_time=from_time
            )
            
            print(f"API 응답: {response}")  # API 응답 출력
            
            if response.get('ret_code') == 0 and response.get('result', []):
                df = pd.DataFrame(response['result'])
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # 숫자형으로 변환
                for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                    df[col] = pd.to_numeric(df[col])
                    
                return df
            
            return None
            
        except Exception as e:
            print(f"Error fetching kline data: {e}")
            return None
            
    def get_market_data(self) -> Dict[str, Dict[str, Dict[str, Optional[Any]]]]:
        """
        모든 심볼에 대한 시장 데이터 수집
        
        Returns:
            Dict[str, Dict[str, Dict[str, Optional[Any]]]]: 
            {
                'BTCUSDT': {
                    '15': {
                        'kline': DataFrame or None,
                        'analysis': Dict or None
                    },
                    '1h': {...},
                    '4h': {...}
                },
                'ETHUSDT': {...}
            }
        """
        market_data = {}
        
        for symbol in self.symbols:
            symbol_data = {}
            
            for timeframe in self.timeframes:
                kline_data = self.get_kline_data(symbol, timeframe)
                tv_analysis = self.get_tradingview_analysis(symbol, timeframe)
                
                print(f"{symbol} {timeframe} 데이터: {kline_data}")  # 디버깅용 출력
                
                symbol_data[timeframe] = {
                    'kline': kline_data,
                    'analysis': tv_analysis
                }
            
            market_data[symbol] = symbol_data
            
        return market_data 