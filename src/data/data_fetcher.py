import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from tradingview_ta import TA_Handler, Interval
from pybit.unified_trading import HTTP
import pandas as pd
from dotenv import load_dotenv
import time
import numpy as np

class DataFetcher:
    """
    Bybit과 TradingView에서 데이터를 수집하는 클래스
    """
    def __init__(self, config: Dict[str, Any]):
        """
        데이터 수집기 초기화
        """
        self.config = config
        self.client = self._init_client()
        
        load_dotenv()
        
        self.symbols = config['trading']['symbols']
        self.timeframes = config['trading']['timeframes']
        
    def _init_client(self) -> HTTP:
        """Bybit API 클라이언트 초기화"""
        return HTTP(
            testnet=self.config['bybit']['testnet'],
            api_key=self.config['bybit'].get('api_key', ''),
            api_secret=self.config['bybit'].get('api_secret', '')
        )
        
    def get_tradingview_analysis(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """TradingView 기술적 분석 데이터 수집"""
        # TradingView 인터벌 매핑
        tv_interval_map = {
            '15': Interval.INTERVAL_15_MINUTES,
            '60': Interval.INTERVAL_1_HOUR,
            '240': Interval.INTERVAL_4_HOURS,
            '1h': Interval.INTERVAL_1_HOUR,
            '4h': Interval.INTERVAL_4_HOURS,
            'D': Interval.INTERVAL_1_DAY,
            'W': Interval.INTERVAL_1_WEEK
        }
        
        if timeframe not in tv_interval_map:
            print(f"지원하지 않는 TradingView 인터벌: {timeframe}")
            return None
            
        handler = TA_Handler(
            symbol=symbol,
            exchange="BYBIT",
            screener="crypto",
            interval=tv_interval_map[timeframe]
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
            print(f"TradingView 데이터 조회 오류: {str(e)}")
            return None
            
    def fetch_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """특정 심볼의 K라인 데이터 조회"""
        try:
            # 현재 시간 기준으로 end_time 설정
            current_time = int(time.time())
            end_time = current_time * 1000
            
            # interval에 따른 시작 시간 계산
            interval_seconds = self._interval_to_seconds(interval)
            start_time = end_time - (interval_seconds * limit * 1000)
            
            # 데이터 조회 시간 로깅
            print(f"데이터 조회: {symbol}, 인터벌: {interval}")
            print(f"시작: {datetime.fromtimestamp(start_time/1000)}")
            print(f"종료: {datetime.fromtimestamp(end_time/1000)}")
            
            # API 요청
            response = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                start=start_time,
                end=end_time,
                limit=limit
            )
            
            if response['retCode'] != 0:
                print(f"API 오류: {response['retMsg']}")
                return pd.DataFrame()
                
            # 데이터 변환 및 검증
            if not response['result'].get('list'):
                print("데이터가 없습니다")
                return pd.DataFrame()
                
            data = response['result']['list']
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # 데이터 타입 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = df[col].astype(float)
                
            # 시간순 정렬
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 데이터 검증
            if len(df) < limit * 0.5:  # 50% 이상의 데이터가 있어야 함
                print(f"데이터가 충분하지 않습니다: {len(df)}/{limit}")
                return pd.DataFrame()
                
            # 현재 시간 기준으로 데이터 필터링
            now = pd.Timestamp.now()
            df = df[df['timestamp'] <= now]
            
            # 마지막 캔들 시간 체크
            if len(df) > 0:
                last_candle = df.iloc[-1]
                time_diff = now - last_candle['timestamp']
                
                # 인터벌별 허용 시간차 설정
                allowed_delay = {
                    '15': pd.Timedelta(minutes=30),
                    '60': pd.Timedelta(hours=2),
                    '240': pd.Timedelta(hours=8)
                }.get(interval, pd.Timedelta(hours=1))
                
                if time_diff > allowed_delay:
                    print(f"마지막 캔들이 너무 오래되었습니다: {time_diff}")
                    return pd.DataFrame()
            else:
                print("유효한 데이터가 없습니다")
                return pd.DataFrame()
                
            return df
            
        except Exception as e:
            print(f"데이터 조회 오류: {str(e)}")
            return pd.DataFrame()
            
    def _interval_to_seconds(self, interval: str) -> int:
        """인터벌 문자열을 초 단위로 변환"""
        # Bybit API 인터벌 매핑
        interval_map = {
            '15': '15',  # 15분
            '60': '60',  # 1시간
            '240': '240',  # 4시간
            '1h': '60',
            '4h': '240',
            'D': '1440',  # 1일
            'W': '10080'  # 1주
        }
        
        if interval not in interval_map:
            raise ValueError(f"지원하지 않는 인터벌: {interval}")
            
        minutes = int(interval_map[interval])
        return minutes * 60  # 초 단위로 변환
            
    def get_market_data(self) -> Dict[str, Dict[str, Dict[str, Optional[Any]]]]:
        """모든 심볼에 대한 시장 데이터 수집"""
        market_data = {}
        
        for symbol in self.symbols:
            # 현재가 조회
            current_price = self.get_current_price(symbol)
            if current_price == 0:
                print(f"{symbol} 현재가 조회 실패")
                continue
                
            print(f"\n{symbol} 현재가: {current_price}")
            symbol_data = {}
            
            for timeframe in self.timeframes:
                # 과거 데이터 조회
                kline_data = self.fetch_klines(symbol, timeframe)
                if len(kline_data) == 0:
                    continue
                    
                # 현재가로 마지막 종가 업데이트
                close_idx = kline_data.columns.get_loc('close')
                kline_data.iat[-1, close_idx] = current_price
                
                # TradingView 분석
                tv_analysis = self.get_tradingview_analysis(symbol, timeframe)
                
                symbol_data[timeframe] = {
                    'kline': kline_data,
                    'analysis': tv_analysis,
                    'current_price': current_price
                }
            
            market_data[symbol] = symbol_data
            
        return market_data

    def get_current_price(self, symbol: str) -> float:
        """현재가 조회"""
        try:
            response = self.client.get_tickers(
                category="spot",
                symbol=symbol
            )
            
            if not isinstance(response, dict):
                print("API 응답이 딕셔너리 형식이 아닙니다")
                return 0.0
                
            ret_code = response.get('retCode')
            if ret_code != 0:
                print(f"API 오류: {response.get('retMsg', 'Unknown error')}")
                return 0.0
                
            result = response.get('result', {})
            price = float(result.get('list', [{}])[0].get('lastPrice', 0))
            
            print(f"현재가 조회: {symbol} = {price}")
            return price
            
        except Exception as e:
            print(f"현재가 조회 오류: {str(e)}")
            return 0.0 