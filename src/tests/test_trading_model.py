import sys
import os
import yaml
import pandas as pd
from datetime import datetime, timedelta

# 상위 디렉토리 추가하여 모듈 import 가능하게 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_fetcher import DataFetcher
from src.models.trading_model import TradingModel

def load_config():
    with open('config/trading_config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_trading_model():
    """
    트레이딩 모델 테스트
    """
    try:
        # 설정 로드
        config = load_config()
        
        # 데이터 수집기 초기화
        fetcher = DataFetcher(config)
        
        # 트레이딩 모델 초기화
        trading_model = TradingModel(config)
        
        # BTCUSDT 데이터 수집
        print("데이터 수집 중...")
        market_data = fetcher.get_market_data()
        
        for symbol in config['trading']['symbols']:
            print(f"\n{symbol} 분석 중...")
            
            for timeframe in config['trading']['timeframes']:
                print(f"\n{timeframe} 타임프레임 분석:")
                
                kline_data = market_data[symbol][timeframe]['kline']
                if kline_data is None:
                    print("캔들스틱 데이터 없음")
                    continue
                    
                # 거래 기회 분석
                trade_opportunity = trading_model.analyze_trade_opportunity(kline_data, timeframe)
                
                print(f"매매 신호: {trade_opportunity['signal']}")
                print(f"진입 가격: {trade_opportunity['entry_price']}")
                print(f"스탑로스: {trade_opportunity['stop_loss']}")
                print(f"신뢰도: {trade_opportunity['confidence']:.2f}")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    test_trading_model() 