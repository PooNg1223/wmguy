import sys
import os
import yaml
import pandas as pd
from datetime import datetime, timedelta

# 상위 디렉토리 추가하여 모듈 import 가능하게 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_fetcher import DataFetcher
from analysis.technical_indicators import TechnicalAnalyzer
from analysis.market_sentiment import MarketSentimentAnalyzer

def load_config():
    with open('config/trading_config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_system():
    """
    전체 시스템 테스트
    """
    # 설정 로드
    config = load_config()
    
    # 데이터 수집기 초기화
    fetcher = DataFetcher(config)
    
    # 기술적 분석기 초기화
    tech_analyzer = TechnicalAnalyzer(config)
    
    # 시장 심리 분석기 초기화
    sentiment_analyzer = MarketSentimentAnalyzer(config)
    
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
                
            # 기술적 지표 계산
            indicators = tech_analyzer.calculate_indicators(kline_data)
            signals = tech_analyzer.analyze_signals(indicators)
            combined_signal = tech_analyzer.get_combined_signal(signals)
            
            # 시장 심리 분석
            volume_trend = sentiment_analyzer.analyze_volume_trend(kline_data)
            momentum = sentiment_analyzer.analyze_price_momentum(kline_data)
            market_sentiment = sentiment_analyzer.get_market_sentiment(volume_trend, momentum)
            
            print(f"기술적 분석 신호: {combined_signal}")
            print(f"시장 심리: {market_sentiment}")
            print("\n주요 지표:")
            print(f"RSI: {indicators['rsi']['value'].iloc[-1]:.2f}")
            print(f"MACD 히스토그램: {indicators['macd']['histogram'].iloc[-1]:.2f}")
            print(f"거래량 강도: {volume_trend['volume_strength']:.2f}")
            print(f"모멘텀 강도: {momentum['momentum_strength']:.2f}")

if __name__ == "__main__":
    test_system() 