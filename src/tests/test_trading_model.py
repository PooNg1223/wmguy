import os
import sys
import yaml
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any
import numpy as np

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.trading.trading_model import TradingModel

def load_config():
    """설정 파일 로드"""
    config_path = os.path.join(project_root, 'config', 'trading_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_test_data(symbol: str, interval: int, periods: int = 200) -> pd.DataFrame:
    """테스트용 OHLCV 데이터 생성"""
    now = datetime.now()
    dates = pd.date_range(end=now, periods=periods, freq=f'{interval}min')
    
    # 기본 가격 설정
    base_price = 90000 if symbol == 'BTCUSDT' else 2700  # BTC or ETH
    
    # 랜덤 가격 변동 생성
    np.random.seed(42)  # 재현성을 위한 시드 설정
    
    # 시간대별 변동성 조정
    volatility_scale = {
        15: 0.02,   # 15분봉: 높은 변동성
        60: 0.015,  # 1시간봉: 중간 변동성
        240: 0.01   # 4시간봉: 낮은 변동성
    }.get(interval, 0.015)
    
    # 추세 성분 (상승 추세 + 사이클)
    trend = np.linspace(0, 0.3, periods)
    cycle = 0.1 * np.sin(np.linspace(0, 4*np.pi, periods))
    
    # 변동성 성분 (시간대별 조정)
    volatility = np.random.normal(0, volatility_scale, periods)
    
    # 가격 변동 계산 (추세 + 사이클 + 변동성)
    price_changes = trend + cycle + volatility
    
    # OHLCV 데이터 생성
    data = {
        'timestamp': dates,
        'open': [base_price * (1 + price_changes[i]) for i in range(periods)],
        'close': [base_price * (1 + price_changes[i] + np.random.normal(0, volatility_scale/2)) 
                 for i in range(periods)]
    }
    
    # 고가와 저가 생성 (시간대별 스프레드 조정)
    spread_scale = volatility_scale * 0.5
    data['high'] = [max(o, c) + abs(np.random.normal(0, o * spread_scale)) 
                    for o, c in zip(data['open'], data['close'])]
    data['low'] = [min(o, c) - abs(np.random.normal(0, o * spread_scale)) 
                   for o, c in zip(data['open'], data['close'])]
    
    # 거래량 생성 (가격 변동과 시간대 고려)
    volume_base = 1000 if symbol == 'BTCUSDT' else 100
    volume_scale = {
        15: 1.0,    # 15분봉: 기본 거래량
        60: 3.0,    # 1시간봉: 3배 거래량
        240: 10.0   # 4시간봉: 10배 거래량
    }.get(interval, 1.0)
    
    data['volume'] = [volume_base * volume_scale * (1 + abs(price_changes[i]) * 10) * 
                     (1 + np.random.normal(0, 0.5)) for i in range(periods)]
    
    # 거래대금 계산
    data['turnover'] = [p * v for p, v in zip(data['close'], data['volume'])]
    
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp')
    
    # 데이터 타입 변환
    for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
        df[col] = df[col].astype(float)
        
    return df

def print_analysis_result(symbol: str, timeframe: str, analysis: Dict[str, Any]):
    """분석 결과 출력"""
    print(f"\n{timeframe} 타임프레임 분석:\n")
    
    # 시장 상태 분석
    market_state = analysis['market_state']
    print("시장 상태 분석:")
    print(f"시장 레짐: {market_state['regime']}")
    print(f"변동성: {market_state['volatility']}")
    print(f"추세 강도: {market_state['trend_strength']:.2f}")
    print(f"리스크 레벨: {market_state['risk_level']}")
    print(f"주요 지지선: {market_state['support_levels']}")
    print(f"주요 저항선: {market_state['resistance_levels']}\n")
    
    # 포지션 관리
    print("포지션 관리:")
    print("시장 상태:")
    print(f"  - 레짐: {market_state['regime']}")
    print(f"  - 변동성: {market_state['volatility']}")
    print(f"  - 리스크 레벨: {market_state['risk_level']}\n")
    
    # 매매 신호
    signal = analysis['signal']
    print("매매 신호:")
    print(f"  - 신호: {signal['signal']}")
    print(f"  - 신뢰도: {signal['confidence']:.2f}\n")
    
    # 포지션 정보
    position = analysis['position']
    print("포지션 정보:")
    print(f"  - 크기: {position['size']:.4f}")
    print(f"  - 진입가격: {position['entry_price']:.2f}")
    print(f"  - 스탑로스: {position['stop_loss']:.2f}")
    print(f"  - 이익실현: {position['take_profit']:.2f}")
    if 'risk_ratio' in position:
        print(f"  - 리스크:리워드 = {position['risk_ratio']}")
    
    # 예상 손익
    risk = abs(position['entry_price'] - position['stop_loss']) * position['size']
    reward = abs(position['take_profit'] - position['entry_price']) * position['size']
    print(f"\n예상 손익:")
    print(f"  - 최대손실: ${risk:.2f}")
    print(f"  - 목표이익: ${reward:.2f}")

def main():
    """테스트 실행"""
    # 설정 로드
    config = load_config()
    
    # 트레이딩 모델 초기화
    model = TradingModel(config)
    
    print("데이터 수집 중...")
    
    # 각 심볼에 대해 테스트 데이터 생성
    for symbol in config['trading']['symbols']:
        for timeframe in config['trading']['timeframes']:
            interval = int(timeframe)  # '15' -> 15
            df = create_test_data(symbol, interval)
            print(f"{symbol} {timeframe} 데이터:\n{df.head()}\n...")
            
            # 데이터를 모델의 캐시에 저장
            if symbol not in model.data:
                model.data[symbol] = {}
            model.data[symbol][timeframe] = df
    
    # 분석 실행
    for symbol in config['trading']['symbols']:
        print(f"\n{symbol} 분석 중...")
        
        for timeframe in config['trading']['timeframes']:
            analysis = model.analyze_timeframe(symbol, timeframe)
            
            if analysis:
                print_analysis_result(symbol, timeframe, analysis)
            else:
                print(f"\n{timeframe} 타임프레임 분석: 오류 발생")
                print(f"오류 내용: {analysis}")

if __name__ == "__main__":
    main() 