import logging
from datetime import datetime
from src.core.wmguy import WMGuy

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def analyze_market(wmguy: WMGuy):
    """시장 분석 실행"""
    try:
        # BTC 분석
        symbol = "BTCUSDT"  # 비트코인으로 변경
        days = 30
        
        print(f"\n=== {symbol} 시장 분석 시작 ===")
        print(f"과거 {days}일 데이터 수집 및 학습 중...")
        
        # 시장 학습 및 분석
        analysis = wmguy.learn_market(symbol, days)
        
        # 결과 출력
        print("\n=== 시장 분석 결과 ===")
        print(f"심볼: {analysis['symbol']}")
        print(f"신호: {analysis['signal']}")
        print(f"강도: {analysis['strength']:.2f}")
        print(f"신뢰도: {analysis['confidence']:.2f}")
        print("\n=== 성과 지표 ===")
        print(f"정확도: {analysis['performance']['accuracy']:.2f}")
        print(f"평균 수익: {analysis['performance']['avg_return']:.2f}%")
        print(f"승률: {analysis['performance']['win_rate']:.2f}%")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise

def main():
    """메인 실행 함수"""
    setup_logging()
    
    try:
        # WMGuy 초기화
        wmguy = WMGuy()
        
        # 시장 분석 실행
        analyze_market(wmguy)
        
    except Exception as e:
        logging.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main() 