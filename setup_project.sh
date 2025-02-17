#!/bin/bash

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# setuptools 먼저 설치
pip install --upgrade pip setuptools wheel

# 필요한 패키지 설치
pip install -r requirements.txt

# 프로젝트 디렉토리 구조 생성
mkdir -p src/{analysis,data,utils,tests}
mkdir -p config
mkdir -p logs

# 설정 파일 템플릿 생성
cat > config/trading_config.yaml << EOL
# Bybit API 설정
bybit:
  testnet: true  # 테스트넷 사용 여부
  market_type: "linear"  # linear/inverse
  leverage: 1  # 레버리지 설정

# 트레이딩 설정
trading:
  symbols:
    - 'BTCUSDT'
    - 'ETHUSDT'
  timeframes:
    - '15'    # 15분
    - '60'    # 1시간
    - '240'   # 4시간
  initial_capital: 10000  # 초기 자본금
  max_positions: 3  # 최대 동시 포지션 수

# 포지션 관리 설정
position:
  risk_per_trade: 0.02      # 트레이드당 리스크 비율 (0-0.1)
  max_position_size: 0.3    # 최대 포지션 크기 비율 (0-1)
  min_confidence: 60.0      # 최소 신뢰도 요구사항 (0-100)
  profit_target_ratio: 2.0  # 익절 비율 (손실 대비)
  trailing_stop: true       # 트레일링 스탑 사용 여부
  
  # 시장 상태별 리스크 조정 설정
  market_regime_multipliers:
    uptrend: 1.2     # 상승 추세: 리스크 증가
    downtrend: 1.0   # 하락 추세: 기본 리스크
    ranging: 0.8     # 횡보: 리스크 감소
    
  volatility_multipliers:
    low: 1.2         # 낮은 변동성: 리스크 증가
    medium: 1.0      # 중간 변동성: 기본 리스크
    high: 0.8        # 높은 변동성: 리스크 감소
    
  risk_level_multipliers:
    low: 1.2         # 낮은 리스크: 포지션 증가
    medium: 1.0      # 중간 리스크: 기본값
    high: 0.8        # 높은 리스크: 포지션 감소

# 기술적 지표 설정
indicators:
  rsi:
    period: 14
    overbought: 70
    oversold: 30
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
  bollinger:
    period: 20
    std_dev: 2.0
  stochastic:
    k_period: 14
    d_period: 3

# 백테스팅 설정
backtest:
  start_date: '2024-01-01'
  end_date: '2024-02-17'
  commission: 0.0004     # 거래 수수료 (0.04%)
  slippage: 0.0002      # 슬리피지 (0.02%)

# 로깅 설정
logging:
  level: "INFO"
  save_trades: true 

# 시장 분석 설정
market_analysis:
  trend_threshold: 25  # ADX 기준값
  volatility_multiplier: 1.5  # 변동성 체계 구분 배수
  support_resistance_lookback: 20  # 지지/저항 탐색 기간
EOL

# 환경변수 파일 템플릿 생성
cat > .env << EOL
# Bybit API 키 설정
BYBIT_API_KEY=''
BYBIT_API_SECRET=''

# 데이터베이스 설정
DB_HOST='localhost'
DB_PORT=5432
DB_NAME='trading_db'
DB_USER='trading_user'
DB_PASSWORD='your_password'
EOL

# .gitignore 파일 생성
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.env

# IDE
.idea/
.vscode/
*.swp
*.swo

# 로그 및 데이터
logs/
data/
*.log
*.csv
*.sqlite3

# 기타
.DS_Store
EOL

echo "프로젝트 설정이 완료되었습니다." 