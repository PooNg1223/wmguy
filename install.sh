#!/bin/bash

# 시스템 패키지 업데이트
sudo apt-get update && sudo apt-get upgrade -y

# 필요한 시스템 의존성 설치
sudo apt-get install -y \
    python3.10 \          # Python 3.10
    python3-pip \         # pip 패키지 관리자
    docker.io \           # Docker 엔진
    docker-compose \      # Docker Compose
    git \                 # 버전 관리
    postgresql \          # PostgreSQL 데이터베이스
    redis-server \        # Redis 서버
    ta-lib                # 기술적 분석 라이브러리

# Python 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# Python 패키지 설치
pip install -r requirements.txt

# 필요한 디렉토리 구조 생성
mkdir -p {data,logs,models}/{trading,work,life}

# 데이터베이스 초기화
python scripts/init_db.py

# Docker 서비스 시작
docker-compose up -d 