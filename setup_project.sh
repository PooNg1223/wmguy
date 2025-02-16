#!/bin/bash

# 1. 프로젝트 구조 정리
echo "프로젝트 구조 정리 중..."
rm -rf .ai .github tests venv wmguy_memory.json AI-STATUS.md

mkdir -p src/analysis src/data src/models src/utils config scripts

# 2. GitHub 저장소 생성 안내
echo "GitHub에서 새로운 저장소를 생성하세요."

# 3. 로컬 저장소 초기화 및 연결
echo "로컬 저장소 초기화 및 연결 중..."
git init
echo "GitHub 저장소 URL을 입력하세요:"
read repo_url
git remote add origin $repo_url

# 4. 자동 푸시 스크립트 설정
echo "자동 푸시 스크립트 설정 중..."
cat <<EOL > scripts/auto_push.sh
#!/bin/bash

# GitHub에 자동으로 변경사항 푸시
git add .
git commit -m "자동 업데이트"
git push origin main
EOL

chmod +x scripts/auto_push.sh

# 5. 프로젝트 파일 정리 및 작성
echo "프로젝트 파일 정리 및 작성 중..."
touch src/analysis/technical_indicators.py
touch src/analysis/market_sentiment.py
touch src/data/data_fetcher.py
touch src/models/trading_model.py
touch src/utils/logger.py
touch config/trading_config.yaml
touch requirements.txt
touch README.md
touch communication.md

# 6. 자동 푸시 스크립트 실행
echo "변경사항을 GitHub에 푸시합니다..."
bash scripts/auto_push.sh

echo "설정이 완료되었습니다." 