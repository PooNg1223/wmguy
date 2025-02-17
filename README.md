# WM Guy

스스로 학습하는 트레이딩 AI 모델 프로젝트입니다. 이 프로젝트는 데이터를 수집하고, 기술적 분석을 통해 매수/매도 타점을 구분하며, 개선점을 스스로 고치는 기능을 포함합니다.

## 주요 기능

- 데이터 수집
- 기술적 지표 분석
- 시장 심리 분석
- 매수/매도 신호 생성
- 자동화된 GitHub 푸시

## 설치 및 실행

1. 저장소 클론:
   ```bash
   git clone <your-github-repo-url>
   ```

2. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

3. 설정 파일 수정:
   - `config/trading_config.yaml` 파일을 수정하여 API 키와 설정을 입력합니다.

4. 스크립트 실행:
   ```bash
   bash setup_project.sh
   ```

## 사용 방법

- `src/` 폴더 내의 스크립트를 사용하여 데이터 수집 및 분석을 수행합니다.
- `scripts/auto_push.sh`를 통해 변경사항을 GitHub에 자동으로 푸시합니다.

