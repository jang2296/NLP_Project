# Phase 3.4: Vertex AI Custom Training Job - Dockerfile
# KoELECTRA NER 학습을 위한 컨테이너 이미지

FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (빌드 도구)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 학습 스크립트 복사
COPY train_koelectra_ner.py .

# KoELECTRA 모델 사전 다운로드 (빌드 시 캐싱)
RUN python3 -c "from transformers import ElectraTokenizer, ElectraForTokenClassification; \
    ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator'); \
    ElectraForTokenClassification.from_pretrained('monologg/koelectra-base-v3-discriminator', num_labels=3)"

# 출력 디렉토리 생성
RUN mkdir -p /app/output /app/checkpoints /app/runs

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache

# Vertex AI 학습 실행을 위한 엔트리포인트
ENTRYPOINT ["python3", "train_koelectra_ner.py"]
