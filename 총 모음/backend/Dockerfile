# Backend Dockerfile for K-Euphemism Detector API

FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY . .

# 모델 디렉토리 생성
RUN mkdir -p /app/models

# KoELECTRA 토크나이저 다운로드 (캐싱)
RUN python -c "from transformers import ElectraTokenizer; \
    ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')"

# 포트 노출
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 실행 명령
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
