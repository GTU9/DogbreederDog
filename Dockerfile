FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements 먼저 복사 (레이어 캐시 활용)
COPY requirements.txt .

# 1단계: requirements.txt 설치 (torch 제외 - 아래에서 별도 설치)
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2단계: NAS(CPU only) - requirements.txt의 torch를 덮어써서 v2.6+ 보장
RUN pip install --no-cache-dir torch==2.6.0+cpu torchvision==0.21.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 앱 소스 복사 (models/, testvenv/ 제외는 .dockerignore에서 처리)
COPY . .

ARG APP_PORT=8502
ARG BASE_PATH=dogbreeder
ENV APP_PORT=${APP_PORT}
ENV BASE_PATH=${BASE_PATH}

EXPOSE ${APP_PORT}

CMD ["sh", "-c", "python -m streamlit run app.py \
     --server.port=${APP_PORT} \
     --server.baseUrlPath=${BASE_PATH} \
     --server.address=0.0.0.0 \
     --server.headless=true"]
