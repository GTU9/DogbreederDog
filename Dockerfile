FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1단계: torch(CPU) 먼저 설치 — 가장 무겁고 거의 변하지 않으므로 별도 레이어로 캐시.
#   requirements.txt 가 바뀌어도 이 레이어는 재빌드/재업로드되지 않음.
RUN pip install --no-cache-dir torch==2.6.0+cpu torchvision==0.21.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 2단계: 나머지 의존성 — torch>=2.0.0 요구는 위 2.6.0+cpu 로 이미 충족되어 재설치되지 않음(이중 설치 제거)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

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
