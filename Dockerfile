FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements 먼저 복사 (레이어 캐시 활용)
COPY requirements.txt .

# NAS 환경(CPU only) - torch CPU 버전으로 설치
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사 (models/, testvenv/ 제외는 .dockerignore에서 처리)
COPY . .

EXPOSE 8502

CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8502", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
