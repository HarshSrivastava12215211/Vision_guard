# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# copy
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy app
COPY app /app/app
COPY src /app/src
# Create models directory (you must add weights here or mount them)
RUN mkdir -p /app/models

EXPOSE 8501

CMD ["gunicorn", "app.server:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--worker-tmp-dir", "/dev/shm", "--timeout", "120"]
