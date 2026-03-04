FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build tools are needed for packages with native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# appnope is macOS-only and fails on Linux containers.
RUN sed '/^appnope==/d' requirements.txt > requirements.docker.txt \
    && python -m pip install --upgrade pip \
    && pip install -r requirements.docker.txt

COPY . /app

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
