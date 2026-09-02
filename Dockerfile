ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EVENTLENS_MODEL_ROOT=/app/deploy/models

WORKDIR /app

COPY requirements-deploy.txt pyproject.toml ./
RUN pip install --no-cache-dir --index-url ${PIP_INDEX_URL} -r requirements-deploy.txt

COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

COPY configs ./configs
COPY deploy ./deploy
COPY data/raw/事件类型_标的.json data/raw/事件类型_行业.json ./data/raw/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "eventlens.webapp:app", "--host", "0.0.0.0", "--port", "8000"]
