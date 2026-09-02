#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-eventlens:final}"
PYPI_INDEX="${PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"

# Docker Hub 的 python:3.11-slim 由 Docker daemon 中配置的阿里云镜像
# 加速器代理拉取；这样 Dockerfile 本身仍保持第三方环境可移植。
echo "[EventLens] Docker image: ${IMAGE_TAG}"
echo "[EventLens] Python packages: ${PYPI_INDEX}"
echo "[EventLens] Docker Hub pull 将使用 daemon.json 中的国内 registry-mirrors。"

docker build \
  --build-arg PYTHON_IMAGE="${PYTHON_IMAGE:-python:3.11-slim}" \
  --build-arg PIP_INDEX_URL="${PYPI_INDEX}" \
  -t "${IMAGE_TAG}" \
  .
