#!/usr/bin/env bash
set -euo pipefail

PYPI_INDEX="${PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"

echo "[EventLens] 使用国内 PyPI: ${PYPI_INDEX}"
python -m pip install --upgrade pip -i "${PYPI_INDEX}"
python -m pip install -i "${PYPI_INDEX}" modelscope huggingface_hub

echo "[EventLens] 模型下载工具安装完成。"
echo "示例: python scripts/download_models_cn.py bge-m3 --source modelscope"
