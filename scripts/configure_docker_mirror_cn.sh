#!/usr/bin/env bash
set -euo pipefail

# 阿里云 ACR 的 Docker Hub 加速地址通常是账号/地域相关的专属地址，
# 请从阿里云控制台 -> 容器镜像服务 ACR -> 镜像工具 -> 镜像加速器复制。
ALIYUN_MIRROR="${1:-${ALIYUN_DOCKER_MIRROR:-}}"

if [[ -z "${ALIYUN_MIRROR}" ]]; then
  cat >&2 <<'EOF'
缺少阿里云 Docker 镜像加速地址。

请先在阿里云 ACR 控制台复制你的专属镜像加速地址，例如：
  https://<你的专属ID>.mirror.aliyuncs.com

然后执行：
  sudo bash scripts/configure_docker_mirror_cn.sh 'https://<你的专属ID>.mirror.aliyuncs.com'

或：
  export ALIYUN_DOCKER_MIRROR='https://<你的专属ID>.mirror.aliyuncs.com'
  sudo -E bash scripts/configure_docker_mirror_cn.sh
EOF
  exit 2
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "请用 sudo/root 执行，以便写入 /etc/docker/daemon.json" >&2
  exit 2
fi

mkdir -p /etc/docker
if [[ -f /etc/docker/daemon.json ]]; then
  cp /etc/docker/daemon.json "/etc/docker/daemon.json.eventlens.bak.$(date +%Y%m%d%H%M%S)"
fi

python3 - "${ALIYUN_MIRROR}" <<'PY'
import json
import sys
from pathlib import Path

path = Path("/etc/docker/daemon.json")
mirror = sys.argv[1].rstrip("/")
payload = {}
if path.exists():
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"现有 {path} 不是合法 JSON，请先人工修复: {exc}")

mirrors = [mirror]
for item in payload.get("registry-mirrors", []):
    if item.rstrip("/") != mirror:
        mirrors.append(item)
payload["registry-mirrors"] = mirrors
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(path.read_text(encoding="utf-8"))
PY

if command -v systemctl >/dev/null 2>&1; then
  systemctl daemon-reload || true
  systemctl restart docker
else
  service docker restart
fi

echo "[EventLens] Docker 镜像加速配置完成。"
docker info 2>/dev/null | sed -n '/Registry Mirrors/,+5p' || true
