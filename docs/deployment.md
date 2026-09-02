# EventLens 部署指南

## 1. 先选择运行模式

### 标准验收：CPU Lite

评委或第三方验收应使用此模式。运行包已经包含：

- 个股事件分类模型：`deploy/models/company/baseline.joblib`
- 行业事件分类模型：`deploy/models/industry/baseline.joblib`
- 完整源码、配置、事件 Schema、FastAPI Web/API 和 smoke 脚本

**CPU Lite 默认不需要下载模型**，也不需要 GPU、Ollama、DeepSeek API 或 BGE-M3。`deploy/runtime_manifest.json` 中 `runtime_model_download_required=false` 是该边界的机器可读声明。

### 可选研究增强

只有在复现实验或单独部署语义增强服务时，才需要下载 BGE-M3 等大模型：

- `bge-m3`：用于可选语义召回、聚类和 Gold exemplar 增强，约 4.6GB。
- `bge-reranker-v2-m3`：仅用于历史候选重排实验复现，约 2.3GB。
- DeepSeek 是外部 Shadow/HITL API，不是本地模型下载项，也不会自动覆盖 CPU Lite 结果。

下载大模型不会自动切换标准 CPU Lite 服务；增强能力必须按相应实验命令显式启用。

## 2. 标准运行环境

- 操作系统：Windows 10/11、主流 Linux 或 macOS
- Python：3.11
- CPU：2 核及以上
- 内存：4GB 及以上，建议 8GB
- 磁盘：CPU Lite 至少 1GB；研究增强另预留 8GB 以上
- Docker：可选
- 数据库：不需要；示例输入和运行结果使用 JSON/JSONL 文件

## 3. Python 部署

在解压后的 `EventLens_Runtime` 根目录执行。

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-deploy.txt
python -m pip install -e .
python scripts/run_deploy_smoke.py
uvicorn eventlens.webapp:app --host 0.0.0.0 --port 8000
```

### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-deploy.txt
python -m pip install -e .
python scripts/run_deploy_smoke.py
uvicorn eventlens.webapp:app --host 0.0.0.0 --port 8000
```

验收地址：

- Web 页面：`http://127.0.0.1:8000/`
- 健康检查：`GET http://127.0.0.1:8000/health`
- 分析接口：`POST http://127.0.0.1:8000/api/analyze`

## 4. Docker 部署

```bash
docker build -t eventlens:final .
docker run -d --name eventlens-demo --restart unless-stopped -p 8000:8000 eventlens:final
curl http://127.0.0.1:8000/health
```

查看状态：

```bash
docker ps --filter name=eventlens-demo
docker logs --tail 50 eventlens-demo
```

停止并删除容器：

```bash
docker rm -f eventlens-demo
```

## 5. 国内网络加速

### 5.1 Python 依赖

临时使用阿里云 PyPI：

```bash
python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements-deploy.txt
python -m pip install --no-deps -e .
```

也可设置当前环境的默认源：

```bash
python -m pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

Dockerfile 默认使用阿里云 PyPI，可通过构建参数覆盖：

```bash
docker build -t eventlens:final \
  --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  .
```

### 5.2 Docker Hub 镜像

推荐先在 **阿里云 ACR 控制台 → 镜像工具 → 镜像加速器** 获取账号对应的专属地址。

Ubuntu/WSL：

```bash
sudo bash scripts/configure_docker_mirror_cn.sh \
  'https://<你的专属ID>.mirror.aliyuncs.com'
bash scripts/build_docker_cn.sh eventlens:final
```

脚本会备份已有 `/etc/docker/daemon.json`，追加 `registry-mirrors` 并重启 Docker。

Windows/macOS Docker Desktop 可在 **Settings → Docker Engine** 中加入：

```json
{
  "registry-mirrors": [
    "https://<你的专属ID>.mirror.aliyuncs.com"
  ]
}
```

如果没有可用的专属加速地址，可显式指定国内代理基础镜像：

```bash
docker build -t eventlens:final \
  --build-arg PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.11-slim \
  --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  .
```

## 6. 可选模型拉取脚本

以下操作**不用于标准验收**。只有需要复现 BGE 语义实验或部署增强服务时才执行。

### 6.1 先查看计划

该命令不会下载文件：

```bash
python scripts/download_models_cn.py all --dry-run
```

输出会列出模型用途、来源、保存位置、预计大小和 `runtime_required=false` 状态。

### 6.2 安装下载工具

Linux/WSL：

```bash
bash scripts/install_model_tools_cn.sh
```

Windows PowerShell：

```powershell
python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ modelscope huggingface_hub
```

### 6.3 优先从 ModelScope 下载

```bash
python scripts/download_models_cn.py bge-m3 --source modelscope
python scripts/download_models_cn.py bge-reranker-v2-m3 --source modelscope
```

默认保存到：

```text
artifacts/models/bge-m3/
artifacts/models/bge-reranker-v2-m3/
```

### 6.4 Hugging Face 国内镜像备用

```bash
python scripts/download_models_cn.py bge-m3 --source hf-mirror
```

脚本会为当前进程使用 `https://hf-mirror.com`。需要手动设置缓存时，可在运行前配置：

Linux/macOS：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/large-disk/hf_cache
```

Windows PowerShell：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = "D:\hf_cache"
```

## 7. API 调用示例

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "scope": "company",
    "articles": [{
      "title": "上市公司收到监管问询函",
      "source": "交易所公告",
      "content": "交易所要求公司就信息披露事项进一步说明并按期回复。"
    }]
  }'
```

返回结果包含文章事件、事件簇、可信预警、生命周期、Evidence Gate 与 Learning Signal。

## 8. 第三方验收

建议依次执行：

```bash
python scripts/run_deploy_smoke.py
python -m pytest -q
python -m compileall -q src scripts tools
```

验收重点：

1. `/health` 返回 `models_ready=true`；
2. `/api/analyze` 返回 predictions、clusters、alerts、lifecycles 和 evidence_gates；
3. 高风险但证据不足时 `delivery_allowed=false`；
4. `deploy/runtime_manifest.json` 中两项 `runtime_required=true` 模型均存在且哈希一致；
5. 启动和推理不下载模型、不调用外部 API；
6. 包内没有 `.env`、原始新闻、第三方公共类库源码和编译中间文件。

## 9. 常见问题

### 服务提示模型不存在

确认以下文件均存在：

```text
deploy/models/company/baseline.joblib
deploy/models/industry/baseline.joblib
```

然后执行：

```bash
python scripts/run_deploy_smoke.py
```

标准成果包已经携带这两项模型，不应通过可选大模型脚本重新下载。

### 安装速度慢

使用第 5.1 节的阿里云 PyPI；不要为 CPU Lite 安装 `requirements-gpu.txt`。

### 需要公网展示

在服务器安全组或防火墙放行 TCP `8000`，访问 `http://<服务器IP>:8000/`。正式公网环境建议另行配置 Nginx 与 HTTPS，但它们不是本次成果包运行的前置条件。
