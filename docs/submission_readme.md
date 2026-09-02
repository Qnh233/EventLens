# EventLens 可运行成果包

EventLens 面向上市公司与行业新闻，将多源资讯转化为可追溯的事件识别、同源聚合、可信评估、生命周期追踪和分级预警结果。

## 使用的技术

- **事件识别**：字符 TF-IDF + 线性分类器，适合中文简称、数字和长尾金融术语。
- **同源聚合**：主体、时间窗口和稀疏候选边组合，避免全量两两比较。
- **可信控制**：Claim→Evidence、Evidence Gate 与 Proof-or-Stop，证据不足时停止高风险推送。
- **运行架构**：CPU Lite 为默认主链路；BGE-M3 与 DeepSeek 仅是可选研究增强，不影响标准验收。

## 核心优势

1. **可直接运行**：包内自带个股与行业两套 CPU 模型，不需要现场下载模型。
2. **低资源部署**：Python 3.11、2 核 CPU、4GB 内存即可运行，无需 GPU 或外部 API。
3. **结果可审计**：预警、证据、事件簇和生命周期之间具有可追溯引用。
4. **安全可回退**：证据不足或增强服务失败时，不允许高风险结果越权发布。
5. **工程完整**：包含完整源码、配置、模型、FastAPI Web/API、Dockerfile、测试和技术文档。

## 快速部署

### Python 方式

```bash
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-deploy.txt
python -m pip install -e .
python scripts/run_deploy_smoke.py
uvicorn eventlens.webapp:app --host 0.0.0.0 --port 8000
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install -r requirements-deploy.txt
python -m pip install -e .
python scripts/run_deploy_smoke.py
uvicorn eventlens.webapp:app --host 0.0.0.0 --port 8000
```

浏览器访问 `http://127.0.0.1:8000/`，健康检查地址为 `http://127.0.0.1:8000/health`。

### Docker 方式

```bash
docker build -t eventlens:final .
docker run --rm -p 8000:8000 eventlens:final
```

国内网络、模型拉取、服务器部署和验收命令见根目录 [`部署指南.md`](部署指南.md)。交付内容与赛题要求的逐项对应见 [`交付合规清单.md`](交付合规清单.md)。

## 运行边界

- 标准 CPU Lite 路径只使用 `deploy/models/` 中已随包提供的模型。
- 系统不依赖数据库；示例输入与运行结果使用 JSON/JSONL 文件。
- 原始赛题新闻、密钥、第三方公共类库源码和编译中间文件不在成果包内。
