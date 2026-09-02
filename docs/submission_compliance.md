# EventLens 交付合规清单

本清单对应赛题对“可运行的项目成果”的要求，检查对象为 `EventLens_Runtime.zip`。

| 赛题要求 | 状态 | 包内证据 |
| --- | --- | --- |
| 团队开发的全部源代码 | 满足 | `src/`、`scripts/`、`tools/`、`tests/` |
| 可运行程序与系统演示包 | 满足 | `README.md`、`scripts/run_deploy_smoke.py`、`src/eventlens/webapp.py` |
| 相关工程设计文件 | 满足 | `configs/`、`docs/technical_report.md`、`Dockerfile`、`deploy/runtime_manifest.json` |
| 运行所需模型文件 | 满足 | `deploy/models/company/baseline.joblib`、`deploy/models/industry/baseline.joblib` |
| 模型清单与完整性校验 | 满足 | `deploy/models/manifest.json` 与 `deploy/runtime_manifest.json` 保存 SHA-256 |
| 安装包 | 满足 | `EventLens_Runtime.zip`；解压后可直接安装依赖并运行 |
| 数据库文件 | 不适用 | 系统不依赖数据库；示例输入在 `deploy/sample_articles.json`，结果使用 JSON/JSONL |
| 运行环境配置要求 | 满足 | `部署指南.md`、`requirements-deploy.txt`、`environment.yml`、`Dockerfile` |
| 标准环境直接运行和测试 | 满足 | Python 3.11 CPU 环境执行 `python scripts/run_deploy_smoke.py` |
| 不包含编译中间文件 | 满足 | 打包脚本排除 `__pycache__`、`.pyc`、`.pyo`、构建缓存和临时验证目录 |
| 不包含开源软件源或公共类库代码 | 满足 | 不打包 `node_modules`、`site-packages`、`.venv` 或 vendor 目录；仅通过依赖文件声明第三方库 |
| 不包含敏感信息和原始大数据 | 满足 | 不包含 `.env`、API Key、原始新闻 Excel；只保留公开 Schema 与演示样本 |

## 默认运行边界

- CPU Lite 的两套本地模型已经随包提供，启动阶段无需下载模型。
- BGE-M3 与 DeepSeek 为可选研究增强，不属于第三方验收的必需运行依赖。
- 无 GPU、无外网 API 时仍可完成 smoke、Web/API 启动与示例推理。

## 建议验收顺序

```bash
python -m pip install -r requirements-deploy.txt
python -m pip install -e .
python scripts/run_deploy_smoke.py
python -m pytest -q
```

打包后还应核对 `submission/package_manifest.json` 中的压缩包大小、SHA-256、模型数量、禁止项和合规检查字段。

## WSL Docker 实测记录（2026-08-29）

- 环境：Ubuntu 24.04.4 LTS，Docker client/server 29.6.2，x86_64 overlayfs。
- 构建：从正式 `submission/EventLens_Runtime` 目录成功构建 `eventlens:submission-verify`。
- 容器：Docker Healthcheck 状态为 `healthy`。
- 健康检查：`GET /health` 返回 HTTP 200、`models_ready=true`、版本 `0.7.0-final`。
- 推理接口：`POST /api/analyze` 返回 HTTP 200，并产生文章事件、事件簇和预警摘要。
- 清理：临时验收容器已删除；验证镜像保留，便于现场复查。
