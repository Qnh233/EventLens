# EventLens 初赛提交清单

根据大赛初赛要求，四项材料必须全部提交。当前工程交付状态如下。

| 材料 | 当前状态 | 推荐文件 |
|---|---|---|
| ① 精益画布 | 待按大赛附件 3 模板制作 | 后续单独完成，一页 |
| ② 可运行项目成果 | **已完成** | `submission/EventLens_Runtime.zip` |
| ③ 技术文档 | **已完成** | `EventLens_技术文档_初赛提交版.docx/pdf`；仓库源稿 `docs/technical_report.md` |
| ④ 原创性声明 | 待按大赛附件 4 模板签署 | 后续单独完成 |

## 材料②检查项

- [x] 完整核心源代码：`src/`、`tools/`、`scripts/`、`tests/`
- [x] 配置与工程设计：`configs/`、`Dockerfile`、`pyproject.toml`
- [x] 本地运行模型：`deploy/models/company/`、`deploy/models/industry/`
- [x] 模型哈希清单：`deploy/models/manifest.json`
- [x] 运行时权重/网络声明：`deploy/runtime_manifest.json`，默认推理 `runtime_model_download_required=false`
- [x] 实验过程与结果：`reports/` 整体随包
- [x] 依赖声明：`requirements-deploy.txt`、`requirements-gpu.txt`、`environment.yml`
- [x] 部署文档：`docs/deployment.md`
- [x] Web/API 演示：`eventlens.webapp:app`
- [x] 标准环境 smoke：`scripts/run_deploy_smoke.py`
- [x] 第三方冷启动：从最终 ZIP 解压后再次执行 smoke + Web tests
- [x] 不包含 `.env`、API Key、SSH 凭证
- [x] 不包含 365MB 原始新闻 Excel
- [x] BGE-M3/DeepSeek 明确标记为 research/shadow，**不属于提交 Docker 运行时依赖**；CPU Lite 不现场下载模型
- [x] ZIP 小于大赛 500MB 限制

研究阶段的远端 logs、Embedding、review 资产和代码快照使用 `scripts/sync_remote_repro_assets.py` 拉回本地归档；这些文件用于实验追溯，不影响第三方 CPU Lite 冷启动。完整说明见 `docs/reproducibility_assets.md`。

## 推荐上传方式

1. 材料①：按大赛附件 3 制作的一页精益画布 PDF/PPTX。
2. 材料②：`EventLens_Runtime.zip`。
3. 材料③：技术文档 PDF，建议同时保留 DOCX 备份。
4. 材料④：使用大赛附件 4 原创性声明，按要求签字/盖章后扫描 PDF。

不要把四项材料再次套一层包含原始数据或模型缓存的大 ZIP，避免超过 500MB。

## 第三方平台部署入口

CPU Lite：

```bash
python -m pip install -r requirements-deploy.txt
python -m pip install -e .
python scripts/run_deploy_smoke.py
uvicorn eventlens.webapp:app --host 0.0.0.0 --port 8000
```

Docker 平台：

```bash
docker build -t eventlens:final .
docker run --rm -p 8000:8000 eventlens:final
```

健康检查：`GET /health`；首页：`GET /`；分析接口：`POST /api/analyze`。
