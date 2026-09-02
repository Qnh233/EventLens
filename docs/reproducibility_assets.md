# EventLens 可复现资产与模型权重说明

## 1. 第三方验收运行路径

初赛提交的唯一承诺运行路径为 **Competition CPU Lite**。该路径在启动和推理阶段不会下载任何模型权重，也不会调用外部大模型/Embedding 服务。

运行时必需且已随工程提交的模型只有：

| 模型 | 路径 | 是否必需 | 是否随包 | SHA-256 |
|---|---|---:|---:|---|
| Company event classifier | `deploy/models/company/baseline.joblib` | 是 | 是 | `64c3ed47...9b07` |
| Industry event classifier | `deploy/models/industry/baseline.joblib` | 是 | 是 | `4730b9b5...0b04` |

完整哈希见 `deploy/models/manifest.json`，运行时依赖声明见 `deploy/runtime_manifest.json`。

## 2. BGE-M3 与大模型边界

`BAAI/bge-m3` 用于研发阶段的主体/事件候选、语义聚合、Gold exemplar、Triplet 等实验。固定 revision 为 `5617a9f61b028005a4858fdac845db406aefb181`。这些路线没有取代最终 CPU Lite 生产路径，因此 **BGE-M3 不是初赛 Docker 的运行时依赖**，第三方验收不会触发模型下载。

DeepSeek hard-case Agent 在研发阶段通过 OpenAI-compatible API 调用，并经过 harmed-case 实测后冻结为 Shadow/HITL。提交运行路径不调用该 API，因此不存在需要随 CPU Lite 提交的本地 LLM 权重。

换言之，“未随包提交 BGE/LLM 大权重”不是依赖现场下载，而是因为它们不在声明的第三方运行图中。若后续把增强模式升级为正式部署能力，则必须另行提供相应完整权重或独立受控模型服务，不能让验收环境临时下载。

## 3. 实验记录

`reports/` 作为实验决策和量化证据目录整体进入 `EventLens_Runtime.zip`，包含 production-like 稳定性、temporal backtest、主动学习预算、Triplet/Reranker 否决证据、聚类、控制安全与交付冻结记录。

2026-08-29 赛前归档已实际从 `eventlens-gpu` 拉回：远端 `reports` 156 份，其中本地主目录原先缺失的 89 份已补入；7 份训练日志已复制到 `reports/logs/`。主 `reports/` 因此同时保留本地后续 temporal/交付实验和远端 GPU 实验结果。远端代码快照也已保存为 `artifacts/remote_archive/remote_code_snapshot.tar.gz`，比对后仅发现并补回 `subject_masking.py` 及其测试两个真实漏同步源码，其余同名差异以更新的本地版本为准。

同时已归档 8 组核心 company/industry event/duplicate embeddings（约 21MB）、review packet/queue 与同步 manifest。远端 BGE-M3 Hugging Face cache 实测约 **4.3GB**，因此只记录固定 revision 与研究边界，不进入 500MB 初赛 ZIP，也不属于 CPU Lite 运行时。

远端训练日志、Embedding 资产、review queue 及远端代码快照属于**研究复现归档**，不属于 CPU Lite 启动的必要文件。为避免云实例销毁后丢失，使用：

```bash
python scripts/sync_remote_repro_assets.py
```

同步到：

```text
artifacts/remote_archive/
├─ remote_inventory.txt
├─ reports/
├─ logs/
├─ embeddings/                  # 默认只同步 8 个核心 event/duplicate embedding 目录
├─ review_packet/
├─ review_queue/
└─ remote_code_snapshot.tar.gz
```

如果确实需要把远端 BGE-M3 基础模型也保存到本地研究归档，可显式执行：

```bash
python scripts/sync_remote_repro_assets.py --include-bge
```

该选项可能产生数 GB 数据，**不会自动进入 500MB 初赛 ZIP**。

默认同步会主动跳过 `untagged_train` / `untagged_test` 等约 GB 级无标签 embedding，避免为了留档复制与部署无关的大缓存；远端 inventory 仍会记录其规模。

## 4. 完整复现口径

第三方“运行复现”只需最终 ZIP：源码、配置、两套完整 CPU 权重、事件 Schema、依赖、Dockerfile 和 smoke 均在包内。

“研究实验复现”额外依赖比赛原始训练数据、指定 BGE-M3 revision，以及部分 GPU 实验资产。项目保留完整训练/benchmark 源码与结果，但失败/Shadow 路线不会被伪装成默认生产依赖。
