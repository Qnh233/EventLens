### Iteration readiness gate

`tools/audit_iteration_readiness.py` provides a fail-closed preflight for hourly award-mode iteration. It records whether the remote GPU path is reachable, whether a complete local BGE-M3 cache is available, and whether human-approved Gold exists in the review queue. The gate never reads tagged external data and prevents repeatedly launching experiments when all valid next-action inputs are unavailable.

### Article Triplet BGE 多 seed 稳定性复跑

`benchmark_article_triplet_bge_oof.py` 现支持显式 `--random-state`，用于固定 last-2 配方做 train-only duplication-safe 3-fold 多 seed 稳定性验证。默认协议不变；未传 `--allow-external` 时，即使 OOF 增益过门禁也不会读取 external tagged test。

固定 `last-2 / epochs=3 / lr=1e-5 / margin=0.08 / top3 / fusion=0.2` 的 clean seed=`17/42/73` 均未触碰 external，三次 OOF 增益分别为 `+0.012949/+0.005299/+0.005620`；平均增益=`+0.007956`、std=`0.003533`、最差增益仍为 `+0.005299`。三 seed challenge 聚合进一步显示：`ambiguous_subject` 平均 `+0.018806`、`long_tail_source` `+0.006229`、`long_text` `+0.011650`，三者均为 `3/3` seed 正增益；`anti_subject_prior` 与 `rare_event` 分别仅 `1/3`、`2/3` seed 正向，属于不稳定切片。该证据确认 article-to-article 表征学习存在稳定 train-side 信号，但历史 full-train/CV ensemble external 仍回退，因此继续保持 shadow/研究态，不据此重新读取或调参 external；后续训练若重开，应优先解决跨 seed 不稳定切片，而不是重新搜索已经冻结的 lr/epoch/margin/fusion。统一汇总见 `reports/article_triplet_bge_multiseed_challenge_company.json`。

为针对 `rare_event` 的跨 seed 方差而不重开 lr/epoch/margin 网格，训练器新增可选 `--class-balanced-sampling`。正式开训前的 train-only 暴露度审计发现纯倒数频率会把最稀有类放大约 `7.7x`、头部类压到约 `0.13~0.15x`，与“降低方差”的目标冲突；因此预注册方案收紧为 **sqrt inverse-frequency** 正则化采样，三折最大稀有类放大约 `3.12~3.14x`、头部类约 `0.41~0.43x`，总训练步数不变。该路线仍只允许先做 duplication-safe train-only OOF/challenge 验证，未过稳定门禁前不得读取 external。审计见 `reports/article_triplet_class_balance_exposure_audit.json`。

# EventLens

> **初赛最终工程交付已进入冻结态。** 可运行成果包由 `scripts/build_submission_package.py` 生成到 `submission/EventLens_Runtime.zip`，压缩约 70MB，随包包含两套 CPU 模型、完整核心源码、配置、FastAPI Web/API、Dockerfile、依赖与部署说明，不包含原始大体积新闻、`.env`、BGE/LLM 大权重。第三方最小验证：`python scripts/run_deploy_smoke.py`；Web：`uvicorn eventlens.webapp:app --host 0.0.0.0 --port 8000`。提交总清单见 `docs/submission_checklist.md`，技术文档源稿见 `docs/technical_report.md`。

> 冲奖版当前严格 production-like company 最优 external Macro-F1=`0.772131`。标准流程中带 labeled-only 主体真值字段的 `0.803095` 不作为生产泛化成绩。article-to-article triplet BGE 固定 last-2 配方在 3 个 clean duplication-safe OOF seed 上平均增益 `+0.007956`，但历史 full-train/CV ensemble external 均回退；因此确认“train-side 有信号、跨域稳定性不足”，保持 shadow/研究态，不进入 production。

> **Release evidence gate（2026-08-30）**：当前 `0.772131` 候选的安全门禁仍满足 `unsafe_continue_rate=0`，且冻结的 duplication-safe OOF=`0.772805`；但历史 `complementary_prototype_fusion_company.json` 尚未保存 paired-bootstrap gain CI 与逐 fold candidate/baseline 明细，因此 `tools/audit_competition_release_evidence.py` 当前判定为 `conditional_hold`，缺失项为 `paired_oof_bootstrap_present / fold_stability_present / all_folds_positive`。这不否定当前最优点估计，而是明确限制其证据等级；待完整 BGE-M3 环境恢复后应先补齐 train-only OOF stability audit，再决定是否提升发布候选等级，过程中禁止再次用 external 调参。

> **最终提交冻结（2026-08-28）**：最后冲刺的 HistGradientBoosting candidate reranker 虽 train OOF=`0.782939`，external 仅 `0.758987`；Selective Triplet Expert 的 train-side gate 也没有超过 global triplet，最终 external=`0.767359`。因此停止无新增 Gold 的 GPU 搜索，保留 `0.772131` 为真实 production-like 最优。0.85 不虚报为当前成绩：当前 `disagreement + class balance` 人工复核 acquisition 的 15%=229 / 20%=305 条预算 oracle 分别为 `0.861547/0.887480`，作为受治理数据飞轮下一阶段的量化演进路径。

> **最新 train-only temporal 结论（2026-08-29）**：严格 chronology + duplication-safe rolling backtest 已确认 temporal/domain drift 是主要泛化瓶颈。100% review-window approved Gold refresh 五窗口 `5/5` 正增益、平均 `+0.037004`；50%/75% 时间覆盖均为 `4/5` 正向。recent-75% history truncation 已被 `0/5` 正向、平均 `-0.060727` 明确否决；“保留全历史 + 最新 25% history 重复一次”虽达到 `4/5` 正向、平均 `+0.012326`，但与 50% Gold coverage 没有稳定互补性。进一步尝试按 duplication group 仅审批 1 篇代表文再传播 Gold：五窗口平均 gain=`+0.035619` 但仍只有 `4/5` 正向，而且平均仍需 `97.9%` 的人工动作；全 train `96.66%` 行属于 singleton group，且 multi-member duplication group 中 `87.5%` 含多个 event label，因此 **duplication_id 只能用于泄漏控制/同源处理，不能作为事件 Gold 自动传播键**。本地已有完整 frozen BGE-M3 train embeddings 后，真实 `SVC vs BGE Top1 Gold exemplar disagreement + class balance` 已按固定五窗口、固定 20% review budget 完成复验：future gain=`+0.002511/+0.016700/-0.007155/+0.017331/-0.001826`，平均 `+0.005512`、仅 `3/5` 正向，未达到 harmed=0。进一步只做一个预声明的 `50% disagreement + 50% predicted-class temporal coverage` 混合单点后，五窗口 gain=`-0.014176/+0.051263/-0.003525/-0.007767/-0.019032`，平均仅 `+0.001353`、中位数 `-0.007767`、仅 `1/5` 正向；说明 temporal coverage 并不能补救当前 frozen-BGE disagreement 的跨期不稳定，反而削弱错误富集。该 hybrid 与纯 disagreement 均冻结，不继续配比/budget/margin/Top-K 网格；当前最稳主线仍是“保留历史 + 持续补充近期逐条 approved Gold”。

面向上市公司的事件驱动智能识别、可信评估、生命周期追踪与受治理持续学习系统。

当前交付版本：**v0.7.0 最终赛题冻结版**。在 v0.6.0 可信控制闭环上新增真实 DeepSeek V4-Pro hard-case Agent，但严格采用 shadow/HITL 模式，不自动覆盖 baseline；最终不再继续无 Gold 的模型动物园搜索。赛题交付总览见 `reports/competition_delivery_readiness.md`，冻结依据见 `reports/competition_freeze_decision.md`。

## 当前边界

- 做：赛题 Excel 读取、数据画像、稀疏线性事件识别 baseline、情感方向、同源聚合、可信度评分、严重性评分、生命周期/证据演化、预警 JSON、受治理数据飞轮。
- 暂不做：Neo4j、MinerU、VLM、外部数据接口接入。

决策理由：先保证 B 榜可评测主链路和答辩可解释输出，避免第一版被重型工程组件阻塞。

## Conda 环境

```bash
conda env create -f environment.yml
conda run -n eventlens python -m pip install -e .
conda run -n eventlens pytest
```

如果环境已存在：

```bash
conda env update -n eventlens -f environment.yml --prune
conda run -n eventlens python -m pip install -e .
```

### 4090D / CUDA 可选环境

GPU 节点沿用同一个 `eventlens` 环境，额外安装经过实测的 CUDA Torch 与 BGE 依赖：

```bash
python -m pip install --upgrade \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements-gpu.txt
```

国内网络无法直连 Hugging Face 时，运行前设置镜像与模型缓存目录：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/large-disk/hf_cache
```

原生 GPU provider 为可选依赖，本机 Ollama 链路仍保留；默认预测不会因为没有 `sentence-transformers` 而失败。

## 配置中心

所有运行配置统一由 `configs/app.yaml` 管理，包括路径、模型参数、聚类阈值、可信度权重、生命周期状态机、学习发布门禁和泛化评测参数。配置通过 Pydantic 启动校验，未知字段或类型错误会立即失败，避免配置静默失效。

外部 LLM 的密钥与环境差异配置统一放在仓库根目录 `.env`。复制 `.env.example` 后只填写本机密钥即可；`.env` 已加入 `.gitignore`，程序启动时自动加载，且系统环境变量优先于 `.env`，方便服务器侧安全注入。当前 DeepSeek 验证默认使用 `deepseek-v4-pro`、thinking mode 与 `reasoning_effort=max`，只用于 hard-case Agent 对照实验，不会自动覆盖生产主链路。

旧的 `configs/model.yaml`、`configs/cluster.yaml`、`configs/credibility.yaml` 仅保留为历史记录，不再被运行代码读取。

外部 LLM/Agent 的密钥和运行参数集中放在仓库根目录 `.env`，模板见 `.env.example`；`.env` 已被 `.gitignore` 忽略。当前 DeepSeek V4-Pro 只作为 hard-case shadow Expert + Verifier，不自动覆盖比赛主预测。可在 `predict-assets` 后追加 `--agent-shadow --agent-max-samples 12` 生成 `agent_shadow.jsonl` 与 `agent_shadow_summary.json`，用于 HITL、补采和答辩审计。

一键赛题演示：

```bash
PYTHONPATH=src python scripts/run_competition_demo.py --limit 100
```

需要展示真实 Agent 工具循环时使用小样本 shadow，避免把所有新闻都送入高延迟推理：

```bash
PYTHONPATH=src python scripts/run_competition_demo.py --limit 20 --agent-shadow --agent-max-samples 3
```

## 数据放置

当前真实脱敏数据位于：

```text
data/raw/news_with_tags_train.xlsx
data/raw/news_with_tags_test.xlsx
data/raw/news_without_tags_train.xlsx
data/raw/news_without_tags_test.xlsx
data/raw/事件类型_标的.json
data/raw/事件类型_行业.json
```

读取器同时支持中文字段和真实赛题字段：`article_file_id`、`article_title`、`article_publish_time`、`article_source`、`content`、`trading_code`、`secu_abbr`、`industry_code`、`industry_name`、`event_name`、`event_emotion`、`event_impact_analysis`、`duplication_id`。

有标签文件包含四个工作表。`read_competition_labeled_excel` 会将其分流为：`company_event`、`industry_event`、`company_duplicate`、`industry_duplicate`，避免把重复监督混入事件分类。

重复新闻主体回填仅接受原字段、标题唯一精确命中、正文开头唯一精确命中三种可审计方式。未解析组可以作为 duplication_id 正样本，但不会进入难负样本。个股难负已覆盖多个主体；行业主体覆盖不足，暂不训练行业重排器。

运行只读数据审计：

```bash
python tools/audit_raw_data.py --full \
  --json-output reports/raw_data_audit.json \
  --markdown-output reports/raw_data_audit.md
```

审计结论和利用方案：

- `reports/raw_data_audit.md`
- `reports/raw_data_recommendations.md`

## 常用命令

### 冲奖版：company production-like 稳定性验收

当前固定 production-like 配方为 `no-subject + char-SVC + 2400 chars + schema description x1 + exact fallback k5`。它不读取 labeled-only 的 `entity/trading_code/industry` 真值字段。5 个固定 seed 的 3-fold OOF Macro-F1 均值=`0.755864`、std=`0.009098`；冻结后 external Macro-F1=`0.770798`，2000 次 bootstrap 95% CI=`[0.715566,0.796814]`。因此当前版本仍不能宣称稳定超过 0.80；305 条 class-balanced disagreement 人工 Gold 队列仍是下一阶段最高价值输入。

```bash
HF_HOME=/root/autodl-tmp/hf_cache TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
PYTHONPATH=src python tools/benchmark_production_like_svc_stability.py \
  --train-embeddings-dir artifacts/embeddings/company_event_train \
  --test-embeddings-dir artifacts/embeddings/company_event_external \
  --output reports/production_like_svc_stability_company.json
```

同一冻结预测的 challenge Macro-F1：anti-subject-prior=`0.327976`、rare-event=`0.581530`、long-tail-source=`0.674466`、long-text=`0.567896`。分类器批量 transform+predict 约 `2.51ms/article`，无需 GPU 或外部 API；BGE 继续只复用既有离线向量/候选资产。

### 冲奖版：生成 company 高价值 Gold 复核队列

production-like train OOF 对比显示：单纯最低 margin 20% 的错误富集率为 `39.34%`；SVC/BGE Top-1 分歧优先可提升到 `46.89%`。进一步按 SVC 预测事件类型做轮转，再在类内按低 margin 排序，虽然错误富集率为 `45.57%`，但相同 305 条人工预算的离线 oracle Macro-F1 从 `0.88220` 提升到 `0.88748`，候选真值覆盖为 `96.39%`。因此优先使用兼顾 Macro-F1 类别覆盖的 class-balanced disagreement 队列：

人工预算 frontier 进一步表明：5%/76 条、10%/152 条、15%/229 条、20%/305 条的离线 oracle Macro-F1 分别为 `0.803728/0.833216/0.861547/0.887480`。因此若目标是以最小人工预算跨过 0.85，当前优先复核 **15%=229 条**；20%=305 条作为更高上限的第二阶段扩展，不必一开始全部标完。

```bash
PYTHONPATH=src python tools/build_company_oof_review_queue.py \
  --embeddings-dir artifacts/embeddings/company_event_train \
  --fraction 0.15 \
  --strategy disagreement_class_balanced_margin \
  --output artifacts/review_queue/company_oof_disagreement_class_balanced_15pct.jsonl \
  --report reports/company_oof_review_queue_disagreement_class_balanced_15pct.json
```

旧的 `company_oof_low_margin_20pct.jsonl` 保留为历史对照，不覆盖。采样策略只使用推理时可见的 SVC margin、BGE 候选与主体路由状态，Gold 仅用于 OOF 离线评估采样效率，不进入待复核队列。

队列只包含 baseline 预测、margin、候选事件、路由状态、provenance 与人工审批要求，不输出训练 Gold 真值。人工完成复核后，单独保存 JSONL（字段：`review_id`、原队列 `provenance_hash`、`expected_event_type`、`reviewer`、`approved`、可选 `note`），再通过严格导入门禁进入 Feedback/Data Flywheel：

为降低人工启动成本，可先把 229 条队列导出为与 5%→10%→15% frontier 对齐的 `76+76+77` 三个审阅 tranche。审阅包只包含标题、来源、时间、正文前 1200 字、baseline/candidates/margin/provenance，不包含 `event_label/entity/trading_code/industry` 等 Gold 或主体真值字段：

```bash
PYTHONPATH=src python tools/export_human_review_packet.py \
  --queue artifacts/review_queue/company_oof_disagreement_class_balanced_15pct.jsonl \
  --input data/raw/news_with_tags_train.xlsx \
  --output-dir artifacts/review_packet/company_oof_15pct \
  --batch-sizes 76,76,77 \
  --content-chars 1200
```

建议按 tranche 顺序逐批复核：完成 76 条先做一次 OOF 回流评估；若真实增益不足再扩到 152 条，最后才扩到 229 条，避免一次性承担全部标注成本。

```bash
PYTHONPATH=src python tools/import_review_feedback.py \
  --queue artifacts/review_queue/company_oof_disagreement_class_balanced_15pct.jsonl \
  --reviews data/feedback/company_oof_disagreement_class_balanced_15pct.reviewed.jsonl \
  --scope company
```

只有 `approved=true`、provenance 匹配且事件类型属于正式 schema 的记录才会生成确定性 `feedback_id`；重复导入会跳过已有反馈。反馈仍需后续 Evaluation/Release Agent 审批、评测与 rollback 门禁，不能直接自动覆盖 production 预测。

### 竞赛主路径（v0.6.0：可信控制与轻量运行控制面）

当前竞赛路径不再追加大模型或重排器：事件类型由字符 TF-IDF + SGD Logistic 负责；BGE-M3 只负责主体 Top-K、事件体系候选召回和稀疏候选边语义聚合。文章向量只离线编码一次，后续步骤复用 `vectors.npy + index.jsonl + manifest.json`，避免重复消耗 GPU。v0.6.0 在该主链路上增加 Proof-or-Stop、Claim→Evidence、挑战切片、Skill shadow/rollback 与轻量 Runtime Controller，不改变已经冻结的事件分类/聚类参数。

远端已经完成 199999 条 train 与 107889 条 test 的 BGE-M3 向量资产。完成主体路由和事件候选后，可直接运行资产驱动闭环：

```bash
PYTHONPATH=src python -m eventlens.cli predict-assets \
  --input data/raw/news_without_tags_train.xlsx \
  --embeddings-dir artifacts/embeddings/untagged_train \
  --routes artifacts/subject_routes/company_train.jsonl \
  --recalls artifacts/event_recall/company_train.jsonl \
  --scope company \
  --model-dir artifacts/models/company \
  --limit 5000 \
  --output-dir artifacts/run_company

PYTHONPATH=src python -m eventlens.cli validate-run \
  --input-dir artifacts/run_company \
  --output reports/run_company_validation.json
```

`predict-assets` 会在同一条可审计主链路输出：`article_event.jsonl`、`subject_route.jsonl`、`event_recall.jsonl`、`candidate_edge.jsonl`、`event_cluster.jsonl`、`cluster_decision.jsonl`、`alert_output.jsonl`、`event_lifecycle.jsonl`、`claim_evidence.jsonl`、`evidence_gate.jsonl`、`learning_signals.jsonl` 和 `run_summary.json`。`validate-run` 对文章、事件簇、预警、生命周期、Claim 与 Gate 的引用做硬校验，并量化 claim evidence coverage / gate pass rate / blocked alert；不通过时返回非零退出码。

高风险预警使用 Proof-or-Stop：至少需要 2 个独立信源，或 1 条达到门槛的官方证据，否则事件仍保留在系统中，但 `delivery_allowed=false`，并由 Runtime Controller 生成定向 `collection_requests.jsonl`。补采优先级为官方→公司→主流财经媒体；当前仓库只生成可审计补采任务，不在无数据源授权情况下主动抓取外网，后续 MCP/API Collector 直接消费该队列。

轻量运行控制面：

```bash
PYTHONPATH=src python -m eventlens.cli benchmark-control-safety \
  --output reports/control_safety_benchmark.json

PYTHONPATH=src python -m eventlens.cli runtime-plan \
  --input-dir artifacts/run_company \
  --queue-depth 800 \
  --active-workers 1 \
  --output artifacts/runtime/runtime_plan.json \
  --collection-output artifacts/runtime/collection_requests.jsonl

PYTHONPATH=src python -m eventlens.cli benchmark-trust-controls \
  --output reports/trust_control_benchmark.json
```

`runtime-plan` 只承担控制面决策：队列高水位扩容、低水位缩容、单次失败退避重试、进程退出重启、连续失败 degraded/stop、依赖失败 fallback、证据不足触发补采。它不执行金融推理，也不把 EventLens 改造成重型 Agent 平台；部署层可把这些确定性动作绑定到 systemd/K8s/RocketMQ 等执行器。

当前模型冻结依据：外部有标签测试中，TF-IDF 事件分类 Macro-F1 为个股 `0.758846`、行业 `0.836283`；BGE 固定向量 + Logistic 分别只有 `0.671306/0.780152`，因此不替换主分类器。BGE routed event Top-3 覆盖率为个股 `0.901832`、行业 `0.918440`，保留为候选召回、审计与 HITL 信号，不自动覆盖分类结果。暂不启动 Qwen 3B/7B、LightGBM 重排或 BGE 微调。

同源聚合固定使用 `7 天 + Top-20` 稀疏候选，不做全量两两比较。跨工作簿外部重复新闻评测：个股候选边 recall=`1.000000`、语义聚合 Pairwise P/R/F1=`1.000000/0.867713/0.929172`；行业候选边 recall=`0.980159`、Pairwise P/R/F1=`0.994723/0.748016/0.853907`。行业语义聚合存在约 `1.2%` overmerged article，业务稳健模式继续视为 shadow 证据；竞赛评测路径保留其 Pairwise F1 增益。

5000 篇双路 smoke 均通过 `validate-run`：公司约 79 秒、98680 条候选边、最大簇 8；行业约 78 秒、99579 条候选边、最大簇 8。该结果已经满足当前工程规模验证，不再追加更大 smoke 或参数网格。

生成数据画像：

```bash
conda run -n eventlens python -m eventlens.cli profile --input data/raw/train.xlsx --output reports/data_profile.md
```

训练个股事件 baseline：

```bash
conda run -n eventlens python -m eventlens.cli train \
  --input data/raw/news_with_tags_train.xlsx \
  --sheet-name 个股新闻 \
  --model-dir artifacts/models/company
```

训练行业事件 baseline：

```bash
conda run -n eventlens python -m eventlens.cli train \
  --input data/raw/news_with_tags_train.xlsx \
  --sheet-name 行业新闻 \
  --model-dir artifacts/models/industry
```

预测并输出事件、聚合、预警、生命周期和学习信号：

```bash
conda run -n eventlens python -m eventlens.cli predict --input data/raw/news_without_tags_test.xlsx --model-dir artifacts/models/company --output-dir artifacts/run
```

在规则近邻候选上启用本机 Ollama BGE-M3 语义复核：

```bash
PYTHONPATH=src python -m eventlens.cli predict \
  --input data/raw/news_without_tags_test.xlsx \
  --model-dir artifacts/models/company \
  --output-dir artifacts/run_semantic \
  --semantic-cluster
```

默认不开启。开启后，现有规则达到阈值的聚合结果保持不变；BGE 只处理同主体、7 天内、规则分数位于候选区间的 Top-K 文章对。服务失败时按 `fail_open` 回退规则结果。向量缓存在 `artifacts/cache/bge_embeddings.sqlite3`，聚合决策写入 `cluster_decision.jsonl`。

无训练模型时可用规则启发式跑通链路：

```bash
conda run -n eventlens python -m eventlens.cli predict --input data/raw/test.xlsx --output-dir artifacts/run
```

生成真实泛化评测报告（时间/公司/来源切分，包含 Accuracy、Macro-F1、分类指标和混淆矩阵）：

```bash
conda run -n eventlens python -m eventlens.cli generalization-report --input data/raw/train.xlsx --output reports/generalization_eval.md
```

构造小样本同源文章对：

```bash
PYTHONPATH=src python -m eventlens.cli build-duplicate-pairs \
  --scope company \
  --max-pairs 200
```

默认输出 `artifacts/training_pairs/company_pairs.jsonl`。正样本来自同一 `duplication_id`，难负样本来自 7 天内标题相似但不同组的文章。

调用本机 Ollama `bge-m3` 做主体约束候选事件召回：

```bash
PYTHONPATH=src python -m eventlens.cli recall-events \
  --scope company \
  --limit 20 \
  --top-k 3
```

召回层只输出 Top-K 候选，不直接覆盖最终分类结果。验证留痕见 `reports/duplication_retrieval_validation.md`。

使用原生 BGE-M3 分块导出可续跑 float32 向量：

```bash
PYTHONPATH=src python -m eventlens.cli encode-embeddings \
  --input data/raw/news_without_tags_train.xlsx \
  --output-dir artifacts/embeddings/untagged_train
```

输出为 `vectors.npy + index.jsonl + manifest.json`。每个 chunk 完成后刷新 manifest；中途退出后重复执行同一命令会从 `completed_count` 继续，不使用体积约 4.5 倍的 JSON 向量存储。

在已导出的文章向量上做主体候选路由：

```bash
PYTHONPATH=src python -m eventlens.cli route-subjects \
  --input data/raw/news_without_tags_train.xlsx \
  --embeddings-dir artifacts/embeddings/untagged_train \
  --scope company \
  --output artifacts/subject_routes/company.jsonl

PYTHONPATH=src python -m eventlens.cli route-subjects \
  --input data/raw/news_without_tags_train.xlsx \
  --embeddings-dir artifacts/embeddings/untagged_train \
  --scope industry \
  --output artifacts/subject_routes/industry.jsonl
```

主体路由遵循“可拒识”原则：公司只有唯一精确别名，或 BGE top1 同时通过 score/margin 门禁时才硬路由；其余仅保留 Top-3。行业外部 Top-3 召回稳定，但未找到 95% Precision 的硬路由门禁，因此默认不强制归属行业。

在主体路由结果上生成“共享主体候选 + 7 天窗口”的可审计候选边：

```bash
PYTHONPATH=src python -m eventlens.cli build-candidate-edges \
  --input data/raw/news_without_tags_train.xlsx \
  --routes artifacts/subject_routes/company_train.jsonl \
  --output artifacts/candidate_edges/company_train.jsonl
```

候选边通过主体倒排桶、时间滑窗和 `cluster.top_k` 上限生成，不做全量两两比较；每篇文章最多保留固定数量的历史候选，优先保留双方保守主体分数更高、时间更接近的文章对，避免行业 Top-3 等宽主体在 7 天窗口内产生近似平方级边数。若同一文章对共享多个 Top-K 主体，只保留一条边并记录全部共享主体。主体置信度取双方对该主体分数的较小值，再在共享主体中取最大值，避免单边高分掩盖另一侧不确定性。

候选层上线前必须用有标签 `duplication_id` 做召回门禁：同时记录全部正对召回率与“7 天策略内可召回正对”召回率，只以后者作为候选生成器门禁；默认要求 `candidate_edge_evaluation.minimum_eligible_recall >= 0.95`。这样避免把本就超出产品时间窗的正对误判为候选算法漏召回。

门禁通过独立 CLI 产出 JSON 报告；不达标时返回非零退出码，便于远端流水线直接停止后续聚类：

```bash
PYTHONPATH=src python -m eventlens.cli evaluate-candidate-edges \
  --input data/raw/news_with_tags_train.xlsx \
  --sheet-name 个股重复 \
  --edges artifacts/candidate_edges/company_duplicate.jsonl \
  --output reports/candidate_edge_recall_company.json
```

该命令不负责“补救性”降低阈值。若 eligible recall `< 0.95`，应优先检查主体候选覆盖、`cluster.top_k` 与候选排序，再重新评测。

候选层通过后，可在有标签 duplication_id 数据上评测保守聚类：仅当候选边两端的事件 Top-1（主体代码 + 事件名）一致，且已导出文章 BGE 向量余弦相似度达到 `cluster.semantic.similarity_threshold` 时合并；其余保持分离，并可输出逐边审计决策。

```bash
PYTHONPATH=src python -m eventlens.cli evaluate-candidate-clusters \
  --input data/raw/news_with_tags_train.xlsx \
  --sheet-name 个股重复新闻 \
  --embeddings-dir artifacts/embeddings/company_duplicate_train \
  --edges artifacts/candidate_edges/company_duplicate_train.jsonl \
  --recalls artifacts/event_recall/company_duplicate_train.jsonl \
  --output reports/candidate_cluster_company.json \
  --decisions-output artifacts/cluster_decisions/company_duplicate_train.jsonl
```

该评测只提供候选聚类证据，不自动切换生产默认策略。公司聚类仍必须满足既有“Pairwise Precision 不低于规则 baseline、过合并不增加，且 B-cubed F1 提升 >=0.02 或 Pairwise Recall 提升 >=0.05”发布门禁；缺少同口径 baseline 对照时保持影子模式。

从主体路由与事件 Top-K 召回中生成难例池：

```bash
PYTHONPATH=src python -m eventlens.cli build-hard-examples \
  --routes artifacts/subject_routes/company_train.jsonl \
  --recalls artifacts/event_recall/company_train.jsonl \
  --output artifacts/hard_examples/company_train.jsonl
```

难例只收集可解释的不确定样本：主体拒识、主体 top1 margin 过小、事件 top1/top2 margin 过小或事件无候选；阈值统一位于 `configs/app.yaml -> hard_examples`。其中只有该 scope 实际启用了 hard-route 时，“主体拒识”才作为难例信号；例如行业当前按外部门禁主动关闭 hard-route，因此不会把策略性拒识误当成模型错误。该产物用于后续人工标注/轻量微调价值评估，不直接回流生产模型。

运行受治理学习周期（只有离线指标达标且人工审批后的 Skill 才会激活）：

```bash
conda run -n eventlens python -m eventlens.cli learning-cycle \
  --feedback data/feedback/feedback.jsonl \
  --baseline-metrics artifacts/metrics/baseline.json \
  --candidate-metrics artifacts/metrics/candidate.json \
  --human-approved \
  --approved-by analyst-01
```

预测阶段会自动读取 `artifacts/skills/registry.jsonl` 中的 ACTIVE Skill。任何 Skill 修正都会写入文章预测的 `applied_skill_ids` 和 `decision_trace`。

同源相似度当前决策：使用 Ollama `bge-m3` 余弦相似度作为主分数，标题相似度和时间差保留为审计特征。三特征逻辑回归在单次外部实验有小幅提升，但五种子稳定性未通过，因此没有接入生产链路。详见：

- `reports/duplicate_pair_subject_resolution.md`
- `reports/duplicate_pair_reranker_decision.md`

构造同源文章对并比较标题相似度与本机 BGE-M3：

```bash
PYTHONPATH=src python -m eventlens.cli build-duplicate-pairs \
  --scope company \
  --max-pairs 200

PYTHONPATH=src python -m eventlens.cli evaluate-duplicate-pairs \
  --input artifacts/training_pairs/company_pairs.jsonl \
  --output reports/duplicate_pair_benchmark_company.json
```

小样本跨文件结果显示：BGE-M3 在个股和行业上均比标题阈值获得更高外部 F1，详见 `reports/duplicate_pair_similarity_benchmark.md`。当前只将 BGE 作为同源候选特征，尚未直接修改生产聚类策略，也暂不引入 LightGBM。

## 生命周期与数据飞轮

- 生命周期账本：`artifacts/ledger/event_lifecycle.jsonl`，只追加不覆盖。
- 证据演化：每次新增报道都记录来源类型、独立信源、立场、权威分与可信度快照。
- 难例发现：多证据下仍低置信、证据冲突、官方澄清、高影响事件会生成 `learning_signals.jsonl`；单来源低可信只继续观察，不直接污染学习池。
- Skill 沉淀：重复人工反馈形成候选 Skill，经过 Macro-F1、关键错误率和人工审批门禁后才能激活。
- 可回放：生命周期、Skill 注册、审批人、来源反馈和模型决策轨迹均有留痕。

## 留痕位置

- 环境：`environment.yml`
- 统一配置中心：`configs/app.yaml`
- 实验记录：`reports/experiment_log.md`
- 数据画像：`reports/data_profile.md`
- 错误分析：`reports/error_analysis.md`
- 泛化评测：`reports/generalization_eval.md`
- 生命周期与飞轮架构决策：`reports/architecture_traceability.md`
- 训练算力规划：`reports/compute_resource_plan.md`
- 同源文章对与主体约束召回：`reports/duplication_retrieval_validation.md`
- BGE 候选精排接入：`reports/semantic_cluster_integration.md`
- 4090D 实测与全量执行计划：`reports/remote_gpu_execution_plan.md`
- 竞赛冻结决策与最终保证实验：`reports/competition_freeze_decision.md`
- v0.6.0 Proof-or-Stop / 动态调度 / 挑战切片量化：`reports/v060_trust_runtime_experiments.md`
- Review tranche 稳定性前沿：`reports/company_temporal_review_tranche_operational_frontier.json`。固定三批人工审阅包对应 33% / 67% / 100% 累计审批节奏；train-only 五窗口结果显示 33% 仅 2/5 正向、67% 仅 4/5 正向，只有 100% 达到 5/5 正向，因此禁止“第一批/第二批一完成就直接提升生产模型”，部分 tranche 只能进入 shadow 验证，需继续累计 reviewed Gold 并通过 harmed=0 门禁。
- Triplet class-balanced sampler 多样性审计：`reports/article_triplet_sampler_diversity_audit.json`。在 seed=42 duplication-safe 三折、固定 3 epochs 下，sqrt inverse-frequency 的采样 ESS 约为训练集的 78%–79%，明显高于 full inverse-frequency 的 42%–46%；期望唯一 anchor 覆盖约 89.5%–89.9%，也高于 full inverse-frequency 的 75.7%–76.5%。因此远端 GPU 恢复后只保留 sqrt-inverse 单点进入 seed=42 train-only OOF/challenge，full-inverse 不再训练。
