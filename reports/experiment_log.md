## 2026-08-30 iteration readiness gate

- Checked local EventLens processes/GPU and remote `eventlens-gpu` before launching work. No local Triplet/complementary-fusion experiment was running; local RTX 3050 was desktop-occupied only. Remote SSH returned `Connection refused`, so no remote GPU experiment was started.
- Confirmed there is still no approved human-reviewed Gold in the archived review queues; all discovered review records are pending. Therefore the governed data flywheel has no legal new Gold tranche to ingest this round.
- Added `tools/audit_iteration_readiness.py` plus tests. The fail-closed gate records remote availability, local BGE-M3 cache readiness, approved-Gold availability, allowed next actions, and `external_touched=false` so future automation does not repeatedly run invalid work when resources are unavailable.
- Strict production-like external Macro-F1 remains frozen at `0.772131`; no tagged external data was read for selection or tuning in this round.
## 2026-08-30 冲奖迭代：production release-evidence gate 落地

- 前置检查：本地 RTX 3050 仅桌面占用约 `1403/4096 MiB`、util=`17%`，未启动本地 BGE 训练；远端 `eventlens-gpu` 实时 SSH 返回 `Connection refused`，无法可信核验远端 GPU/进程/reports，因此未重复启动 seed=42 sqrt-inverse Triplet。严格 production-like external best 继续冻结为 Macro-F1=`0.772131`，本轮未读取 tagged external 数据。
- 当前阻塞下优先补齐企业发布证据门禁：新增 `tools/audit_competition_release_evidence.py`，只读取冻结 reports，不加载 tagged test、不改模型/阈值/融合/候选/路由参数。门禁同时校验 frozen external=`0.772131`、duplication-safe OOF=`0.772805`、`unsafe_continue_rate=0`、paired OOF bootstrap、3-fold stability 与 all-folds-positive。
- 实际审计结果：`decision=conditional_hold`。已有分数与 runtime safety 均匹配，但历史 `reports/complementary_prototype_fusion_company.json` 没有 `oof_gain_bootstrap`、`fold_stability`、`all_folds_positive`，因此三项稳定性证据明确缺失；该结论不把 `0.772131` 降级为错误成绩，只限制其 production release evidence 等级，避免仅凭点估计升级候选。
- 新产物：`reports/competition_release_evidence_gate.json`，其中 `external_touched_by_audit=false`。新增专项测试 `2 passed`。下一断点：远端/完整 BGE-M3 环境恢复后优先运行已增强的 `benchmark_complementary_prototype_fusion.py --skip-external` 补齐 paired-bootstrap + fold stability；若 train-side 稳定性不成立，则不得因历史 external=`0.772131` 放宽门禁。Triplet 分支仍只允许 seed=42 sqrt-inverse train-only OOF/challenge 单点。
- 邮件：当前任务运行环境禁用邮件能力，因此无法安全执行 Gmail profile + send；未向任何未知地址发送邮件。

## 2026-08-29 冲奖迭代：Triplet class-balance 暴露度审计与正则化收紧

- 前置检查：本地无 EventLens/Triplet 同类模型实验；RTX 3050 Laptop 仅桌面占用约 `1444/4096 MiB`、util=`11%`。远端 `eventlens-gpu` 实时返回 `Connection refused`，无法可信读取 GPU/进程/reports，因此未启动任何远端/GPU 训练；严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external。
- 审计动机：上一轮预注册的纯 inverse-frequency anchor sampler 尚未实际训练。为避免 GPU 恢复后用一个会放大长尾噪声的采样器浪费训练轮次，本轮先在 seed=`42` 的同一 duplication-safe 3-fold train split 上做 exposure audit，仅使用 train 标签频次，不涉及 external 或模型调参。
- 发现：纯 inverse-frequency 会让每类期望总采样质量完全相同，但三个 fit fold 的最大稀有类放大分别为 `7.720x/7.720x/7.667x`，头部类只保留 `0.142x/0.145x/0.133x`；每折有 `9/12/11` 个类被放大至少 `3x`。这与当前目标“降低 rare_event/anti_subject_prior 方差”冲突，存在把极少数样本重复过多的明显过拟合风险。
- 收紧方案：不重开任何连续超参网格，直接把唯一待训单点改为 sqrt inverse-frequency 正则化采样。相同三折下最大放大降为 `3.123x/3.143x/3.122x`，头部类保留 `0.424x/0.430x/0.411x`；每折仅 `1/2/2` 个类超过 `3x`。总采样数、epochs/batch/lr/margin/last-2/top3/fusion 均保持冻结。
- 工程：`regularized_class_balance_anchor_weights()` 替代原纯 inverse-frequency helper；正式报告新增 `class_balanced_sampling_strategy=sqrt_inverse_frequency`，避免未来产物口径歧义。专项测试 `7 passed`，`compileall -q src tools` 与 `git diff --check` 通过。
- 产物：`reports/article_triplet_class_balance_exposure_audit.json`。下一断点仍为远端 GPU 恢复后只跑 seed=`42` duplication-safe 3-fold OOF + challenge；只有总体 OOF 不退化且 `rare_event/anti_subject_prior` 改善才扩 seed=`17/73`，仍禁止触碰 external。

## 2026-08-29 冲奖迭代：Triplet 三 seed challenge 稳定性聚合
- 前置检查：本地 `tasklist` 未发现 Python EventLens 实验进程；本机 RTX 3050 为桌面占用（约 `1667/4096 MiB`、23%），不承担 BGE 训练。远端 `eventlens-gpu` 本轮首次可建立 SSH 并读到最新 reports：仅 Jupyter/TensorBoard 基础 Python 进程，无 EventLens 训练进程；最新正式产物已包含 `article_triplet_bge_oof_seed73_challenge_company.json`。随后 SSH 被远端主动关闭，因此没有在 GPU 状态不可信时启动新训练。
- 新证据：seed=73 固定 `last-2 / epochs=3 / lr=1e-5 / margin=0.08 / top3 / fusion=0.2` duplication-safe 3-fold OOF baseline=`0.762884`、triplet=`0.768504`，gain=`+0.005620`，group overlap=`0`，external 未触碰。结合 seed=17/42，三 seed OOF gain=`+0.012949/+0.005299/+0.005620`，平均=`+0.007956`、std=`0.003533`、最差=`+0.005299`，3/3 均过既定 +0.005 train-side 门禁。
- challenge 稳定性：三 seed 聚合后，`ambiguous_subject` gain mean=`+0.018806`（min=`+0.013354`，3/3 正）、`long_tail_source` mean=`+0.006229`（min=`+0.002968`，3/3 正）、`long_text` mean=`+0.011650`（min=`+0.000272`，3/3 正）；这些是目前 Triplet 最稳定的 train-side受益切片。`anti_subject_prior` mean=`+0.004125` 但只有 1/3 seed 正，`rare_event` mean=`+0.003121` 且只有 2/3 seed 正，因此明确视为不稳定/潜在 harmed 切片。
- 决策：不根据三 seed 正增益重新触碰 external，也不重开 lr/epoch/margin/fusion 网格。历史 full-train external=`0.767359`、3-fold CV ensemble external=`0.768263` 的回退结论继续有效。若后续 GPU 训练重开，最高价值应是 train-only 稳定性目标：降低 `anti_subject_prior/rare_event` 方差或引入更稳健正则/冻结策略；`ambiguous_subject/long_tail_source/long_text` 仅作为稳定收益诊断，不做事件名/事件对硬编码 gating。
- 工程：增强 `tools/summarize_article_triplet_bge_multiseed.py`，自动给出 `stable_positive_slices` 与 `unstable_or_harmed_slices`，规则固定为“非空切片在所有固定 seed 上 Macro-F1 gain > 0”；对应测试 `5 passed`。新产物：`reports/article_triplet_bge_multiseed_challenge_company.json`，`all_external_untouched=true`。
- 严格 production-like external best 继续冻结为 `0.772131`。本轮没有新增 approved Gold；`data/feedback` 当前不存在，现有 15% class-balanced disagreement review queue 仍待人工审批。
- 邮件：当前运行环境禁用邮件发送能力，因此无法执行 Gmail profile + send；未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：75% temporal coverage 成本点与人工 Gold 稳定性前沿
- 前置检查：本地未发现 EventLens 同类训练/benchmark 进程；本机 RTX 3050 仅桌面占用。通过既定 `eventlens-gpu` SSH alias 访问远端仍返回 `Connection refused`，因此无法可信核验远端 GPU、seed=73 challenge、日志或新 reports，按门禁未重复启动任何 GPU/Triplet 任务。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮没有读取 external 做选择或调参。
- 设计理由：此前固定 50% temporal coverage review 为 `4/5` 正窗口、最差 gain=`-0.031451`，而 100% full-review Gold refresh 为 `5/5` 正窗口。为判断是否存在明显低于 full-review 的稳定成本点，本轮只补一个预声明的中间点 `review_fraction=0.75`；不继续做 80/85/90% 预算细网格，不使用 review/future Gold 参与选样，仍只按 predicted-class balance + 类内时间均匀覆盖。
- 五窗口结果（end=`0.65/0.75/0.85/0.95/1.00`）：future Macro-F1 gain=`-0.014331/+0.052441/+0.014079/+0.024961/+0.061426`；mean=`+0.027715`、median=`+0.024961`、positive windows=`4/5`、`all_windows_positive=false`。每窗约审批 `172/229` 条 recent Gold；group overlap=`0`，`external_touched=false`。
- 稳定性：最早窗口 harmed 从 50% 的 `-0.031451` 收窄到 `-0.014331`，但仍未满足 harmed=0。end=`0.75` 的 paired-bootstrap 95% CI=`[+0.003031,+0.088712]`、positive probability=`0.981`，成为当前 coverage 路线首个 CI 全正窗口；其余窗口 CI 仍跨 0，不能包装为全局稳定提升。
- 成本前沿：50% 平均 gain=`+0.028699`、4/5 正；75% 平均=`+0.027715`、4/5 正；100% 平均=`+0.037004`、5/5 正。75% 虽改善 worst-case harmed，但没有以更多人工成本换来更高平均 gain 或 harmed=0，因此当前已测成本点中最低稳定点仍是 100%。正式汇总见 `reports/company_temporal_review_cost_stability_frontier.json`。
- 决策：停止对同一 temporal-coverage acquisition 做预算网格；下一最高价值仍是远端/BGE 恢复后，将真正异构的 `SVC-BGE disagreement + class balance` 放进同一冻结五窗口 protocol，或导入真实 approved Gold tranche 做 rolling + duplication-safe OOF/challenge 验证。企业数据飞轮继续保持 provenance/审批/rollback/审计，不能把模拟 Gold 当真实生产成绩。
- 工程：`rolling_temporal_coverage_review_experiment/backtest` 参数化 `review_fraction` 并新增 fail-fast `(0,1]` 校验；专项测试 `20 passed`。产物：`reports/company_temporal_coverage_review_75pct_five_window.json`、`reports/company_temporal_review_cost_stability_frontier.json`。
- 邮件：当前运行环境禁用邮件发送能力，无法执行 Gmail profile + send；本轮有新实验结果但只能在项目留痕和任务通知中记录，未向未知地址发送邮件。
### 2026-08-29 50% temporal-coverage reviewed-Gold backtest (train-only)

- 运行前巡检：本地无同类 EventLens 实验进程；远端 `eventlens-gpu` 继续 SSH `Connection refused`，因此未重复启动 seed=73 challenge 或 GPU/Triplet 任务，也未假设远端 GPU 空闲。严格 production-like external best 继续冻结为 `0.772131`，本轮未读取 external 做模型选择或调参。
- 上一轮 100% review-window approved Gold refresh 在五个固定 rolling window 上 `5/5` 正增益，但人工成本过高。本轮预声明单一成本点：review window 仅复核 `50%`，选样完全不读取 review/future Gold，也不使用 margin；只按 primary 的**预测类别均衡**分配预算，再在每个预测类的时间轴上均匀取点，验证“时间覆盖”能否以半量人工成本保留 full-review refresh 的跨期收益。未搜索 budget、quota 权重或时间采样参数。
- 五窗口 future Macro-F1：`0.617653→0.586202 (-0.031451)`、`0.581998→0.636996 (+0.054998)`、`0.585091→0.620944 (+0.035853)`、`0.692952→0.703766 (+0.010813)`、`0.737818→0.811100 (+0.073281)`。平均 gain=`+0.028699`，中位数=`+0.035853`，`4/5` 窗口为正，但 `all_windows_positive=false`。
- 选中样本 primary error rate 仅约 `20.2%~29.8%`，明显低于此前 class-risk/low-margin 的错误富集，但 future 泛化反而更好；这支持“近期分布/事件边界覆盖”比单纯挑错更关键。不过最早窗口出现明确 harmed=`-0.031451`，因此该固定 50% temporal-coverage acquisition **未通过生产稳定门禁**，不能进入 production，也不继续围绕 50% 做预算细网格。
- 五个 paired-bootstrap 95% gain CI 仍全部跨 0；最强 end=0.75 gain=`+0.054998`、positive probability=`0.973`，CI=`[-0.000672,+0.095876]`；最后窗口 gain=`+0.073281`、future Macro-F1=`0.811100`，CI=`[-0.011593,+0.133206]`。这些均为 train-only temporal 内部证据，不是新的 external production-like 成绩。
- 决策：当前数据飞轮证据排序进一步清晰：100% recent Gold refresh=`5/5` 正向 > 50% temporal coverage=`4/5` 正向 > 多类“挑错” SVC-only acquisition。下一高价值断点仍是远端/BGE 资产恢复后，在同一固定 rolling protocol 中验证真正异构的 `SVC-BGE disagreement + class balance`，目标是用低于 100% 的人工成本达到 harmed=0/all-windows-positive；若仍失败，则企业侧应优先真实 approved Gold 持续覆盖，而不是继续 heuristic 搜索。
- 工程：新增 `_predicted_class_balanced_temporal_coverage_indices()`、`rolling_temporal_coverage_review_experiment/backtest()`；专项测试 `19 passed`。正式产物：`reports/company_temporal_coverage_review_50pct_five_window.json`，`external_touched=false`。
- 邮件通知：本轮运行环境 `Email enabled=False`，无法执行 Gmail profile + send；未向任何未知地址发送邮件。

### 2026-08-29 five-window reviewed-Gold robustness backtest (train-only)

- 运行前巡检：本地无同类 EventLens 实验进程；远端 `eventlens-gpu` 继续 SSH `Connection refused`，因此未重复启动 seed=73 challenge 或任何 GPU/Triplet 任务，也未假设远端 GPU 空闲。严格 production-like external best 继续冻结为 `0.772131`，本轮未读取 external 做模型选择或调参。
- 本地未发现能合法按 chronology 窗口重算真实 SVC-BGE disagreement 的 BGE embedding/cache；现有 OOF review queue 只覆盖原 OOF，不能冒充 temporal BGE 视角。为避免伪造语义分歧，本轮改做低自由度稳健性诊断：固定既有 `predicted-class-balanced low-margin + 20% review tranche`，预声明五个 rolling end fraction=`0.65/0.75/0.85/0.95/1.00`，只复验跨期稳定性，不搜索 budget/margin/模型参数。
- 五窗口 future Macro-F1 gain 依次为 `-0.008928/+0.006203/-0.030395/-0.008914/+0.040052`；均值=`-0.000396`，中位数=`-0.008914`，最差=`-0.030395`，仅 `2/5` 窗口为正，`all_windows_positive=false`。对应 baseline→reviewed-Gold future Macro-F1：`0.617653→0.608724`、`0.581998→0.588201`、`0.585091→0.554696`、`0.692952→0.684039`、`0.737818→0.777870`。
- 五个 paired bootstrap 95% gain CI 均跨 0；其中 end=0.85 的正增益概率仅 `0.1065`。因此当前“简单 low-margin + class balance + 固定 20% Gold 回流”不能作为稳定 production acquisition 策略。此前单窗口 `+0.040052` 只能视为局部信号，不能外推。
- 这不否定主动学习/Gold 主路线：原始 production-like OOF 的真实 `SVC-BGE disagreement + class balance` 仍有 15%/20% oracle=`0.861547/0.887480`。但在其 temporal BGE 视角尚未合法复验前，不能声称该 acquisition 已跨期稳定；下一高价值断点仍是远端/BGE 资产恢复后按同一 fixed rolling protocol 验证真实 SVC-BGE disagreement，或等待真实 approved Gold 做受治理 tranche 回流。
- 工程：新增 `rolling_review_robustness_backtest()`，固定五窗口/20% budget 协议；专项测试 `14 passed`。正式产物：`reports/company_temporal_review_20pct_five_window_robustness.json`。
- 邮件通知：本轮运行环境 `Email enabled=False`，无法执行 Gmail profile + send；未向任何未知地址发送邮件。

### 2026-08-29 strict temporal cutoff sensitivity (train-only)

- 运行前巡检：本地无 EventLens/Python 实验进程；远端 `eventlens-gpu` 继续 SSH `Connection refused`，因此未重复启动 seed=73 challenge 或任何 GPU 训练，也未假设远端 GPU 空闲。
- 为判断上一轮 20% temporal holdout Macro-F1=`0.741846` 是否只是单一 cutoff 偶然，复用同一严格 `temporal_group_split`，补跑 15%/25% 两个 cutoff；所有切分均 duplication group overlap=`0`、`external_touched=false`、无 unseen holdout label。
- 15% strict temporal：train=`1296`、holdout=`229`，Macro-F1=`0.822714`；20% strict temporal：train=`1220`、holdout=`305`，Macro-F1=`0.741846`；25% strict temporal：train=`1143`、holdout=`382`，Macro-F1=`0.679496`。时间窗口扩大后性能呈显著下降，说明 temporal/domain drift 风险不是单 cutoff 噪声，且早期训练覆盖不足会明显伤害长尾/相邻事件边界。
- 结构性混淆跨 cutoff 持续出现：`产能布局调整 -> 关键技术突破` 在 15%/20%/25% 分别约 `5/9/10` 条，技术研发/产品技术创新、新车发布定价等相邻事件族也反复进入弱类或 top confusion。该证据仅用于 train-side drift 诊断，不把具体事件对硬编码进 production。
- 简单 recency repeat 结论不稳定且总体不具备生产依据：15% cutoff 上 `+0.005428`，20% 上 `-0.008366`，25% 上 `-0.007327`；因此继续维持“简单近期样本重采样/权重搜索冻结”，不根据单一 15% 正增益重开参数网格。
- 人工 Gold 价值在较近窗口仍显著：15% temporal holdout 的 class-balanced low-margin 20% review oracle=`0.883656`；20% temporal holdout=`0.846204`；但 25% 更早 cutoff 仅=`0.804721`。这表明 0.85 路线需要“持续补充近期 reviewed Gold”，而不是一次性修正固定旧训练集；更符合企业数据飞轮持续采样/审批/回流的交付逻辑。
- 验证：`python -m pytest tests/test_benchmark_company_temporal_holdout.py -q` -> `5 passed`。新增产物：`reports/company_temporal_holdout_15pct_strict.json`、`reports/company_temporal_holdout_25pct_strict.json`。
- 严格 production-like external best 继续冻结为 `0.772131`，本轮未读取 external 做选择或调参。下一优先级：远端恢复后补齐 seed=73 challenge fail-closed 汇总；本地侧若无新增人工 Gold，则优先把 temporal review queue/Gold tranche 做成按时间滚动的受治理 acquisition，而不是继续模型超参搜索。
- 邮件通知：本轮运行环境明确禁用了邮件能力，无法执行“先取 Gmail profile、只向当前账号发送”的安全流程；项目迭代未因此中断，也未向任何未知地址发送邮件。

### 2026-08-28 temporal recency robustness (train-only)

- 远端 `eventlens-gpu` 仍为 SSH `Connection refused`，未重复启动 GPU/Triplet 任务。
- 严格 duplication-safe temporal 20% holdout 基线保持：train=1220、holdout=305、group overlap=0、external_touched=false，Macro-F1=0.741846。
- 单变量低自由度对照：仅依据训练发布时间，将训练集中最新 25%（305 条）样本额外重复一次；不读取 external、不用 labeled-only 主体字段。
- 结果：Macro-F1=0.733480，较 temporal baseline **-0.008366**；Accuracy 0.806557 -> 0.800000。
- 结论：简单 recency reweight/resampling 明确否决，不继续网格化 recent ratio/repeat 次数。时间泛化缺口更符合条件分布/语义漂移，需要后续优先做 train-only temporal confusion/challenge 诊断、reviewed Gold 或更稳健目标，而非样本新旧比例微调。
- 验证：`python -m pytest tests/test_benchmark_company_temporal_holdout.py -q` -> 3 passed。
- 产物：`reports/company_temporal_holdout_20pct_recency.json`；实现：`tools/benchmark_company_temporal_holdout.py`。

# 2026-08-28 最终冲刺：树重排与 Selective Triplet 均被 external 否决，正式冻结交付

- 用户要求在算力成本较高的前提下快速尝试少数高赔率手段，若仍无明显收益则以当前版本完成赛题交付。本轮遵守“不重复已有实验、先检查 GPU/进程”的门禁；开始时远端已有固定 seed=42 triplet challenge-slice train-only 任务，因此先续跑而未抢卡。
- Triplet challenge 诊断：固定 last-2 / epochs=3 / lr=1e-5 / margin=0.08 / Top3 / fusion=0.2 的 duplication-safe OOF 从 `0.767373` 到 `0.772672`（`+0.005299`）。切片增益为 ambiguous-subject=`+0.013354`、rare-event=`+0.005580`、long-tail-source=`+0.002968`、long-text=`+0.000272`、anti-subject-prior=`-0.002089`；该任务未读取 external。
- 激进方案 1：新增 `benchmark_tree_candidate_reranker.py`，用 `HistGradientBoostingClassifier` 对 SVC Top5∪BGE Top5 的 inference-visible SVC/BGE/schema-route 特征做非线性 candidate 排序；base/meta 两层均使用 `StratifiedGroupKFold`，duplication group overlap=0，仅比较两个预先固定的小自由度树配置。最佳 train OOF Macro-F1=`0.782939`，相对同折 SVC=`+0.015566`，因此触发一次 final external；external Macro-F1=`0.758987`，大幅低于当前 best `0.772131`，路线立即否决，不再按 test 调树深/权重。
- 激进方案 2：依据 challenge 证据给固定 article-triplet BGE 增加 inference-visible selective gates，只比较 `ambiguous_route / rare_predicted_event / ambiguous_or_rare`。三者 OOF Macro-F1 分别为 `0.769951/0.770654/0.769220`，均低于 global triplet=`0.772672`，因此 train-only 自动选择仍为 global；冻结配置 external=`0.767359`，相对当前 best `-0.004772`。Selective Expert 假设被否决，不继续烧 GPU。
- 最终严格 production-like company external best 冻结为 `0.772131`，来自 `char-SVC + BGE Top1 Gold exemplar + TF-IDF class centroid` 低自由度互补融合；其 duplication-safe OOF=`0.772805`，相对同折 SVC=`+0.005432`。标准带 labeled-only 主体真值的 `0.803095` 不作为 production 成绩。
- 冲奖模型结论：triplet clean 3-seed paired OOF gain 均为正且平均 `+0.007956`，树 reranker OOF 甚至 `+0.015566`，但多条复杂路线 external 均回退，证据共同指向“1525 Gold 下 domain/temporal drift > 模型容量不足”。因此无新增 reviewed Gold 时正式停止 lr/epoch/layer/reranker 等 GPU 搜索。
- 可量化的 0.85 演进路径保留为受治理人工 Gold：class-balanced disagreement acquisition 15%=229 条 oracle Macro-F1=`0.861547`，20%=305 条=`0.887480`。该 oracle 仅用于人工预算估算，不冒充自动模型成绩；review packet、provenance、人工 approval、feedback import、Evaluation/Release Agent 与 rollback 已落地。
- 最终提交故事冻结为：廉价 production classifier 处理 easy traffic → BGE/Schema 做主体与事件候选 → Claim→Evidence + Proof-or-Stop 控制金融风险 → 稀疏同源聚合与生命周期追踪 → credibility×severity 预警 → DeepSeek hard-case Shadow/HITL → 高价值 review acquisition 驱动数据飞轮。company 聚类 Pairwise F1=`0.929172`，5k 双路 smoke、控制安全 11/11、`unsafe_continue_rate=0`、Claim evidence coverage=`1.0` 已有实测证据。
- 本轮新增报告：`reports/tree_candidate_reranker_company.json`、`reports/article_triplet_bge_oof_seed42_challenge_company.json`、`reports/article_triplet_bge_selective_company.json`。下一步只做完整回归/compileall/diff-check/validate-run、更新交付文档并停止小时级 GPU 自动迭代；只有新增 reviewed Gold 才重新打开模型训练。

# 2026-08-28 冲奖版迭代：固定 production-like SVC 多 seed / bootstrap / challenge 稳定性验收

- 运行前经 `windowsdev` 检查：本地/远端均无 EventLens 实验进程；远端 RTX 4090 D=`0% / 1 MiB`，`/root/autodl-tmp` 约剩余 25GB；305 条 class-balanced disagreement review queue 仍存在，但没有新增 reviewed Gold，因此不重复已失败的 reranker/listwise/NB-SVM/文本重权实验。
- 决策：不再用 external 做任何模型选择，冻结当前 production-like 配方 `no-subject + char_wb(2,5) + 2400 chars + schema description x1 + class_weight_power=1 + C=1 + exact_fallback_k5`，补齐候选生产模型必须具备的多 seed、bootstrap CI、challenge slices 与分类器时延/调用成本证据。
- 新增 `tools/benchmark_production_like_svc_stability.py`。5 个预先固定 seed（17/29/42/73/101）分别做 3-fold train OOF，只测稳定性、不选 seed：Macro-F1=`0.739928/0.757658/0.764504/0.752832/0.764400`，均值=`0.755864`，总体标准差=`0.009098`，范围=`[0.739928,0.764504]`。
- 固定全量训练后 external 只做一次冻结复现：Accuracy=`0.799738`、Macro-F1=`0.770798`，与既有 production-like 最优完全一致。2000 次 sample bootstrap 的 Macro-F1 95% CI=`[0.715566,0.796814]`，上界仍低于 0.80；因此当前模型没有“统计上稳定超过 0.80”的证据，更不能宣称接近 0.85。
- challenge slices（同一固定 external 预测）：anti-subject-prior=`0.327976`、ambiguous-subject=`0.606715`、rare-event=`0.581530`、long-tail-source=`0.674466`、long-text=`0.567896` Macro-F1。弱项与此前错误分析方向一致，说明下一批 Gold 应优先覆盖反主体先验、稀有事件和长文本，而不是继续做字符线性边界的小修补。
- 分类器运行成本：全量 fit≈`19.18s`；764 篇 transform+predict≈`1.92s`，约 `2.51ms/article`；分类器本身不需要 GPU、外部 API 调用=`0`、API 成本=`0`。BGE 仅复用现有文章 embedding/主体与事件候选资产，本轮没有重新编码文章。
- 专项测试：本地/远端 stability + OOF + challenge 共 `8 passed`。产物：本地/远端 `reports/production_like_svc_stability_company.json`；复现命令见工具 `--help`，远端使用既有 `company_event_train/company_event_external` embedding 与离线 BGE cache。
- 企业门禁复验：远端 `benchmark-control-safety` 11/11，`action_match_rate=1.0`、`unsafe_continue_rate=0.0`；`benchmark-trust-controls` Evidence Gate 4/4、Skill governance 5/5。首次 `validate-run` 误指向不存在的历史目录 `artifacts/run_company`，只产生 `FileNotFoundError`，未修改数据；定位现存 `artifacts/competition_demo/company` 后复验 `passed=true`，20/20 article→cluster→alert→lifecycle 引用闭合、claim evidence coverage=`1.0`、unsupported high-risk claim=`0`。
- 阶段回归：本地完整 `147 passed`，`compileall -q src tools`、`git diff --check` 通过；远端完整 `143 passed`、`compileall` 通过。两端测试数量差异来自既有测试库存差异；本轮新增 stability 专项两端一致通过。结束时远端 RTX 4090 D=`0% / 1 MiB`。
- 决策：当前稳定证据进一步支持“无新增 Gold 时停止自动模型小修”。production-like 最新最优仍为 `0.770798`；下一步优先完成企业安全门禁复验与冻结证据，然后等待/获取 305 条受治理人工 Gold，再重新打开严格 train-only OOF 模型实验。

# 2026-08-28 冲奖版迭代：长文本均匀采样假设淘汰，external 未触碰

- 运行前经 `windowsdev` 检查：本地/远端均无 EventLens 实验进程；远端 RTX 4090 D=`0% / 1 MiB`，`/root/autodl-tmp` 约剩余 25GB，因此没有重复启动已失败的 reranker/listwise/LLM 自动覆盖路线。
- 当前没有新增人工 reviewed Gold。基于 production-like OOF 的 `long_text` slice Macro-F1=`0.619428` 明显弱于整体 `0.764504`，验证一个由错误切片直接驱动的低自由度假设：固定 `schema desc x1 + exact_fallback_k5 + 2400 content chars`，只比较正文 `head / head+tail / head+middle+tail` 三种通用采样，不读取 labeled-only 主体字段、不硬编码事件对。
- 新增 `tools/benchmark_long_context_svc.py` 与 3 个采样边界测试。远端首次执行因 external embedding 目录名误写为不存在的 `company_event_test` 直接失败，未进入实验；修正为已有 `company_event_external` 后又发现 `HF_HOME` 未指向远端既有 BGE 缓存，`local_files_only` 安全失败。最终显式复用 `/root/autodl-tmp/hf_cache`，未下载新模型、未重复文章 embedding。
- 严格 3-fold train OOF：`head=0.764504`、`head_tail=0.761679`、`head_mid_tail=0.756533` Macro-F1。最佳替代方案 `head_tail` 相对稳定 head baseline 仍下降 `-0.002825`，未达到预先固定的 `+0.005` external 触发门禁，因此 **external tagged test 未读取**。
- 结论：long-text slice 的弱项不能通过简单保留正文尾部/中部解决；均匀文本采样会丢失前部高价值事件上下文。该路线按 Proof-or-Stop 淘汰，不进入 production，也不继续做位置比例网格。
- 当前 company production-like 最新最优仍为 external Macro-F1=`0.770798`，尚未超过 0.80，更未达到 0.85。本轮没有修改 production 模型、阈值、候选策略、Proof-or-Stop、Claim→Evidence、DeepSeek Shadow/HITL、runtime fallback/degrade/stop 或数据飞轮 provenance/审批/rollback。
- 下一步：没有新增人工 reviewed Gold 时，继续停止文本采样/重权/SVC pair patch 等同类低收益试错；优先等待 `company_oof_disagreement_class_balanced_20pct` 305 条高价值复核队列产生 Gold。若仍无 Gold，只推进候选生产版本的 bootstrap/challenge/延迟成本与最终冻结质量证据，不针对 external 做模型选择。

# 2026-08-28 冲奖版迭代：DeepSeek 分歧样本复核仍不足，Macro-F1 导向 Gold 队列提升至 oracle 0.88748

## 2026-08-28 冲奖版迭代：公开主体遮罩假设淘汰，external 未触碰

- 运行前经 `windowsdev` 检查：本地无 EventLens Python 实验进程；远端仅 Jupyter/TensorBoard 基础服务，RTX 4090 D=`0% / 1 MiB`，因此没有重复启动已失败的 reranker/listwise/LLM 自动覆盖路线。
- 基于既有 challenge slice 中 anti-subject-prior 明显偏弱的证据，验证一个低自由度、production-like 的“主体捷径抑制”假设：固定当前最强配方 `2400 chars + schema desc x1 + exact_fallback_k5`，只改变文本是否遮罩主体名；主体名仅来自公开事件 schema 或高精度 hard-route 接受结果，不读取 labeled-only `trading_code/entity/industry` 真值。
- 新增 `tools/benchmark_subject_mask_svc.py`，协议为严格 3-fold train OOF；只有遮罩相对 baseline OOF Macro-F1 至少 `+0.005` 才允许读取 external tagged test。首次全公开 schema 主体名遮罩：baseline OOF=`0.764504`，mask=`0.739887`，下降 `-0.024617`，因此 external 未触碰。
- 为排除“全局遮罩误删正文中其他公司”的混杂因素，再验证更保守的 `accepted_subject_name_mask`：仅当 production route 已接受高精度主体时遮罩该主体名。结果 OOF=`0.744324`，相对 baseline 仍下降 `-0.020180`，external 继续未触碰。
- 结论：company 文本中的主体词不仅是 shortcut，也携带事件判别所需上下文；简单删除主体信息会稳定伤害 Top-K→Top1。该方向按 Proof-or-Stop 淘汰，不进入 production，也不继续扩展遮罩强度/位置网格。
- 工程边界：失败实验逻辑仅保留在 benchmark 工具中，未新增 production 模块/配置/阈值；当前 production-like external 最新最优仍为 `0.770798`，尚未超过 0.80。
- 下一步：在没有新增人工 reviewed Gold 前，不再继续“文本删除/重权/SVC pair patch”同类小修。优先把现有 class-balanced disagreement 305 条队列转化为高质量 Gold；若仍无人工 Gold，则只推进可证明的 challenge/治理/冻结质量，不制造 test-driven 增益。

- 运行前经 `windowsdev` 检查：本地/远端均无 EventLens 实验进程，RTX 4090 D=`0% / 1 MiB`，`/root/autodl-tmp` 约剩余 25GB；因此没有重复启动已否决的 reranker/listwise/SVC 路线。远端已有新的 OOF bootstrap 与 acquisition-v2 报告，先读取并复用，避免重复计算。
- OOF bootstrap 再确认 company production-like 基线 Macro-F1=`0.764504`，95% sample-bootstrap CI=`[0.732160, 0.783968]`；SVC+BGE union Top-5 对 307 个 OOF 错误覆盖 `93.1596%`，瓶颈仍是 Top-K→Top1，而不是候选召回。
- 为验证 DeepSeek 是否能在更高错误富集的 SVC/BGE 分歧区间成为“高精度选择性纠错器”，扩展 `benchmark_llm_teacher_gold.py --selection-strategy disagreement_then_margin`。首次运行因 shell `source .env` 把 CRLF `\r` 带入 API header，30 条均在本地 header 校验失败且 API request=0；修正为不 source `.env`、由项目 `load_env_file()` 正常读取后重跑，未修改 key、模型或 prompt。
- 真实 train-OOF 30 条分歧样本 replay：candidate truth hit=`0.966667`，baseline accuracy=`0.266667`；DeepSeek 非 abstain precision=`0.315789`，11 abstain，confidence>=0.9 时 accepted=15、precision=`0.4`、corrected=4、harmed=1，P95 latency≈`32.1s`，30 requests / 111658 tokens。该精度仍远低于自动覆盖要求，因此继续严格保持 Shadow/HITL，不把 DeepSeek 接管 production 预测。
- acquisition-v2 证明 long-text / long-tail-source / candidate-only 风险优先并未超过 `disagreement_then_margin`，因此停止继续堆风险启发式。改为直接优化“相同人工预算可恢复的 Macro-F1”：`benchmark_review_acquisition.py` 新增离线 oracle Macro-F1 指标，以及两个仅依赖 inference-visible SVC/BGE 分歧的多样性策略。
- 20%=305 条 OOF 预算结果：`disagreement_then_margin` 抓 143 个错误、oracle Macro-F1=`0.882200`；`disagreement_class_balanced_margin` 抓 139 个错误，但按 SVC 预测类轮转后 oracle Macro-F1=`0.887480`、gain=`+0.122976`，candidate truth hit=`0.963934`，优于原 disagreement-first 的 Macro-F1 上限。`disagreement_pair_balanced_margin` 为 `0.884823`，次优。
- 工程落地：`build_company_oof_review_queue.py` 新增 `disagreement_class_balanced_margin`，正式生成 `artifacts/review_queue/company_oof_disagreement_class_balanced_20pct.jsonl` 305 条；队列仍不包含 Gold truth，保留 provenance 与 `requires_human_approval=true`。报告：`reports/company_oof_review_queue_disagreement_class_balanced_20pct.json`。
- 本轮没有读取 external tagged test 调参，也没有修改 production 分类器、候选门禁、Proof-or-Stop、Claim→Evidence、DeepSeek Shadow/HITL、runtime fallback/degrade/stop 或数据飞轮审批/rollback。production-like external 最新最优仍为 `0.770798`，尚未超过 0.80；本轮提升的是获取能支撑 0.85 的 Gold 预算效率，不虚报自动模型分数提升。
- 局部验证：修改后的 LLM/acquisition/review-queue 工具 `py_compile` 通过；review/oof/calibration 相关测试 `12 passed`，随后 queue 工程相关 `9 passed`。结束时远端 GPU 空闲且无 EventLens 实验进程。
- 下一步：优先让 class-balanced disagreement 队列获得人工 reviewed Gold，并按 tranche 进入受治理 Feedback/Data Flywheel 后做严格 train-only OOF/challenge-slice 重训验证；在没有新 Gold 前不继续已连续失败的自动 reranker/LLM 覆盖路线。

# 2026-08-28 冲奖版迭代：review acquisition 优化，同预算错误富集率提升至 46.89%

- 运行前经 `windowsdev` 检查：本地无 EventLens Python 实验进程；远端无 EventLens 训练/评测进程；RTX 4090 D=`0% / 1 MiB`，`/root/autodl-tmp` 约剩余 25GB。原 305 条 `company_oof_low_margin_20pct.jsonl` 完整存在，但没有新增人工 reviewed Gold，因此不重复启动已否决的 reranker/listwise/SVC 修补路线。
- 决策：在没有新 Gold 时，最高价值动作不是继续扩模型，而是提高相同人工复核预算的错误富集效率。新增 `tools/benchmark_review_acquisition.py`，只在 company production-like 3-fold train OOF 上比较 5 个通用采样策略；采样只使用推理时可见的 SVC margin、SVC/BGE Top-1 分歧、主体 route 状态与 BGE margin，Gold 仅用于聚合离线评估，不参与单样本选样。
- 基准结果（20%=305 条）：`low_margin` 错误 `120/305=0.393443`、candidate truth hit=`0.954098`；最佳 `disagreement_then_margin` 错误 `143/305=0.468852`、candidate truth hit=`0.960656`。相同人工预算额外富集 **23 个 OOF 错误**，错误率相对提升约 `+19.2%`，且候选覆盖没有牺牲。`candidate_only_then_margin=0.367213`，说明主体拒识本身不是高价值 Gold 充分条件。
- 工程落地：`build_company_oof_review_queue.py` 新增 `--strategy {low_margin,disagreement_then_margin}`，默认保持旧行为以兼容已有复现命令；`ReviewQueueItem.reason` 支持可审计策略原因。正式生成新队列 `artifacts/review_queue/company_oof_disagreement_20pct.jsonl` 305 条，离线错误 143 条，候选 truth hit=`0.960656`。旧 low-margin 队列保留，不覆盖历史 provenance。
- 产物：`reports/review_acquisition_company_oof.json`、`reports/company_oof_review_queue_disagreement_20pct.json`、`artifacts/review_queue/company_oof_disagreement_20pct.jsonl`。README 已切换推荐命令到 disagreement-first 队列；人工反馈仍必须通过 provenance、reviewer、approval、schema、Evaluation/Release Agent 与 rollback 门禁。
- 验证：本地 review/oof 专项 `8 passed`、两个工具 `py_compile` 通过；阶段回归本地完整 `139 passed in 6.86s`、`compileall -q src tools` 与 `git diff --check` 通过。同步远端后专项 `8 passed`，完整 `133 passed in 2.50s`、`compileall` 通过；两端测试数量差异来自既有测试库存差异。本轮没有读取 external tagged test 做调参，也没有改 production 分类器、候选门禁、Proof-or-Stop、Claim→Evidence、DeepSeek Shadow/HITL 或 runtime 安全策略。
- production-like 最新最优仍为 company external Macro-F1=`0.770798`，因此尚未超过 0.80、更未实际达到 0.85。train OOF 的 20% hard-case oracle 上限仍为 `0.854131`；本轮的意义是把到达该上限所需的人工 Gold 成本降低，而不是虚报自动模型增益。
- 结束状态：远端 RTX 4090 D=`0% / 1 MiB`，无 EventLens 实验进程。尝试通过 Gmail `get_profile` 获取当前账户身份以只发给自己时，连接器返回 `FORBIDDEN: This conversation is restricted to developer MCPs`；因此本轮进度邮件无法安全发送，未向任何未知地址发送邮件。
- 下一步：优先等待 disagreement-first 队列的人工作业结果；一旦出现 reviewed Gold，按 tranche 导入并做 train-only OOF/challenge-slice 增益验证。若仍无新 Gold，则继续强化 Gold 获取/审核效率与企业交付稳定性，不回到已连续失败的同类 reranker 小修补。

# 2026-08-28 冲奖版迭代：confusion specialist 淘汰，启动标题/正文分权 SVC

## 2026-08-28 冲奖版迭代：305 条 hard-case 人工 Gold 回流门禁闭环

- 运行前经 `windowsdev` 检查：本地无 Python/EventLens 实验进程；远端仅有 Jupyter/TensorBoard 基础服务，无 EventLens 训练/评测进程；RTX 4090 D=`0% / 1 MiB`，`/root/autodl-tmp` 约剩余 25GB。305 条 `artifacts/review_queue/company_oof_low_margin_20pct.jsonl` 完整存在，因此没有重复启动已否决的 reranker/listwise/SVC 修补实验。
- 当前没有新增人工 Gold。按上一轮 oracle 与 DeepSeek replay 证据，本轮不启动 GPU 微调，转而补齐“人工复核结果 -> FeedbackRecord -> 受治理数据飞轮”的最小生产闭环，避免 review queue 只是静态 JSONL。
- `src/eventlens/review_queue.py` 新增 `HumanReviewDecision`、queue/review JSONL loader 与 `convert_reviews_to_feedback`。硬门禁：`review_id` 必须存在且唯一；`provenance_hash` 必须与原队列一致；只有 `approved=true` 且 reviewer 非空才可进入反馈；人工标签必须属于正式 event schema。人工正确标签允许落在原 Top-K 之外，避免把约 4.6% candidate miss 强制改成错误标签，但会用 `candidate_hit=false` 留审计痕迹。
- 每条通过反馈生成确定性 `feedback_id`，metadata 保留 `review_id`、原 provenance、candidate_hit、route_status、SVC margin 与人工审批要求；`kind` 区分 `wrong_type/confirmed_type`，后续仍必须经过现有 Evaluation/Release Agent 门禁，不能直接改写 production 预测。
- 新增 `tools/import_review_feedback.py`：把人工 reviewed JSONL 安全追加到统一 `data/feedback/feedback.jsonl`，自动跳过已有 feedback_id，防止重复导入污染数据飞轮。README 增加可复现命令与 reviewed JSONL 字段约定。
- 验证：本地 review queue + learning 专项 `9 passed`，`compileall -q src tools`、`git diff --check` 通过；本地完整 `138 passed in 6.92s`。同步远端后专项 `9 passed`、完整 `132 passed in 2.46s`、`compileall` 通过。远端同步目录不是 Git checkout，因此远端 `git diff --check` 不适用；本地真实 Git workspace 已通过同一检查。
- production-like 最新最优仍为 char-SVC external Macro-F1=`0.770798`，本轮未读取 external 做任何调参，也未声称超过 0.80/达到 0.85。当前可量化上限仍是最低 margin 20% hard cases 完美纠错时 OOF Macro-F1=`0.854131`；真正跨过 0.80/0.85 的下一必要条件是获得这批样本的高质量人工 Gold，而不是继续扩模型。
- 企业门禁保持：Proof-or-Stop、Claim→Evidence、DeepSeek Shadow/HITL、runtime retry/restart/fallback/degrade/stop、provenance/审批/rollback 均未放宽；本轮没有改 production 模型、阈值、候选策略或聚类逻辑。
- 下一步：当 reviewed JSONL 出现新增人工 Gold 后，先导入并做重复/合法性审计，再按 Gold tranche 规模进行严格 train-only 重训/OOF 增益与 challenge-slice 评估；无新 Gold 时停止同类模型试错，优先保持交付稳定性。

## 2026-08-28 冲奖版迭代：SVC 跨类校准淘汰，20% hard-case Gold 队列落地

- 运行前经 `windowsdev` 检查：本地/远端均无 EventLens 实验进程；RTX 4090 D 空闲（0% / 1 MiB），`/root/autodl-tmp` 约剩余 25GB。因此未重复启动 reranker/listwise 等已否决实验，按上一断点验证“LinearSVC OVR decision score 跨类别不可比”假设。
- 新增 `ClasswiseScoreCalibrator`，只比较 `identity / zscore / OVR Platt` 三个通用低自由度校准族。为了避免 OOF 自身过拟合，采用二层协议：第一层 3-fold 生成 base OOF score；第二层再 3-fold 用 meta-train 拟合校准器、meta-val 评估，所有选择仅依据 train OOF，不读取 labeled-only 主体真值，不按事件名称硬编码。
- 校准结果：meta-OOF `identity=0.758070`、`zscore=0.737340`、`platt=0.693116` Macro-F1；最佳仍是 identity，因此按照门禁 **没有执行 external tagged test**。结论：当前 Top-K→Top1 问题不是简单的跨类 score scale 失真，停止继续 classwise calibration/temperature 类 SVC 分数修补。
- 进一步扩展 train OOF 错误分析，量化最低 margin hard cases 的“完美专家”上限：最低 10%（153 条）若只纠错不伤害，Macro-F1 上限 `0.807131`；最低 20%（305 条）上限 **`0.854131`**；30% 上限 `0.898980`。这说明 0.85 不要求全量换模型，理论上只需把约 20% hard cases 处理好。
- 结合已有 train-OOF DeepSeek Gold replay（60 条 long-tail/低-margin）做路线判定：候选真值覆盖 `0.916667`，但 DeepSeek 非 abstain precision 仅 `0.361111`；confidence>=0.9 时 corrected=4、harmed=7。因此当前 DeepSeek 不能自动接管这 20%，继续严格限定为 Shadow/HITL/补采建议，Proof-or-Stop 与 unsafe_continue_rate=0 门禁不变。
- 新增受治理 `ReviewQueueItem` 与 `build_company_oof_review_queue.py`，将最低 margin 20% 固化为人工 Gold 入口。正式队列 `artifacts/review_queue/company_oof_low_margin_20pct.jsonl` 共 305 条；队列**不包含 Gold truth**，只记录 article_id、baseline event、margin、candidate events、route status、priority、provenance hash、pending 状态与 `requires_human_approval=true`，避免把离线真值泄漏成“待标注提示”。
- 队列离线容量报告 `reports/company_oof_review_queue_20pct.json`：305 条中当前 baseline 错误 120 条，error rate=`0.393443`，候选集合 truth hit rate=`0.954098`。这与 oracle `0.854131` 一致指向：下一轮最有价值的数据飞轮动作是为这类 hard cases 获取高质量人工 Gold，再按 provenance/审批/rollback 进入训练，而不是继续扩模型动物园。
- 测试：score calibration + schema/event retrieval 本地/远端均 `14 passed`；OOF frontier 新增专项本地/远端 `7 passed`；review queue + OOF/calibration 专项本地/远端 `8 passed`。本轮未改 production config、模型阈值、Proof-or-Stop、Claim→Evidence、runtime fallback/degrade/stop 或 DeepSeek 自动覆盖策略。
- 阶段回归：本地完整 `136 passed in 11.84s`，`compileall -q src tools` 与 `git diff --check` 通过；远端当前测试库存完整 `130 passed in 2.70s`，`compileall -q src tools` 通过。远端结束时 RTX 4090 D=`0% / 1 MiB` 且无 EventLens 实验进程。两端完整测试数量不同来自现有工作区测试文件库存差异，本轮新增 calibration/frontier/review-queue 专项在两端均已单独验证通过。
- 下一步：优先将人工复核后的结构化反馈接入现有 `FeedbackRecord -> Skill/Data Flywheel`，并在新增 Gold 数达到可做严格 train-only 重训/重评的规模后验证 Macro-F1 增益；在没有新 Gold 前不再做同类 SVC/reranker 小修补。

- 断点巡检：远端上一轮 listwise reranker 已结束，orientation 修正后 external company production-like Macro-F1=`0.453792`，显著低于 char-SVC=`0.770798`，确认任务型 listwise BGE reranker 当前不可用；本轮未重复启动该路线。
- 发现已有 `benchmark_confusion_specialist_svc.py` 正在远端执行，遵守“不重复同类任务”原则等待其完成。该实验所有混淆对、最小错误计数和 margin gate 均由 nested train OOF 自动选择，不含 external 调参和事件名称硬编码。
- confusion specialist 完成：train OOF baseline Macro-F1=`0.764504`，最优 `count=5:q=0.4` 为 `0.766485`（仅 `+0.001981`）；external baseline=`0.770798`，specialist=`0.765782`（`-0.005016`）。结论：高频混淆对存在，但 pairwise 决策边界跨 split 漂移，路线淘汰，不进入生产。
- OOF 错误证据：307 个错误中，SVC Top-5 覆盖 `89.25%`、SVC+BGE union Top-5 覆盖 `93.16%`；Top-15 混淆对覆盖 `54.40%` 错误；最低 margin 50% 样本覆盖 `75.24%` 错误。候选召回已不是主瓶颈，继续聚焦 Top-K→Top1 与文本判别表示。
- 基于该错误分析新增 `tools/benchmark_title_weighted_svc.py`：只使用可生产获得的标题与正文，通过通用 `title_repeat={1,2,3}` × `content_limit={1200,1800,2400}` 小网格做 3-fold train OOF 选择，再一次性 external 验证；不读取 labeled-only 主体字段、不按 external 调参、不硬编码事件对。
- 本地验证：新脚本 `py_compile` 通过；confusion specialist 专项 `3 passed`。脚本同步远端后完成 `reports/title_weighted_svc_company.json`：train OOF 最优 `title_repeat=3,content=2400` Macro-F1=`0.764550`，external=`0.759867`，较稳定 production-like 基线 `0.770798` 下降 `-0.010931`。因此“标题重复加权/裁剪正文”路线也淘汰；本轮结束时远端无 EventLens 实验进程，4090D 空闲。
- 企业门禁保持：本轮只做离线候选模型实验，未修改 production config、Proof-or-Stop、Claim→Evidence、DeepSeek shadow/HITL、runtime fallback/degrade/stop 或数据飞轮治理策略。
- Gmail：尝试获取当前 Gmail profile 时连接器返回 `FORBIDDEN: This conversation is restricted to developer MCPs`，无法安全确认“当前用户自己的邮箱”，因此本轮无法发送要求的进度邮件；未向任何未知地址发送邮件。
- 下一步：连续的 confusion specialist 与 title/body reweight 两个低自由度 SVC 修补均未跨 split 泛化，停止继续同类文本重权/局部 pair patch。下一优先级是验证“OVR decision score 跨类不可比”假设：只在 train OOF 上做 class-wise score calibration/校准门禁，OOF 不增益则不触碰 external；若仍失败，再转向受治理 Gold 增量而不是扩大模型动物园。

# 2026-08-27 冲奖版迭代：listwise 失败归因到 reranker 输入方向并重跑

- 运行前经 `windowsdev` 检查：远端无残留 listwise/groupwise/reranker 进程，RTX 4090 D 空闲（0% / 1 MiB），因此没有重复启动同类任务；先读取上一轮 `artifacts/logs/listwise_reranker_company.log` 与 `reports/listwise_reranker_company.json`。
- 上一轮 listwise 明确失败：train-only 最佳 validation Macro-F1=`0.452156`，external Macro-F1=`0.477167`，相对 production-like char-SVC `0.770798` 回退 `-0.293631`；candidate hit rate=`0.989529`，再次排除 Top-K coverage 为主瓶颈。
- 错误归因发现实现偏差：已验证效果更好的 groupwise 路线与 BGE reranker 预训练习惯均采用“短事件语义作为 query、长文章作为 document”，而 listwise 旧实现把文章放左、事件定义放右。对非对称 cross-encoder reranker 该顺序不等价，因此旧 listwise 结果不能作为路线最终否决证据。
- 修复：新增 `build_label_to_article_pairs`，训练和预测统一构造 `事件类型+事件定义 -> 文章` 的 label-to-article 输入；未修改 Gold、OOF candidate mining、class weight、last 4 layers、`lr=1e-5`、3 epoch 或 external 使用协议。
- 测试：本地 `tests/test_listwise_reranker.py` => `4 passed`；同步最小 3 文件到远端后同专项 => `4 passed`。
- 修正后的正式 listwise 实验已在远端启动，使用原 company labeled train/external embedding、同一个 `bge-reranker-v2-m3`、last4/1e-5/3epoch；运行中 Python 进程显存约 `3362 MiB`。输出目标：`reports/listwise_reranker_company_orientation_fix.json`。
- 企业门禁保持不变：production-like 输入禁止 labeled-only 主体真值；模型/epoch 只按 train 内部验证选择；external 只最终验证；Proof-or-Stop、Claim→Evidence、DeepSeek hard-case shadow/HITL、runtime retry/restart/fallback/degrade/stop、飞轮 provenance/审批/rollback 均未修改。
- 下一步：优先读取修正实验的 train-only validation history。若内部验证仍显著低于 SVC/既有 groupwise，则停止 listwise 路线并转向 OOF confusion/challenge-slice 错误归因；若内部明显恢复，再允许 external 一次性验证，不根据 external 回调参数。

# 2026-08-27 冲奖版迭代：groupwise 失败后进入同模型 listwise 排序验证

- 运行前经 `windowsdev` 检查：本地无 Python/EventLens 实验进程；远端 `eventlens-gpu` 正常，RTX 4090 D 空闲（1 MiB / 0%），`/root/autodl-tmp` 约剩余 25GB，未发现并行 reranker/训练任务，因此允许启动下一项实验。
- 当前 production-like 稳定基线仍以 char-SVC external Macro-F1=`0.770798` 为准；`reports/groupwise_reranker_company.json` 的 groupwise 微调在 train-only 80/20 验证上最佳融合仅 `0.783331`，external=`0.760538`，较 SVC 回退 `-0.010260`。候选覆盖 `0.989529`，说明瓶颈不是 Top-K coverage，而是 Top-K→Top1 排序泛化。
- 决策：不扩大模型动物园，也不继续针对 external 调参。沿用同一个 `bge-reranker-v2-m3` 初始化、production-like 无主体真值输入、train OOF SVC+BGE 候选；沿用 groupwise 在 train-only 选择出的最后 4 层可训练与 `lr=1e-5`，只将目标切换为更贴近整组 Top-K→Top1 的 listwise cross-entropy，最多 3 epoch，epoch 仍只由 train 内部验证选择。
- 远端首次启动发现 `tools/benchmark_listwise_reranker.py` 尚未同步，命令未进入训练；该问题属于远端代码版本缺失，不是算法/GPU 失败。仅同步 `benchmark_listwise_reranker.py`、`listwise_reranker.py` 与对应测试，远端专项 `3 passed`；本地 groupwise/listwise 相关测试 `7 passed`。
- 正式 listwise 实验已在远端启动，训练 Python PID=`48778`，观察时显存约 `3362 MiB`；日志：`artifacts/logs/listwise_reranker_company.log`，目标报告：`reports/listwise_reranker_company.json`。本轮结束时任务仍在运行，下一轮必须优先读取该进程/日志/报告，禁止重复启动。
- 企业门禁未变：Proof-or-Stop、Claim→Evidence、DeepSeek hard-case shadow/HITL、runtime retry/restart/fallback/degrade/stop、数据飞轮 provenance/审批/rollback 均未修改；本轮没有改生产模型或阈值。
- 下一步判据：若 listwise train-only 验证不能形成明显且稳定的排序增益，停止该 reranker 路线并基于 OOF confusion/challenge slices 做错误归因；只有内部协议先成立，才允许 external 做一次最终验证，不允许根据 external 结果回调参数。

# 2026-08-27 v0.7.0：DeepSeek V4-Pro 真实 Agent 与赛题交付封装

- 根目录 `.env` 已集中管理外部 LLM 参数与密钥；`.env` 被 Git 忽略，`.env.example` 只保留模板。真实 API smoke 使用 `deepseek-v4-pro + thinking=enabled + reasoning_effort=max + JSON output` 成功返回，未打印密钥。
- Qwen2.5-3B 与 DeepSeek V4-Pro 使用完全相同的 company labeled-test 30 条 hard-case。Qwen3B+Verifier Accuracy/Macro-F1=`0.333333/0.252564`、corrected=1、harmed=0；DeepSeek+Verifier=`0.533333/0.361806`、corrected=9、harmed=1。裸 DeepSeek Expert=`0.566667/0.421825`、corrected=11、harmed=2、abstain=5、collection request=6。
- DeepSeek 30 条共 110 API 请求，209,583 tokens；其中输入 cache hit/miss=`81664/64691`、输出=`63228`、reasoning=`59288`。按当前官方常规价估算约 ¥0.58；主要瓶颈是约 39.5s/hard-case 的 `thinking=max` 串行延迟，而不是 API 成本。
- 检查 Verifier 接受改判发现唯一 harmed 的 confidence=0.90，多条 corrected 同样=0.90，因此不存在可靠简单 confidence gate。停止 prompt/阈值网格，正式角色冻结为 hard-case Shadow Expert + Verifier + HITL/补采建议，不自动覆盖 baseline。
- `predict-assets` 新增 `--agent-shadow --agent-max-samples`：只对主体已 hard-route 的难例运行 Agent，正式预测保持不变；输出 `agent_shadow.jsonl` 与 `agent_shadow_summary.json`，记录改判建议、abstain、补采、Verifier、P95 latency 与 API usage。
- 真实 `untagged_test` company 20 篇 shadow smoke：主链路 `validate-run PASS`，Agent 选择 3 条、valid=1.0、1 abstain、P95≈17.6s，正式 20 条预测未被改写。industry 20 篇全部 `candidate_only`，因此 Event Agent 自动选择 0 条，主链路仍 PASS，证明主体未决时不会越权。
- 新增 `scripts/run_competition_demo.py` 和 `reports/competition_delivery_readiness.md`，将双 scope 主链路、validate-run、控制安全、信任控制与 Runtime 调度封装为一键演示。版本升级到 v0.7.0。
- 下一步只做交付硬验收：本地/远端完整 pytest + compileall + diff check，并在远端真实执行一键 demo；通过后更新 `goal_mode_complete.json`，不再继续模型实验。
- 最终交付硬验收完成：本地 `94 passed` + compileall + `git diff --check`；远端 `94 passed` + compileall。一键 demo 在远端真实执行 company/industry 各 20 篇，company DeepSeek shadow 最多 2 条，双 `validate-run PASS`；控制安全 11/11、信任控制 9/9、Runtime plan 返回 `scale_up`，生成 `artifacts/competition_demo/demo_manifest.json`。v0.7.0 达到当前赛题交付门槛。

# 2026-08-27 v0.6.0：可信控制、挑战切片与轻量 Runtime Controller

- 吸收参考方案中 5 个高价值点，但不复制 6 Agent/Neo4j/RAG/8 MCP 等重型外壳：新增 Proof-or-Stop、Claim→Evidence、控制安全 benchmark、Anti-shortcut slices、Skill provenance/shadow/rollback。
- 新增轻量 `RuntimeController`：queue high/low 扩缩、失败 retry/backoff、process crash restart、连续失败 degrade/stop、依赖失败 fallback、Evidence Gate 失败生成 `CollectionRequest`。控制逻辑与基础设施执行分离，不做 LLM 调度。
- 本地新增能力专项 `20 passed`，随后完整 `84 passed`；同步 4090D 后远端完整同样 `84 passed`。
- 控制安全故障注入：11/11，`action_match_rate=1.0`，`unsafe_continue_rate=0.0`；信任控制故障注入：Evidence Gate 4/4，Skill governance 5/5。
- 真实 5k company/industry 的 Claim→Evidence coverage 均为 `1.0`，Evidence Gate pass rate 均为 `1.0`，blocked alert 均为 0；该结果只说明真实样本都有证据，不用来替代故障注入的阻断验证。
- 外部 challenge slices 暴露新短板：company anti-subject-prior Macro-F1=`0.283007`、long-text=`0.560167`；industry anti-subject-prior=`0.375940`、rare-event=`0.333333`。因此后续 Gold 优先覆盖这些 slice，不启动无监督依据的 QLoRA。
- 详细报告：`reports/v060_trust_runtime_experiments.md`；远端 JSON：`reports/control_safety_benchmark_v060.json`、`reports/trust_control_benchmark_v060.json`、`reports/event_external_{company,industry}_v060.json`、`reports/smoke_*_validation_v060.json`。
- 最终验收：本地 `85 passed` + compileall + `git diff --check`；远端 4090D `85 passed` + compileall，v0.6.0 主链路完成。

# 2026-08-27 Goal Mode：竞赛主链路闭环与 v0.5.1 冻结

- 决策：停止继续增加模型实验，仅完成“经过外部评测的能力正式接入最终输出”。新增资产驱动 `predict-assets`：TF-IDF 负责事件分类，已经完成的 BGE 向量只复用于主体路由、事件 Top-K 与候选聚类，不再次编码正文。
- 外部事件识别：company TF-IDF Acc/Macro-F1=`0.795812/0.758846`，industry=`0.865248/0.836283`；BGE-linear 分别仅 `0.693717/0.671306` 与 `0.819149/0.780152`，因此否决 BGE-linear 替代方案。BGE routed Top-3 命中 company=`0.901832`、industry=`0.918440`，只做召回/审计/HITL。
- 外部重复新闻：company 候选边 eligible recall=`1.000000`，语义聚合 Pairwise P/R/F1=`1.000000/0.867713/0.929172`、overmerged article=`0`；industry eligible recall=`0.980159`，语义聚合=`0.994723/0.748016/0.853907`、overmerged article=`0.012`。个股达到既定 Recall 增益/Precision 门禁；行业竞赛 Pairwise F1 有增益，但业务稳健模式继续保留 shadow 风险标记。
- 新增 `cluster_candidate_events` 将评测过的“Event Top-1 一致 + BGE 阈值”候选聚类产出真实 `EventCluster`；新增主体解析方法/置信度字段，最终 `article_event.jsonl` 可回溯到 exact/BGE/candidate-only 路由。
- 新增 `validate-run`：硬校验 article→cluster→alert→lifecycle 的 ID 闭合和数量一致性，避免演示链路“文件都生成了但互相对不上”。
- 首次 1k smoke 虽结构通过，但 company/industry 的 Learning Signal 分别为 `972/985`，发现单来源低可信被全部送进学习池。修复 Observer：仅“至少两条证据仍低可信”触发低置信学习信号；证据冲突、官方澄清、高影响仍保持原触发逻辑。
- 修复后 1k smoke：company `1000→972 clusters / 18755 edges / 25 signals`，industry `1000→985 / 19573 / 11`，两条 `validate-run` 均通过。
- 最终 5k smoke：company `98680 edges / 4799 clusters / max cluster 8 / 148 signals / ~79s`；industry `99579 / 4870 / max 8 / 85 / ~78s`；两条路径跨文件引用均 `passed=true`。没有继续扩大 smoke，因为已经证明 Top-20 边规模线性受控且输出稳定。
- 本地完整 pytest 首次因 Windows `C:` 临时盘 `No space left on device` 出现 1 个 I/O 测试失败；确认 `C:` 可用空间为 0 后，把 pytest 临时目录显式迁到 `I:`，同一代码随后完整 `72 passed`。该失败不是项目代码回归，也未删除用户系统文件。
- 冻结：不启动 Qwen 3B/7B、BGE 微调、LightGBM 重排、FAISS 或新的聚类参数网格。已知边界为缺少“非事件”和“低质/谣言”独立 Gold，因此不制造未经验证的 no-event/低质分类阈值。
- 留痕：`src/eventlens/asset_pipeline.py`、`src/eventlens/run_validation.py`、`src/eventlens/candidate_clustering.py`、`src/eventlens/event_external_evaluation.py`、`reports/competition_freeze_decision.md`，远端 `reports/smoke_*_validation_v052.json`。

# 2026-08-10 Goal Mode：全量向量完成，启动主体路由

- 检查远端 `artifacts/embeddings/untagged_train/manifest.json`：`199999/199999`；`untagged_test/manifest.json`：`107889/107889`，两份正式“标题 + 正文前 1600 字”向量均已完成，可直接复用，避免重复编码。
- 远端检查时 GPU 空闲，未发现仍在运行的 embedding 任务，因此进入下一阶段而不是继续等待。
- 按既有外部门禁启动全量主体路由，顺序为 train company → train industry → test company → test industry；公司使用配置中心中的高精度 hard-route + Top-3/reject，行业仅 Top-3/reject。输出目录：`artifacts/subject_routes/`，运行日志：`artifacts/subject_routes/full_route.log`。
- 决策理由：主体路由是后续“主体约束事件召回”和“主体 + 7 天候选边”的必要前置；文章向量已经完成，当前步骤只需编码少量主体 schema，计算成本低且不会与 embedding 重复。
- 下一步：检查四份路由 summary 与日志；确认数量完整后，直接复用同一批文章向量执行主体约束事件 Top-K 召回，并开始设计主体 + 7 天窗口候选边的可审计产物。

## 2026-08-27 Goal Mode：同口径规则 baseline 完成，保守 BGE 聚类维持影子模式

- `windowsdev` 已恢复；本地完整测试 `68 passed in 5.61s`。`ssh eventlens-gpu` 正常，RTX 4090 D 空闲，`/root/autodl-tmp` 约剩余 37GB。
- 远端 `evaluate-candidate-clusters` 仍是旧 CLI，缺少 `--baseline-output`。本轮仅同步本地已实现的 `candidate_clustering.py`、`cli.py` 与对应测试到远端，不修改任何生产阈值或候选策略；远端专项测试 `3 passed`，完整测试 `68 passed in 2.19s`。
- 在完全相同的 duplication_id 文章集合、完全相同的主体候选边和事件 Top-1 结果上补齐规则 baseline，对比口径终于一致。
- 公司 999 篇：规则 baseline Pairwise P/R/F1=`0.963740/0.901786/0.931734`，B-cubed F1=`0.980818`，overmerged article rate=`0.012012`；保守 BGE+Event Top-1 为 `0.993827/0.862500/0.923518`，B-cubed F1=`0.977085`，overmerged article rate=`0.004004`。
- 公司发布门禁判定：Precision 提高且过合并下降，但 Recall 下降 `-0.039286`、B-cubed F1 下降 `-0.003733`，既没有 B-cubed F1 `+0.02`，也没有 Recall `+0.05`；因此严格维持影子模式，不替换默认规则聚类。
- 行业 500 篇旁证：规则 baseline Pairwise P/R/F1=`0.991826/0.928571/0.959157`，B-cubed F1=`0.982663`，overmerged article rate=`0.008000`；保守 BGE+Event Top-1 为 `0.988950/0.913265/0.949602`，B-cubed F1=`0.977980`，overmerged article rate=`0.012000`。行业各项均未优于 baseline，继续仅作旁证且不启用 hard-route。
- 决策：当前最有价值用途不是把 BGE 聚类切成默认，而是把“规则合并、BGE 拒绝”以及少量 transitive overmerge 冲突作为高价值人工复核/Gold 队列来源。该能力可直接服务数据飞轮，但不能直接作为伪标签训练集。
- Qwen 3B QLoRA 仍不启动：当前证据表明瓶颈是聚类决策边界与 Gold 监督不足，而不是 embedding 算力或模型容量；在没有新增高质量人工标注前微调缺少收益依据。
- 产物：远端 `reports/candidate_cluster_rule_baseline_company.json`、`reports/candidate_cluster_rule_baseline_industry.json`，以及既有 `reports/candidate_cluster_{company,industry}.json` 和 `artifacts/cluster_decisions/*.jsonl`。
- 下一步：从同口径差异中生成可审计的“聚类冲突复核队列”（优先规则合并但 BGE 拒绝、BGE 合并但规则拒绝、transitive overmerge），接入既有受治理数据飞轮；不改变默认聚类策略。

## 2026-08-11 Goal Mode：保守候选聚类实现与 duplication_id 外部评测

- 远端巡检：当前无 EventLens 运行任务，RTX 4090 D 空闲；`/root/autodl-tmp` 约剩余 37GB。四份主体路由、四份无标签事件 Top-3 召回、company train 候选边、四份难例池均存在；候选召回门禁继续为 company=`0.978571`、industry=`0.982143`，均 >=`0.95`。
- 新增 `candidate_clustering.py`：只在“候选边两端事件 Top-1（主体代码 + 事件名）一致”且文章 BGE 向量余弦相似度 >= `cluster.semantic.similarity_threshold=0.92093` 时 Union-Find 合并；每条边记录 semantic score、event_consistent、merged、reason。没有新增模型、没有改变主体/事件候选、没有降低任何阈值。
- 新增 `evaluate-candidate-clusters` CLI，要求 embedding index、事件召回和评测文章严格同序；duplication_id 必须全覆盖，否则立即失败。支持输出 JSON 指标报告和 JSONL 逐边审计决策。
- 测试：本地候选聚类/候选边/事件召回专项 `17 passed`，本地完整 `67 passed`；同步远端后专项 `17 passed`，远端完整 `67 passed`。
- 为保证同口径评测，先对既有 duplication_id embedding + 主体路由生成事件 Top-3：company=`999/999`、industry=`500/500` 均有事件候选；未重新编码文章向量。
- 个股重复新闻保守聚类：999 篇、12578 候选边，事件 Top-1 一致边 1954，BGE>=0.92093 边 496，实际合并边 482；Pairwise P/R/F1=`0.993827/0.862500/0.923518`，B-cubed F1=`0.977085`，overmerged article rate=`0.004004`。报告：远端 `reports/candidate_cluster_company.json`，审计：`artifacts/cluster_decisions/company_duplicate_train.jsonl`。
- 行业重复新闻旁证：500 篇、8687 候选边，事件一致 1751，BGE 过阈 379，合并 358；Pairwise P/R/F1=`0.988950/0.913265/0.949602`，B-cubed F1=`0.977980`，overmerged article rate=`0.012000`。报告：远端 `reports/candidate_cluster_industry.json`。
- 发布决策：当前规则证明了高精度保守聚类可行，但这次 999/500 篇评测还没有生成“同一文章集合、同一主体候选约束”的规则 baseline，因此不能依据旧 442/88 篇 benchmark 直接做增益比较。按照既定门禁，当前能力保持影子模式，不接管默认聚类。
- 下一步：在同一 duplication_id 全量评测集补规则 baseline 对照，并检查 transitive merge 造成的少量 overmerge；只有公司 Pairwise Precision 不低于 baseline、过合并不增加且 B3/Recall 增益达标才允许进入无标签全量默认聚类。若增益不足，仍可保留高精度影子决策作为聚类冲突 Gold 标注队列来源；现阶段继续不启动 Qwen 3B QLoRA。
- 留痕：`src/eventlens/candidate_clustering.py`、`tests/test_candidate_clustering.py`、`src/eventlens/cli.py`、`README.md`、本文件；远端上述报告与 cluster decision JSONL。

## 2026-08-11 Goal Mode：生成四份难例池并修正行业策略性拒识污染

- 远端巡检确认当前无运行中的 EventLens 任务，RTX 4090 D 空闲；`/root/autodl-tmp` 约剩余 37GB。四份主体路由、四份事件 Top-3 召回与 company train 候选边均存在；真实 duplication_id 候选层门禁继续保持 company=`0.978571`、industry=`0.982143`，均高于 `0.95`。
- 基于现有主体路由与事件召回生成四份难例池：company/industry × train/test 各 `5000` 条，共 `20000` 条，产物位于 `artifacts/hard_examples/`。没有启动 Qwen，也没有让难例直接回流生产。
- 首次统计发现四份 Top-5000 均被 `subject_rejected + subject_low_margin + event_low_margin` 占满。审查后确认行业 `subject_rejected` 是既定生产策略（industry hard-route 明确关闭），不是模型错误信号；继续给它固定 `+2` 优先级会系统性污染行业难例池。
- 修复：`build_hard_examples` 新增 `include_subject_rejection_signal`；CLI 根据当前 scope 的 `exact_alias_hard_route/bge_hard_route` 自动决定是否把拒识计为难例。公司保持原语义；行业关闭该信号，只保留真实的主体低 margin、事件低 margin/无候选信号。未新增配置项，避免重复配置。
- 新增单测验证“策略性拒识可忽略”；本地专项 `3 passed`，本地完整测试 `65 passed`；同步到远端后专项 `3 passed`、远端完整测试 `65 passed`。
- 重新生成行业难例池后，industry train/test Top-5000 均变为 `subject_low_margin + event_low_margin`，priority 从约 `2.10` 回落到真实不确定性量级约 `0.095~0.100`；company 仍保持拒识 + 双低 margin 的高优先级样本。
- 决策：当前仍不启动 Qwen 3B QLoRA。原因不是算力不足，而是难例池目前只有不确定性信号、没有新 Gold 标签；在人工标注或高可信监督不足时训练会退化为无门禁伪标签学习，违反既定路线。下一步应先把保守候选边聚类接通并从聚类冲突/错分中筛出更具信息量的 Gold 标注队列，再判断 3B 微调收益。
- 留痕：`src/eventlens/hard_examples.py`、`src/eventlens/cli.py`、`tests/test_hard_examples.py`、`README.md`、本文件；远端 `artifacts/hard_examples/*.jsonl`。

## 2026-08-11 Goal Mode：四份事件候选完成，候选边召回门禁通过

- 远端四份主体路由已完整：company train/test=`199999/107889`，industry train/test=`199999/107889`；公司 hard-route 接受率分别 `11.2921%/11.6574%`，行业继续严格 `0%` hard-route，只保留 Top-3/reject。
- 完成四份主体约束事件 Top-3 召回：company/industry × train/test 的输出条数均与对应 embedding manifest 一致；产物位于 `artifacts/event_recall/`。其中旧标量余弦实现生成 `company_train` 单份耗时约 10 分钟，确认 CPU 是主要瓶颈。
- 性能修复：`recall_from_vectors` 将“单文章×少量事件定义”的余弦相似度从 Python 标量循环改为 NumPy 小矩阵批量计算，不改变候选集合、Top-K、阈值或排序规则；新增标量/向量化 6 位审计精度一致性测试。
- 验证：本地专项 `8 passed`，本地完整测试 `64 passed`；远端专项 `8 passed`，远端完整测试 `64 passed`。合成 12 候选×1024 维基准中，2000 次计算由 `43.534s` 降到 `0.808s`，约 `53.9x`；正式 `industry_train` 199999 篇事件召回约 52 秒完成。
- 生成无标签 company train “共享主体 + 7 天 + Top-20”稀疏候选边：`3998592` 条，产物 `artifacts/candidate_edges/company_train.jsonl`，未做全量两两比较。
- 为避免在无标签全量资产上先聚类后补门禁，新增真实门禁执行顺序：对有标签“个股重复新闻/行业重复新闻”分别单独导出 embedding，使用同一生产主体路由策略生成候选边，再按 duplication_id 评测 7 天 eligible recall。
- 个股重复新闻：999 篇；生产主体路由 hard-route=`839/999`，候选边 `12578`；eligible 正对 `560`，召回 `548`，`eligible recall=0.978571 >= 0.95`，门禁通过。报告：`reports/candidate_edge_recall_company.json`。
- 行业重复新闻：500 篇；继续 `0` hard-route，全部 Top-3/reject；候选边 `8687`；eligible 正对 `392`，召回 `385`，`eligible recall=0.982143 >= 0.95`，门禁通过。行业结果仍按既定规则只作为旁证，不据此放宽行业 hard-route。
- 首次门禁命令因误用不存在的 sheet 名“个股重复”失败；只读取工作表名称元数据后修正为真实 `个股重复新闻/行业重复新闻`，未输出脱敏原文，也未降低门槛。
- 本轮结论：Top-20 稀疏候选层已获得有标签 duplication_id 证据支持，可进入下一阶段保守同源/事件聚类；公司 BGE 聚类仍受原有 Pairwise Precision、过合并和 B-cubed/Recall 增益门禁约束，不能因候选层通过而自动转默认。
- 下一步：生成剩余无标签 industry/test 候选边或按实际聚类入口最小需要补齐资产；随后在候选边上复用文章 embedding + 事件 Top-1 一致性做保守聚类，并形成难例池统计，再决定是否有必要启动 Qwen 3B QLoRA。
- 留痕：`src/eventlens/event_retrieval.py`、`tests/test_event_retrieval.py`、本文件；远端 `artifacts/event_recall/`、`artifacts/candidate_edges/`、`reports/candidate_edge_recall_{company,industry}.json`。

## 2026-08-11 Goal Mode：当前会话立即触发，SSH 恢复但远端写执行被门禁拦截

- 本轮由当前会话手动立即触发，并明确使用 `windowsdev` 访问 `I:\devspaceGPT\EventLens`；没有使用 ChatGPT Linux 容器替代真实环境检查。
- 本地完整测试实际执行：`py -m pytest -q` => `63 passed in 6.52s`，当前代码基线无回归。
- `ssh eventlens-gpu hostname` 已恢复成功，远端主机返回 `autodl-container-4e0e468bcd-24b72162`；数据盘 `/root/autodl-tmp` 约 `50G`，已用约 `11G`，剩余约 `40G`，检查时无 GPU 计算进程。
- 远端正式 embedding 资产再次核验：`untagged_train=199999/199999`、`untagged_test=107889/107889`，均为 BGE-M3、1024 维、float32、正文上限 1600 字；因此继续严格禁止重复编码。
- `artifacts/subject_routes/` 当前只有历史失败日志 `full_route.log`，四份 company/industry × train/test 路由 JSONL 尚未生成。历史失败仍是 BGE-M3 加载时错误访问 `huggingface.co`；远端代码已经包含 `native_embedding.local_files_only=true`，模型缓存位于 `/root/autodl-tmp/hf_cache`。
- 已验证离线环境下 CLI 可正常启动：`HF_HOME=/root/autodl-tmp/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src ... eventlens.cli route-subjects --help` 成功，说明 SSH、Python 环境和 CLI 本身已恢复。
- 随后尝试正式执行无标签 train/company 路由并写入 `artifacts/subject_routes/company_train.jsonl` 时，命令被当前平台安全门禁拦截，未进入远端进程，因此不能宣称路由已开始或失败，也不能绕过门禁伪造产物。
- 下一步断点固定为：在允许远端写执行后，使用离线缓存环境先完成 `company_train`，再依次 `industry_train -> company_test -> industry_test`；每份完成后校验条数/summary，再进入主体约束事件 Top-K、主体+7天候选边与 duplication_id eligible recall `>=0.95` 门禁。
- 决策理由：当前阻塞来自执行权限而非项目代码/GPU/SSH；所有重计算条件均不成立，保持既有 hard-route、Top-K/reject 与候选召回门禁不变。
- 留痕：本文件；远端 `artifacts/embeddings/untagged_{train,test}/manifest.json`、`artifacts/subject_routes/full_route.log`。

# 2026-08-04 生命周期与受治理数据飞轮

- 新增事件生命周期状态机、追加式证据账本与可信度快照。
- 新增 Observer、Diagnosis、SkillCurator、Evaluation、Release Agent 编排。
- 只有通过 Macro-F1、关键错误率和人工审批门禁的 ACTIVE Skill 才能回流推理。
- 审计字段：`applied_skill_ids`、`decision_trace`、`source_feedback_ids`、`approved_by`、`metrics`。
- 留痕位置：`src/eventlens/lifecycle.py`、`src/eventlens/learning.py`、`configs/app.yaml`、`reports/architecture_traceability.md`。

# 实验日志

## 2026-08-28 冲奖版迭代：人工 Gold acquisition frontier，15% 预算已覆盖 0.85 oracle

- 运行前经 `windowsdev` 检查：本地/远端均无 EventLens 实验进程；远端 RTX 4090 D=`0% / 1 MiB`，`/root/autodl-tmp` 约剩余 25GB；未发现新的 reviewed Gold，因此不重复已失败的 reranker/listwise/NB-SVM/文本重权路线。
- 决策：在现有 `disagreement_class_balanced_margin` 受治理队列上补 5%/10%/15%/20% 人工预算 frontier。选样仍只依赖 train 3-fold OOF 下推理可见的 SVC margin、SVC/BGE 分歧和路由信号；Gold 只用于聚合离线 capacity 评估，不进入待复核队列，external tagged test 未触碰。
- 结果：5%=76 条，错误 36 条，oracle Macro-F1=`0.803728`；10%=152 条，错误 72 条，`0.833216`；15%=229 条，错误 105 条，`0.861547`；20%=305 条，错误 139 条，`0.887480`。15% 已在相同 train-OOF 协议下越过 0.85 oracle，因此相较 305 条全量复核，可先减少约 24.9% 人工预算再验证真实 Gold 收益。
- 工程：`tools/benchmark_review_acquisition.py` 新增 `--fractions`，一次 OOF 复用即可输出 acquisition frontier，避免为不同预算重复训练/编码；新增参数解析测试。正式生成 229 条队列 `artifacts/review_queue/company_oof_disagreement_class_balanced_15pct.jsonl`，仍不包含 Gold truth，保留 provenance 与 `requires_human_approval=true`。
- 产物：`reports/review_acquisition_company_oof_frontier.json`、`reports/company_oof_review_queue_disagreement_class_balanced_15pct.json`、上述 229 条 review queue。
- 决策边界：这不是新的 production external 成绩；当前 company production-like external 最优仍为 `0.770798`，尚未超过 0.80。frontier 只用于降低获取高价值人工 Gold 的成本；真实 reviewed Gold 到位后仍需按严格 train-only OOF/challenge-slice 门禁重训验证，不能把 oracle 当模型成绩。
- 下一步：优先完成 229 条 15% tranche 的人工复核并通过 provenance/approval 导入 Feedback/Data Flywheel；若真实 Gold 重训仍不足，再扩到 305 条 20% tranche，而不是继续扩大模型动物园。

## 2026-08-11 Goal Mode：SSH 恢复，完成无标签 train 主体路由

- 本地完整测试恢复并通过：`py -m pytest -q` => `63 passed in 9.10s`，确认当前主链路代码无回归。
- `ssh eventlens-gpu hostname` 已恢复成功；远端 GPU 为 RTX 4090 D，数据盘 `/root/autodl-tmp` 约剩余 40GB。非交互 SSH 的 PATH 不含 Conda/Python，后续统一使用 `/root/miniconda3/envs/eventlens/bin/python`，避免把 PATH 问题误判为环境损坏。
- 远端旧 `full_route.log` 仍停留在 Hugging Face 联网失败，四份主体路由此前没有真实产出。根因是本地 `native_embedding.local_files_only=true` 修复未同步到远端。
- 同步当前 `src/eventlens/*.py`、`tests/*.py` 与 `configs/app.yaml` 到远端后，远端完整测试 `63 passed in 2.40s`；离线模型专项测试此前也为 `12 passed`。运行环境固定 `HF_HOME=/root/autodl-tmp/hf_cache`、`TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1`，不重新下载 BGE-M3。
- 完成无标签 train 公司主体路由：`199999` 篇，hard-route 接受 `22584`（`11.2921%`）；其中唯一精确别名 `8144`，BGE 高置信 `14440`，其余 `177415` 保留 Top-3/reject。输出：`artifacts/subject_routes/company_train.jsonl`。
- 完成无标签 train 行业主体路由：`199999` 篇，hard-route 接受 `0`，全部 `199999` 保留 Top-3 候选。该结果符合既定策略：行业外部门禁未达到 Precision>=0.95，因此禁止强制归属。输出：`artifacts/subject_routes/industry_train.jsonl`。
- 决策理由：本轮只恢复并推进主体路由，不重复 `199999/107889` embedding，不修改公司 score/margin 阈值，也不降低行业拒识与候选边 eligible recall `>=0.95` 门禁。
- 下一步：继续生成 `company_test` 与 `industry_test` 两份主体路由；核验四份 article_count/route summary 完整后，直接复用文章向量执行主体约束事件 Top-K，再进入主体+7天候选边与 duplication_id eligible recall 门禁。
- 留痕：`reports/experiment_log.md`；远端 `artifacts/subject_routes/company_train.jsonl`、`artifacts/subject_routes/industry_train.jsonl`。

## 2026-08-11 Goal Mode 巡检 13

- 本轮先尝试完整测试 `py -m pytest -q` 与最小远端检查 `ssh eventlens-gpu hostname`；两条命令均被当前执行安全门禁拦截，没有获得进程返回值。因此本轮不能把它解释为测试失败，也不能判断 4090D 当前是否仍为 `Connection refused`。
- 静态检查发现 duplication_id 候选召回门禁虽然已有 `evaluate_duplicate_candidate_recall` 实现和 `candidate_edge_evaluation.minimum_eligible_recall=0.95` 配置，但缺少 CLI 入口，真实候选边生成后无法按可复现命令产出门禁报告或阻断后续聚类。
- 修复：新增 `evaluate-candidate-edges` CLI；读取有标签文章与候选边 JSONL，输出 `CandidateEdgeRecallReport` JSON。eligible recall 未达配置门槛时以退出码 2 失败，允许远端流水线在聚类前硬停止；不修改 0.95 阈值、不修改候选算法和最终聚类门禁。
- 新增 `load_candidate_edges_jsonl` 与 JSONL round-trip 单测，README 补充标准命令和“失败后优先调 Top-K/排序而不是降门槛”的处理原则。由于执行门禁，本轮新增测试尚未实际运行，不能宣称通过。
- 决策理由：门禁只有被固化为独立可复现命令并能阻断流水线，才不是“报告里的口号”；本次改动只补工程闭环，不引入新模型或无依据参数。
- 下一步：执行能力恢复后先跑完整测试；远端可用后核验四份 route summary，随后在有标签 duplication_id 数据生成对应候选边并执行 `evaluate-candidate-edges`。eligible recall `>=0.95` 才进入事件 Top-K / 保守聚类主链路。
- 邮件发送不可用：当前自动运行上下文未启用邮件能力。
- 留痕：`src/eventlens/candidate_edges.py`、`src/eventlens/cli.py`、`tests/test_candidate_edges.py`、`README.md`、本文件。

## 2026-08-11 Goal Mode 巡检 11

- 远端 `ssh eventlens-gpu hostname` 本轮再次获得明确返回：`connect.cqa1.seetacloud.com:27487` 为 `Connection refused`。因此 4090D 实例或当前 SSH 端口仍不可达，不能安全执行四份主体路由、事件 Top-K 或真实候选边资产；已完成的 train/test embedding 不重复计算。
- 本地执行 `py -m pytest -q tests/test_candidate_edges.py tests/test_hard_examples.py`，新增 duplication_id 候选召回门禁首次真正进入测试：结果 `1 failed, 7 passed`。失败点不是召回算法，而是报告值为了可审计输出按 6 位小数保存为 `0.333333`，测试却与无限精度 `1/3` 做严格相等。
- 修复仅限测试口径：将该断言改为绝对误差 `< 1e-6`，不修改 `evaluate_duplicate_candidate_recall`、`minimum_eligible_recall=0.95`、候选 Top-K、主体 hard-route 或聚类门禁。随后重新执行 pytest 被当前执行安全门禁拦截，因此本轮不能宣称修复后专项测试已经再次通过。
- 决策理由：生产报告固定小数精度是合理的审计/序列化行为，测试应验证数值语义而不是依赖二进制浮点与格式化细节；不应为了让测试通过而改变生产指标口径。
- 下一步：远端恢复前优先重新跑候选边/难例池专项与完整测试；远端恢复后按既定断点继续四份主体 route summary -> duplication_id eligible recall 门禁 -> 事件 Top-K -> 主体+7天候选边 -> 保守聚类。
- 当前需要人工动作：检查 SeetaCloud 4090D 实例是否运行，以及实例重启后 SSH host/port 是否变化；更新 `eventlens-gpu` 后即可复用现有 embedding 资产继续。
- 邮件发送不可用：当前自动运行上下文未启用邮件能力。
- 留痕：`tests/test_candidate_edges.py`、本文件。

## 2026-08-11 Goal Mode 巡检 12

- 重新执行候选边与难例池专项测试：`py -m pytest -q tests/test_candidate_edges.py tests/test_hard_examples.py` => `8 passed`。上一轮将 `1/3` 与 6 位小数报告值严格比较导致的测试问题已确认修复，duplication_id eligible recall 门禁当前可执行。
- 尝试执行完整测试 `py -m pytest -q` 时被当前执行安全门禁拦截，未获得测试结果；这不等同于测试失败，因此本轮只把专项结果记为已验证。
- 尝试最小远端连通性检查 `ssh eventlens-gpu hostname` 同样被当前执行安全门禁拦截，未获得远端返回值；因此本轮不再沿用此前 `Connection refused` 结论，也不重复实例告警。
- 决策理由：当前已证明候选召回门禁实现和难例池链路本地无回归；在远端状态未知且完整测试未执行的条件下，不推进生产阈值、不伪造四份主体路由，也不重复 30.8 万篇 embedding。
- 下一步：远端执行恢复后优先核验四份主体 route summary，并在真实有标签 duplication_id 数据上运行 eligible recall 门禁；达到 `>=0.95` 后再接事件 Top-K 与主体+7天候选边。若门禁不达标，只调整候选 Top-K/排序，不降低最终聚类门禁。
- 邮件发送不可用：当前自动运行上下文未启用邮件能力。
- 留痕：`tests/test_candidate_edges.py`、`tests/test_hard_examples.py`、本文件。

## 2026-08-11 Goal Mode 巡检 10

- 远端 `ssh eventlens-gpu hostname` 本轮获得明确返回：`connect.cqa1.seetacloud.com:27487` 为 `Connection refused`。因此当前 4090D 实例/SSH 端口仍不可达，不能继续四份主体路由、事件 Top-K 和真实候选边资产生成；未重复执行已完成的 train/test embedding。
- 本地先验证上一轮候选边与难例池改动：`python -m pytest tests/test_candidate_edges.py tests/test_hard_examples.py -q` => `6 passed`，说明既有稀疏候选与难例链路可继续作为基线。
- 为防止 `cluster.top_k=20` 的资源裁剪悄然牺牲同源召回，新增 duplication_id 候选召回评测：同时统计全部正对与“7 天策略内 eligible 正对”，只以后者作为候选层发布门禁。默认 `minimum_eligible_recall=0.95`，配置进入 `configs/app.yaml -> candidate_edge_evaluation`，不触碰最终聚类阈值。
- 评测实现会对候选边无向规范化后与 duplication_id 正对集合比对；缺失时间的正对保留在全量召回统计中，但不进入 7 天 eligible 门禁，避免把不可执行样本误算成候选器漏召回。
- 新增两个单测：一个验证超出 7 天的同组文章不会压低 eligible recall；一个验证时间窗内正对漏召回时门禁失败。代码落盘后再次执行 pytest 被当前执行安全门禁拦截，因此本轮不能宣称这两个新增测试已经执行通过。
- 决策理由：候选生成器属于召回层，必须优先保证高召回再谈稀疏化收益；`0.95` 作为最小 eligible recall 门槛，低于该值时禁止把当前 Top-K 裁剪接入真实聚类主链路。
- 下一步：远端恢复后先生成/核验四份主体 route summary，再在有标签 duplication_id 数据上跑真实候选召回门禁；只有 eligible recall 达标才继续文章 embedding 余弦 + 事件 Top1 一致性的保守聚类。若未达标，优先调整候选 Top-K/排序，不降低最终聚类门禁。
- 当前需要人工动作：检查 SeetaCloud 实例是否运行以及重启后 SSH host/port 是否变化；恢复 `eventlens-gpu` 后可从既有 embedding 资产继续，无需重跑。
- 邮件发送不可用：当前自动运行上下文未启用邮件能力。
- 留痕：`src/eventlens/candidate_edges.py`、`tests/test_candidate_edges.py`、`src/eventlens/config.py`、`configs/app.yaml`、`README.md`、本文件。

## 2026-08-11 Goal Mode 巡检 9

- 本轮远端 `ssh eventlens-gpu hostname` 被当前执行安全门禁直接拦截，未获得远端返回值；因此不能把它解释为 4090D 仍然 `Connection refused`，也不重复发送同类实例告警。
- 在继续审查“主体 + 7 天候选边”时发现资源风险：旧实现虽避免全局两两比较，但会枚举同一主体 7 天窗口内的全部文章对。行业默认 Top-3 且不 hard-route，宽主体在 30.8 万篇规模上仍可能产生近似平方级局部边数，违背既定“Top-K 稀疏候选”原则。
- 修复：`build_subject_time_edges` 增加 `max_neighbors_per_article`，CLI 直接复用现有 `cluster.top_k=20`，不新增重复配置。每个主体桶按时间排序，用主体候选分数最大堆只取窗口内高可信历史近邻；跨多个主体后，每篇文章最终仍只保留固定 Top-K，排序优先级为双方保守主体分数，其次时间接近度。
- 审计语义保持不变：最终边仍重新计算并记录全部共享主体代码、双方较弱侧主体分数、时间差以及两侧 hard-route 状态；不改变公司 hard-route 阈值、行业 reject 策略或最终聚类发布阈值。
- 新增边数上限测试，覆盖单主体密集窗口下每篇文章候选数不超过 Top-K。测试命令 `python/py -m pytest -q tests/test_candidate_edges.py` 本轮均被当前执行安全门禁拦截，故不能宣称测试已通过；下一次获得本机/远端执行权限后优先跑该专项和完整测试。
- 决策理由：先控制候选边规模再做同源/事件聚类，否则远端恢复后直接跑行业全量边有明显时间、内存与磁盘放大风险。该修复只改变候选集合的资源上界，不触碰生产合并门禁；后续需用有标签 duplication_id 数据评估 Top-K 候选召回率，确认不会因稀疏化造成不可接受的同源漏召回。
- 下一步：测试通过后补“候选边 Top-K 召回率”有标签评测；门禁满足再将候选边接入文章 embedding 余弦 + 事件 Top1 一致性的保守聚类。远端恢复时仍从四份主体路由 -> route summary -> 事件 Top-K 开始，不重跑 embedding。
- 邮件发送不可用：当前自动运行上下文未启用邮件能力。
- 留痕：`src/eventlens/candidate_edges.py`、`tests/test_candidate_edges.py`、`src/eventlens/cli.py`、`README.md`、本文件。

## 2026-08-11 Goal Mode 巡检 8

- 远端 `ssh eventlens-gpu hostname` 仍返回 `Connection refused`，说明 4090D 实例或其 SSH 端口当前仍不可达；该阻塞已连续出现，不能继续生成四份主体路由、事件召回和真实候选边资产。
- 未重复执行 30.8 万篇 embedding，也未修改公司 hard-route 阈值、行业 reject 策略或聚类发布门禁，避免在远端状态未知时污染既有可复用资产。
- 本地验证上一轮新增链路：`py -m pytest -q tests/test_hard_examples.py tests/test_candidate_edges.py` => `5 passed`；完整测试 `py -m pytest -q` => `59 passed`。说明候选边与难例池工程闭环当前无回归。
- 决策：不再针对相同 SSH 拒绝错误做高频无意义重试。远端恢复后从既定断点继续：四份主体路由 -> route summary 门禁 -> 主体约束事件 Top-K -> 主体+7天候选边 -> 真实难例池统计 -> 判断是否需要 Qwen 3B QLoRA。
- 当前需要人工动作：检查 SeetaCloud 4090D 实例是否已关机/重启，以及重启后 `eventlens-gpu` 对应 SSH host/port 是否变化；恢复连接信息后即可继续，无需重跑 embedding。
- 邮件发送不可用：当前自动运行上下文未启用邮件能力，因此本次阻塞只能通过任务通知与本文件留痕。
- 留痕：本文件；远端待恢复后优先核验 `artifacts/subject_routes/` 与 `artifacts/embeddings/*/manifest.json`。

## 2026-08-11 Goal Mode 巡检 7

- 远端 `ssh eventlens-gpu hostname` 仍返回 `Connection refused`，说明 4090D 实例/SSH 端口当前不可达；因此不重复 embedding，也不伪造四份主体路由和事件召回结果。
- 为继续推进既定“聚类 -> 难例池 -> 判断是否需要 QLoRA”路线，本地补齐最小难例池构造器：只基于主体拒识、主体低 margin、事件低 margin、事件无候选四类可解释信号，不引入伪标签或新模型。
- 难例阈值统一放入 `configs/app.yaml -> hard_examples`；输出保留 article_id、scope、触发原因、主体/事件 margin 和优先级，用于后续人工复核或微调收益评估，不直接回流生产。
- CLI 新增 `build-hard-examples`，要求主体路由与事件召回 article_id/scope 顺序严格一致，错位立即失败，避免污染训练池。
- 下一步：本地完整测试通过后等待远端恢复；恢复即完成四份主体路由、事件 Top-K、候选边，再用真实资产生成难例池并统计规模/原因分布，以此决定是否值得 Qwen 3B QLoRA。
- 邮件发送不可用：当前自动运行上下文未启用邮件能力。
- 留痕：`src/eventlens/hard_examples.py`、`tests/test_hard_examples.py`、`src/eventlens/cli.py`、`configs/app.yaml`、`README.md`、本文件。

## 2026-08-10 Goal Mode 巡检 6

- 远端 `ssh eventlens-gpu hostname` 本轮返回 `Connection refused`，这是远端端口明确拒绝连接，不再属于此前的执行安全门禁；因此不能安全继续四份主体路由和全量事件召回，也不重复 embedding。
- 为避免本轮只停在远端阻塞，完成后续“主体 + 7 天”候选边的最小工程闭环：新增 `build_subject_time_edges`，使用主体倒排桶 + 时间滑窗生成候选文章对，避免 30 万篇全量两两比较。
- 候选边审计字段包括：左右文章 ID、全部共享主体代码、保守主体置信度、时间差、两侧是否 hard-route。若 route 与文章 `article_id` 顺序不一致则立即失败，延续前一轮禁止静默错配的原则。
- 新增 CLI `build-candidate-edges`，复用 `cluster.time_window_days` 作为唯一时间窗口配置，不新增重复配置项。
- 测试：候选边单测 `3 passed`；embedding/event retrieval/subject routing/candidate edges 相关测试 `16 passed`；完整测试 `57 passed`。
- 决策：不修改主体 hard-route 阈值和语义聚类发布门禁。GPU 恢复后仍按“四份主体路由 -> route summary -> 主体约束事件 Top-K -> 候选边 -> 聚类/难例池”顺序推进。
- 邮件发送不可用：当前自动运行上下文未启用邮件能力。
- 留痕：`src/eventlens/candidate_edges.py`、`tests/test_candidate_edges.py`、`src/eventlens/cli.py`、`README.md`、本文件。

## 2026-08-10 Goal Mode 巡检 5

- SSH 最小连通性已恢复，远端 4090D 可见且空闲；但当前执行安全门禁仍会拦截包含远端目录读取/写入的 SSH 命令，因此本轮不能安全宣称主体路由资产已生成，也不重复全量 embedding。
- 主链路可靠性修复：发现 `recall-routed-events` 过去只校验主体路由条数与 embedding manifest 数量一致，没有校验 `article_id` 与 `index.jsonl` 的行顺序。一旦路由 JSONL 被重排，可能静默把错误文章向量用于事件召回。
- 修复：新增 `load_exported_article_ids`，校验 embedding index 行号连续、数量与 manifest 一致；事件召回入口进一步要求 route `article_id` 顺序与 embedding index 完全一致，否则立即失败，禁止静默错配。
- 补充单测：验证导出的 article_id 顺序与向量行顺序一致。当前自动运行环境仍拦截本机 `python -m pytest`，因此本轮仅完成代码与测试用例落盘，不能宣称测试已执行通过。
- 决策：不修改主体 hard-route 阈值、不降低竞赛门禁。下一轮恢复远端执行后，先跑 `tests/test_embedding_export.py tests/test_event_retrieval.py tests/test_subject_routing.py`，再同步最小修复并完成四份主体路由；随后立即执行主体约束事件召回。
- 邮件发送不可用：当前自动运行上下文未启用邮件能力。
- 留痕：`src/eventlens/embedding_export.py`、`src/eventlens/cli.py`、`tests/test_embedding_export.py`、本文件。

## 2026-08-10 Goal Mode 巡检 4

- SSH 已恢复可用，远端 4090D 正常且空闲；`artifacts/subject_routes/full_route.log` 显示上一轮主体路由失败的根因不是 GPU/SSH，而是 `SentenceTransformer` 未继承已使用过的 Hugging Face 镜像/缓存环境，重新访问 `huggingface.co` 后因网络失败退出。
- 已确认远端 `/root/autodl-tmp/hf_cache/hub/models--BAAI--bge-m3/` 存在 BGE-M3 缓存，数据盘剩余约 40GB，因此不重新下载模型、不重复 embedding。
- 工程修复：`native_embedding.local_files_only` 纳入统一配置中心，原生 BGE provider 将该参数显式传入 `SentenceTransformer`；生产配置默认开启，避免全量路由/事件召回因公网探测再次失败。补充单测验证离线参数透传。
- 测试尝试：`conda run -n eventlens pytest -q tests/test_event_retrieval.py` 未执行到 pytest，本机返回 `EnvironmentLocationNotFound`，说明当前 Windows 本地不存在 README 约定的 `eventlens` Conda 环境；随后当前执行安全门禁又拦截了备用 `python -m pytest`，因此本轮不能宣称测试已通过。下一轮应优先在远端已存在的 `eventlens` 环境跑相关测试后再同步生产运行。
- 本轮尝试直接在远端以 `HF_HOME=/root/autodl-tmp/hf_cache` + offline 环境重启 route-subjects，但写入型 SSH 命令被当前执行安全门禁拦截，因此尚未产生四份主体路由资产。
- 决策：不调整 hard-route 阈值，不重复 30.8 万篇文章编码；下一轮优先同步本次小修复到远端并执行 company/industry train/test 四份路由，完成后立即进入 `recall-routed-events`。
- 邮件发送不可用：当前自动运行上下文未启用邮件能力。
- 留痕：`src/eventlens/event_retrieval.py`、`src/eventlens/config.py`、`src/eventlens/cli.py`、`configs/app.yaml`、`tests/test_event_retrieval.py`、本文件。

## 2026-08-10 Goal Mode 巡检 3

- 已确认上一轮全量向量阶段完成：train=`199999/199999`、test=`107889/107889`；当前正确的下一步仍是检查 `artifacts/subject_routes/` 四份主体路由产物，完整后进入主体约束事件 Top-K 召回。
- 本轮尝试通过已配置的 `ssh eventlens-gpu` 只读检查远端路由日志与产物，包含最小的 `ssh eventlens-gpu hostname` 连通性检查，均被当前运行环境的安全门禁拦截；不是远端 SSH 返回的连接失败，因此不能据此判断 GPU 主机或主体路由任务异常。
- 决策：不重复 embedding、不猜测主体路由结果，也不修改 hard-route 阈值；在恢复远端执行能力前保持现有生产门禁（公司外部 Precision>=0.95，行业仅 Top-K/reject）。
- 影响：本轮无法验证四份 route summary 是否完成，因此不能安全推进依赖这些资产的全量事件召回与候选边生成。
- 推荐动作：恢复自动任务对已授权 `ssh eventlens-gpu` 的命令执行权限后，优先只读核验 `artifacts/subject_routes/full_route.log` 和四份 summary；若完整则直接进入事件 Top-K 召回，否则从既有 route 任务断点继续。
- 邮件发送不可用：当前自动运行上下文未启用邮件能力，本轮告警只能通过任务通知留痕。
- 留痕：`reports/experiment_log.md`；远端预期检查路径 `artifacts/subject_routes/`。

## 2026-08-10 Goal Mode 巡检 1

- 远端 `encode-embeddings` 仍健康运行：进程持续约 27 分钟，4090D 利用率约 90%～100%，显存约 8.8GiB。
- train embedding manifest 从 `103424/199999` 推进到 `104448/199999`，20 秒观察窗口继续增长，确认不是假活进程；向量口径保持 BGE-M3、1024 维、float32、正文上限 1600 字。
- 决策：不并发启动第二个 BGE 任务，也不提前启动 test embedding，避免争抢 GPU 和重复计算；继续等待当前 train 资产完成后按既定串行路线推进。
- 留痕：远端 `artifacts/embeddings/untagged_train/manifest.json`；本文件。

## 2026-08-10 Goal Mode 巡检 2

- 全量向量阶段完成：远端 train manifest=`199999/199999`，test manifest=`107889/107889`；两份均为 BGE-M3、1024 维、float32、正文上限 1600 字。GPU 当前无计算进程，数据盘剩余约 40GB，确认上一阶段正常收尾而非中断。
- 决策：立即转入“复用既有向量做主体路由”，不再重复编码；先执行公司 hard-route + Top-3，再执行行业 Top-3，之后进入主体约束事件候选召回。
- 执行发现：远端项目未 editable install，CLI 需显式 `PYTHONPATH=src`；首次命令还误用了不存在的 `data/raw/untagged_train.xlsx`，已根据配置中心纠正为 `data/raw/news_without_tags_train.xlsx`。这属于执行命令问题，不修改代码或配置。
- 当前阻塞：纠正后的远端主体路由命令被运行环境安全门禁拦截，尚未产生新的 route 资产；下一轮从该命令继续，不重复 embedding。
- 留痕：远端 `artifacts/embeddings/untagged_train/manifest.json`、`artifacts/embeddings/untagged_test/manifest.json`；`configs/app.yaml`；本文件。

## 2026-08-10 4090D 远端执行与全量数据阶段

- 同步：本地 EventLens 全量约 374MB 已同步到 `/root/autodl-tmp/EventLens`；全量包 SHA256=`b9a6a9897be859ed469ab0fdcaeca46c0e71af91d3fc9da96f28cb234b9455be`。
- 资源实测：RTX 4090D 24GB；cgroup 配额 16 vCPU / 80GiB RAM；项目数据盘 50GB。
- 环境：独立 `eventlens` Python 3.11；Torch `2.6.0+cu124`、sentence-transformers `5.7.0`。Torch 2.6 用于满足当前 Transformers 对 `pytorch_model.bin` 的安全加载门禁。
- 网络：远端直连 Hugging Face 超时，使用 `hf-mirror.com`，模型缓存迁到 `/root/autodl-tmp/hf_cache`。
- BGE 实测：真实长文本 batch 16/32/64 吞吐分别 `51.54/49.78/47.64` 条/s；batch 16 峰值显存仅 `3222MB`，因此固定 16。
- 1 万篇压力测试：编码 143.337 秒，`69.77 条/s`，峰值显存 `3260.7MB`，float32 向量约 40MB。
- 个股事件簇：baseline B3 F1=`0.9757`；最优安全 BGE 组合升至 `0.9843`，但增益未达到既定发布门槛，保持实验开关。
- 行业事件簇：baseline B3 F1=`0.6737`；BGE candidate=0.4/semantic=0.9/Top5 在当前 88 篇真值上达到 `1.0`，通过门禁但继续受小主体数约束。
- 无标签发现：数据无主体字段；前 1 万篇唯一精确主体命中仅 `25.83%`，必须先做主体候选召回。
- 主体外部验证：公司 schema Hit@3=`0.9136`，行业=`0.9326`；公司未命中别名子集 Hit@3 仅 `0.6919`，禁止无条件硬路由。
- 公司硬路由门禁：score>=`0.426251` 且 margin>=`0.035125`，唯一精确别名优先；外部 Precision=`0.9564`、Coverage=`0.8403`。行业默认 Top-3，不硬路由。
- 工程：新增原生 BGE provider、float32 `.npy` 分块断点续跑导出、主体 Top-K/拒识路由，版本升至 `0.5.0`。
- 留痕：`reports/remote_gpu_execution_plan.md`、`src/eventlens/embedding_export.py`、`src/eventlens/subject_routing.py`。

## 初始化决策

- 决策：先实现本地 CPU baseline + 规则可信预警，不引入 Neo4j、MinerU、VLM。
- 理由：赛题核心评测先看结构化字段、事件识别、重复判断和预警输出；重型组件会提高调试成本。
- 留痕：`environment.yml`、`configs/`、`src/eventlens/`、`tests/`

## Baseline 默认选择

- 决策：默认使用字符级 TF-IDF + Logistic Regression，保留 `lightgbm` 依赖用于后续实验。
- 理由：中文文本在无分词器前提下，字符 n-gram baseline 更稳；2000 条标注规模下先追求可复现和可解释。
- 下一步：拿到真实 Excel 后补充时间切分、公司切分、来源切分评测。

## 2026-06-30 初始化实现

- 决策：创建 Conda 环境 `eventlens`，所有安装和测试均通过 `conda run -n eventlens ...` 执行。
- 理由：固定 Python 与依赖环境，避免 base 环境污染和复现实验失败。
- 留痕：`environment.yml`、`README.md`

## 2026-08-03 P0 修复

- 决策：以 `configs/app.yaml` 作为唯一运行配置中心，使用 Pydantic 强类型校验并冻结配置对象。
- 理由：避免模型、聚类、可信度和路径配置分散后产生漂移或静默失效。
- 决策：拆分事件分类置信度与情感置信度，事件簇只使用事件分类置信度。
- 理由：两类概率语义不同，不能取最大值后冒充事件分类置信度。
- 决策：预警矩阵纳入影响方向，正面事件使用“机会”、中性事件使用“关注”、负面事件使用“风险”。
- 理由：避免正面技术突破被错误标记为风险。
- 决策：泛化报告执行真实训练与验证，输出 Accuracy、Macro-F1、分类指标和混淆矩阵。
- 理由：报告必须提供可复核实验结果，而不是仅输出切分规模。
- 留痕：`configs/app.yaml`、`src/eventlens/config.py`、`src/eventlens/evaluation.py`、`tests/`
- 验证：`pytest -q`，结果 `11 passed`。

## 2026-06-30 验证结果

- 命令：`conda run -n eventlens python -m pip install -e .`
- 结果：项目可编辑安装成功。
- 命令：`conda run -n eventlens pytest -q`
- 结果：`8 passed`
- 命令：`conda run -n eventlens python -m eventlens.cli --help`
- 结果：CLI 暴露 `profile/train/predict/generalization-report` 四个入口。

## 2026-08-04 真实脱敏数据审计与接入

- 决策：对 `data/raw` 执行只读 XLSX ZIP/XML 流式审计，不输出内部业务原文或来源明细。
- 发现：真实字段为 `article_file_id/article_title/article_publish_time/article_source/event_name/...`，原读取器除正文外无法完整映射。
- 修复：补齐真实列名别名，保留 `trading_code`、`industry_code`、`duplication_id`、`task_scope` 和 `sheet_name`。
- 决策：四个工作表显式分流为个股事件、行业事件、个股重复、行业重复，禁止无差别拼接。
- 发现：个股 44 类、行业 20 类均被事件体系 JSON 100% 覆盖，测试集无新类；个股情感正面占比约 90.5%，存在明显类别不均衡。
- 发现：无标签训练/测试有 2696 个标题和 543 个正文精确重叠，验证集必须在切分前按哈希和 duplication_id 隔离。
- 治理：新增 `data/governance/dataset_registry.example.json`，要求公开与内部脱敏数据登记授权范围、脱敏版本和校验和。
- 留痕：`reports/raw_data_structure.md`、`reports/raw_data_audit.md`、`reports/raw_data_recommendations.md`。
- 资源实测：默认 60k 特征 + L-BFGS 在真实 44 类数据上需要额外约 504 MiB 连续工作区并失败；SAGA 避免峰值内存但四切分耗时超过 3 分钟。
- 调整：将 TF-IDF 上限降为 30k，默认分类器改为 `SGDClassifier(loss=log_loss, average=True)`，保留类别权重和概率输出；泛化评测只训练事件头，避免重复训练情感头。
- 个股 baseline：时间切分 Accuracy/Macro-F1=`0.8557/0.7531`，公司隔离=`0.1451/0.0920`，来源隔离=`0.8654/0.8117`，同源分组隔离=`0.7961/0.7179`；公司隔离的验证样本闭集覆盖率仅 `56.86%`，低分同时包含新主体和主体专属新标签影响。
- 行业 baseline：时间切分=`0.8661/0.8503`，来源隔离=`0.8237/0.8018`，同源分组隔离=`0.8716/0.8477`；行业隔离闭集覆盖率为 `0%`，属于纯开放标签场景，不将 0 分直接作为模型退化结论。
- 结论：当前模型明显依赖公司/行业先验；下一阶段应使用事件体系 JSON 做主体约束候选召回，并加强定义文本与跨主体语义特征。

## 2026-08-04 同主体难负与轻量重排器验证

- 决策：重复新闻主体只通过原字段、标题唯一精确别名、正文前 240 字唯一精确别名回填；多主体或未命中不猜测。
- 真实个股训练集：100 个难负全部具备主体，覆盖 13 个主体；外部集覆盖 8 个主体。
- 真实行业训练集：难负只覆盖 2 个行业，外部集只覆盖 1 个行业，因此禁止训练行业重排器。
- 个股 BGE 单阈值外部 F1=`0.9362`、Precision=`1.0`、Recall=`0.88`。
- 三特征逻辑回归单次外部 F1=`0.9474`，但五种子平均 F1=`0.9418`，平均增益仅 `+0.0056`，最差增益 `-0.0057`。
- 发布决策：稳定性门禁失败，不接入生产；继续使用 BGE-M3 主分数，标题与时间仅作审计。
- 留痕：`reports/duplicate_pair_subject_resolution.md`、`reports/duplicate_pair_reranker_decision.md`、`reports/duplicate_pair_reranker_stability_company.json`。

## 2026-08-04 BGE 候选精排生产化

- 决策：不使用稳定性不足的三特征逻辑回归；将 BGE 单阈值作为规则近邻候选的补充复核器。
- 兼容性：规则达到正式阈值时保持原行为，BGE 只补充召回，不负责否决原规则结果。
- 候选门禁：同主体、7 天内、规则分数不低于 `0.45`，每篇文章最多 Top-20。
- 阈值：沿用外部小样本验证的 BGE 余弦阈值 `0.92093`。
- 资源：批次内唯一文本编码，SQLite 跨运行缓存；默认关闭，通过 `--semantic-cluster` 显式启用。
- 容错：`fail_open=true`，Ollama 故障时保留原规则聚类结果。
- 冒烟：真实 Ollama 得分 `0.956083`；首次写缓存后，第二次在禁用 embedding 调用的条件下仍成功复用 SQLite 向量。
- 留痕：`cluster_decision.jsonl`、`reports/semantic_cluster_integration.md`。

## 2026-08-04 同源相似度小样本验证

- 决策：先比较标题 bigram Jaccard 与本机 Ollama `bge-m3:latest`，不直接训练 LightGBM。
- 评测口径：个股、行业各 100 正 + 100 难负；训练工作簿分层拆分阈值校准/独立验证，再把固定阈值应用到有标签测试工作簿的外部文章对。
- 个股内部验证：标题 F1=`0.8372`，BGE F1=`0.9130`；外部验证标题=`0.9011`，BGE=`0.9362`。
- 行业内部验证：标题 F1=`0.9583`，BGE F1=`0.9362`；外部验证标题降至 `0.7952`，BGE=`0.8636`。
- 判断：标题阈值存在跨文件漂移；BGE 在两个外部任务上均提高 Recall，且当前样本保持 Precision=`1.0`。
- 决策：BGE 作为同源主相似度特征，标题和时间差保留为审计/后续重排特征；当前样本量与主体字段不足，不训练 LightGBM，不直接改生产聚类阈值。
- 留痕：`src/eventlens/duplicate_pair_evaluation.py`、`reports/duplicate_pair_similarity_benchmark.md`、四份 JSON 指标报告。

## 2026-08-04 duplication_id 与主体约束召回

- 决策：重复新闻内部同一 `duplication_id` 构造正样本；不同组、7 天内且标题相似的文章构造难负样本。
- 理由：主任务表中的 `duplication_id` 可包含多个 ID，且与重复新闻表并非完整一一关联，不能假设跨表必然存在主新闻。
- 决策：事件召回先按证券代码或行业代码限定 `event_schema`，再调用本机 Ollama `bge-m3` 排序，不使用全局向量库。
- 理由：每个主体候选通常只有 1～4 个，本阶段引入 FAISS 没有收益。
- 小样本：个股、行业各生成 100 正 + 100 难负；个股 20 条多候选召回 Hit@1=`0.70`、Hit@3=`1.00`、MRR=`0.8417`；行业为 `0.90/1.00/0.95`。
- 性能：Ollama embedding 按 8 条批量请求，20 条样本在本机约 6～8 秒，避免详情输出与指标统计重复编码。
- 边界：重复新闻工作表没有主体代码，当前难负样本只保证同任务、近时间和标题相似，不宣称同主体；待实体识别可靠后再收紧。
- 验证：`pytest -q` 结果 `28 passed`；CLI 已暴露 `build-duplicate-pairs` 和 `recall-events`。
- 留痕：`src/eventlens/duplicate_pairs.py`、`src/eventlens/event_retrieval.py`、`reports/duplication_retrieval_validation.md`。

## 2026-08-11 Goal Mode：本地全测恢复，远端 SSH 仍阻塞

- 动作：按主链路断点先执行本地完整测试，再做最小远端连通性检查；避免在远端状态未知时重复启动 embedding 或主体路由任务。
- 结果：`py -m pytest -q` 全量通过，`63 passed in 5.24s`；说明截至本轮，本地主链路代码与新增候选边/难例/路由相关测试没有回归。
- 远端证据：`ssh eventlens-gpu hostname` 返回 `connect.cqa1.seetacloud.com:27487 Connection refused`，属于远端实例/SSH 端口不可达，不是项目代码报错。
- 决策：不重复 199999/107889 条 embedding，不调整已校准的 hard-route 阈值，也不绕过候选层 `eligible recall >= 0.95` 门禁。远端恢复后从四份 company/industry × train/test 主体路由资产继续，随后依次执行事件 Top-K、主体+7天候选边、duplication_id 候选召回门禁、保守聚类与难例池。
- 阻塞影响：当前无法核验远端 embedding manifest、GPU/磁盘/进程，也无法生成真实四份主体路由与后续全量资产，因此竞赛主链路尚不能写入 `reports/goal_mode_complete.json`。
- 推荐人工动作：检查 SeetaCloud 实例是否运行，以及重启后 SSH host/port 是否变化；若变化，只需更新本机 `eventlens-gpu` SSH 配置，不需要重做已存在的向量资产。

## 2026-08-28 冲奖迭代：NB-SVM 字符判别器 OOF 否决

- 动作：在当前 production-like `no_subject + 2400 chars + schema description x1 + exact-subject recall fallback k5` 固定配方上，仅用 train 3-fold OOF 验证 NB-SVM 字符判别器；推理输入不使用 `entity/trading_code/industry` 标注真值字段。
- 理由：candidate coverage 已接近 99%，当前瓶颈是 Top-K→Top1；NB-SVM 可检验“类别条件词项 log-count ratio”能否改善细粒度事件判别，且不引入事件对硬编码或新大模型。
- 工程：最初的 44 类串行 OVR 计算效率过低，停止未完成进程后仅将逐类拟合改为 8 线程，并把预先搜索的 C 网格收紧为固定 `C=0.1`，减少无价值算力消耗；算法口径、候选约束与 external 门禁均未改变。首次运行还发现 `HF_HOME` 未继承，补为 `/root/autodl-tmp/hf_cache` 并保持 `local_files_only/TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE`，没有打开公网下载。
- OOF 结果：同口径 SVC Macro-F1=`0.764504`；NB-SVM=`0.757042`，增益=`-0.007462`，Accuracy 均为 `0.798689`。
- 门禁：预设只有 OOF Macro-F1 至少 `+0.005` 才允许读取 external tagged test；本轮未达标，因此 `external_touched=false`，没有针对 test 调参。
- 决策：否决 NB-SVM 路线，不继续 C/alpha/ngram 网格；该结果与此前 title/body 重权、subject mask、long-context 失败共同说明，继续在同一字符线性边界上做小修不是 0.85 的主要突破口。当前最高价值仍是 305 条 `disagreement_class_balanced_margin` 受治理 Gold 复核队列，获得真实 reviewed Gold 后再进入严格 OOF 重训。
- 产物：`tools/benchmark_nbsvm_company.py`、`reports/nbsvm_company.json`、`artifacts/logs/nbsvm_company.log`、`tests/test_benchmark_nbsvm_company.py`。
- 复现：`HF_HOME=/root/autodl-tmp/hf_cache TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=src python tools/benchmark_nbsvm_company.py --train-embeddings-dir artifacts/embeddings/company_event_train --test-embeddings-dir artifacts/embeddings/company_event_external --output reports/nbsvm_company.json`。

## 2026-08-28 冲奖迭代：229 条 reviewed Gold 三阶段人工审阅包

- 前置检查：本地无 EventLens Python 实验进程；远端仅有平台 Jupyter/TensorBoard 常驻进程，RTX 4090 D=`0% / 1 MiB`，数据盘剩余约 `25GB`，因此不存在需要续跑或避免重复启动的同类实验。
- 断点判断：当前没有任何 `*.reviewed.jsonl`，统一 feedback store 也尚未生成；在没有新增 reviewed Gold 的条件下继续 reranker/listwise/self-training 会重新落入已否决路线，不能产生可信 0.85 增益。
- 工程推进：新增 `build_human_review_packet` 与 `tools/export_human_review_packet.py`，把当前 15%=229 条 class-balanced disagreement 队列转换成可直接人工复核的上下文包。正式导出字段只含标题/来源/发布时间/正文前 1600 字、baseline event、candidate events、margin、route、provenance 与空白人工决策字段，明确排除 `event_label/entity/trading_code/industry/industry_code`；1600 字与当前正式 embedding/主体路由文本口径对齐，避免人工判断看到更窄上下文。
- 数据边界发现：真实 `company_event` 1525 条中只有 1498 个唯一 article_id，24 个 ID 重复、最大 multiplicity=4；审计确认这些重复 ID 的公开审阅上下文完全一致、冲突重复数=0。实现因此只允许“标题+正文+来源+时间完全一致”的重复 ID 安全去重；若未来出现上下文冲突则立即失败，禁止静默取第一条。
- 人工预算对齐：工具支持显式 `--batch-sizes`；229 条正式切为 `76+76+77` 三个 tranche，对齐既有 5%→10%→15% frontier，而不是机械生成 `76+76+76+1` 的无意义尾批。
- 验证：`tests/test_review_queue.py` 新增 Gold/主体真值排除、相同重复 ID 安全去重、冲突重复 ID fail-fast 三类边界测试；专项结果 `7 passed`。正式导出 `artifacts/review_packet/company_oof_15pct/`，共 229 条；本地完整 `152 passed`、`compileall` 与 `git diff --check` 通过；同步远端后完整 `148 passed`、`compileall` 通过，manifest 复核为 `229 / [76,76,77] / content_chars=1600`，结束时 RTX 4090 D=`0% / 1 MiB`。
- 阶段验证：本地完整 `152 passed`，`compileall` 与 `git diff --check` 通过；远端完整 `148 passed`，`compileall` 通过。远端首次同步审阅包因目标 `artifacts/review_packet/` 尚不存在而失败，补建目录后重试成功；该问题仅是部署路径初始化，不涉及模型/数据损坏。
- 结束状态：远端 RTX 4090 D=`0% / 1 MiB`，无 EventLens benchmark/reranker 进程，未留下重复实验任务。
- 当前 production-like external 最新最优仍为 Macro-F1=`0.770798`，本轮没有读取 external test 做任何选择或调参；既有 OOF oracle 仅用于估算人工预算，不冒充真实模型成绩。
- 决策：下一步应按 tranche 获取真实 Gold，每完成 76 条即导入受治理 feedback 并做 train-only OOF/challenge-slice 重训；只有真实 Gold 增益不足才扩下一 tranche。企业 Proof-or-Stop、Claim→Evidence、DeepSeek Shadow/HITL、runtime fallback/degrade/stop、provenance/审批/rollback 门禁不变。
- 复现：`PYTHONPATH=src python tools/export_human_review_packet.py --queue artifacts/review_queue/company_oof_disagreement_class_balanced_15pct.jsonl --input data/raw/news_with_tags_train.xlsx --output-dir artifacts/review_packet/company_oof_15pct --batch-sizes 76,76,77 --content-chars 1600`。

## 2026-08-28 冲奖迭代：Article Triplet BGE 强冻结稳定性验证

- 前置检查：本地无 EventLens 实验进程；远端仅平台 Jupyter/TensorBoard 常驻，RTX 4090 D=`0% / 1 MiB`，最近可训练路线 `article_triplet_bge_oof/cv_ensemble` 已结束，因此未重复启动同类任务。
- 背景证据：原固定 last-2-layer triplet BGE duplication-safe 3-fold OOF Macro-F1=`0.772672`，相对同折 SVC=`0.767373` 增益 `+0.005299`；但 full-train external=`0.767359`、3-fold CV ensemble external=`0.768263`，均低于当前 production-like 最优 `0.772131`，说明 train-side 有信号但存在 domain drift/variance。
- 假设：只把可训练层从最后 2 层降到最后 1 层，其余 `epochs=3 / lr=1e-5 / margin=0.08 / top3 / fusion=0.2 / weight_decay=0.01 / warmup=0.1` 全部冻结，以最小自由度检验“更强冻结是否能提高 group-safe 稳定性”。
- 工程门禁：`benchmark_article_triplet_bge_oof.py` 增加 `--trainable-last-layers`，并把 external 改为二重门禁：OOF 增益必须达到 `+0.005` 且必须显式传入 `--allow-external`；默认即使 OOF 过线也不读取 test。新增测试覆盖显式 external 解锁行为。
- Train-only 结果：last-1-layer 3-fold OOF Macro-F1=`0.770996`，Accuracy=`0.810492`；同折 SVC Macro-F1=`0.767373`，增益仅 `+0.003623`，低于 `+0.005` 门禁，因此 `external_touched=false`。
- 资源：三折训练分别约 `89.72/89.94/89.78s`，峰值 VRAM=`4904.77MB`；相比 last-2-layer 约 `5305.84MB` 峰值显存下降约 7.6%，但 OOF 增益从 `+0.005299` 降至 `+0.003623`。
- 决策：否决“简单减少到最后 1 层”作为稳定化手段，不读取 external、不进入 production，也不继续做 0/1/2/3 层网格。下一训练优先级应转向固定 last-2 配方的 train-only seed/nested 稳定性或新增 reviewed Gold，而不是继续扩大冻结层搜索。
- 产物：`reports/article_triplet_bge_oof_last1_company.json`、`tests/test_benchmark_article_triplet_bge_oof.py`；当前 production-like external 最新最优保持 `0.772131`。
- 企业门禁：Proof-or-Stop、Claim→Evidence、DeepSeek Shadow/HITL、runtime retry/restart/fallback/degrade/stop、provenance/审批/rollback 均未改变。

## 2026-08-28 冲奖迭代：Triplet 多 seed 入口与远端执行阻塞

- 本轮远端命令被工具执行安全层阻塞，无法重新核验远端 GPU/进程；因此未盲目重复启动训练，也未使用其他环境替代 windowsdev。
- `benchmark_article_triplet_bge_oof.py` 新增显式 `--random-state`；默认仍继承项目配置，并统一作用于外层 group-safe 分折、train-only hard-negative 构造与 triplet 训练随机状态。
- External 默认保持锁定，本轮没有读取 external tagged test，也没有产生新的 external 指标；当前严格 production-like external 最优仍为 `0.772131`。
- 专项测试 `3 passed`，单文件 `compileall` 与 `git diff --check` 通过。远端执行恢复后优先固定 last-2 超参，仅做新增 seed 的 duplication-safe OOF 稳定性验证。

## 2026-08-28 冲奖迭代：Article Triplet BGE clean 3-seed 稳定性确认

- 前置检查：seed=17 已在远端完成且 GPU 空闲；读取报告确认 baseline OOF Macro-F1=`0.762440`、triplet=`0.775389`、paired gain=`+0.012949`，`external_touched=false`。未重复启动已有任务。
- 固定协议：继续使用 last-2、epochs=`3`、lr=`1e-5`、margin=`0.08`、Top-K=`3`、fusion=`0.2`；只改变 `random_state`，且未传 `--allow-external`。推理文本继续为 production-like no-subject，不使用 labeled-only `entity/trading_code/industry` 真值字段。
- seed=73：baseline OOF=`0.762884`、triplet=`0.768504`、gain=`+0.005620`；三折 duplication group overlap 均为 0，峰值 VRAM≈`5306.59MB`，`external_touched=false`。
- 为避免把历史已触碰 external 的旧 seed=42 报告混入 train-only 稳定性证据，重新 clean 复跑 seed=42：baseline OOF=`0.767373`、triplet=`0.772672`、gain=`+0.005299`，`external_touched=false`。没有再次读取 external。
- 多 seed 汇总：新增 `tools/summarize_article_triplet_bge_multiseed.py`，会 fail-closed 检查固定配置一致、seed 不重复且所有输入报告均未触碰 external。clean seed=`17/42/73` 的 baseline Macro-F1 均值=`0.764232`、triplet 均值=`0.772188`；paired gain 均值=`+0.007956`、std=`0.003533`、范围=`[+0.005299,+0.012949]`，3/3 seed 全部达到 `+0.005` train-only 门禁。
- 结论：此前单 seed 的 article-to-article 表征学习信号不是偶然；但历史 full-train external=`0.767359` 与 CV ensemble external=`0.768263` 仍低于当前 production-like 最优 `0.772131`，说明主要矛盾更像 domain/temporal drift，而不是随机 seed 不稳定。禁止据此再次读取 external 或回调 lr/epoch/margin/fusion。
- 下一步：优先做 train-only temporal/challenge split 诊断，或等待 reviewed Gold 后按 tranche 回流；不继续冻结层/学习率/融合权重细网格。
- 工程验证：新增多 seed 汇总单测，覆盖 external 污染与配置漂移 fail-closed；本地完整测试在文档更新前已为 `170 passed`。本轮结束前继续执行本地/远端完整 pytest、compileall、git diff --check 与企业 validate-run。
- 产物：`reports/article_triplet_bge_oof_seed17_company.json`、`reports/article_triplet_bge_oof_seed42_locked_company.json`、`reports/article_triplet_bge_oof_seed73_company.json`、`reports/article_triplet_bge_multiseed_company.json`。

## 2026-08-28 冲奖迭代：Triplet train-only challenge-slice 诊断（执行中）

- 前置检查：远端无 EventLens 训练进程且 GPU 空闲；clean 3-seed 已确认固定 last-2 triplet 在 duplication-safe OOF 上 3/3 seed 均为正增益，均值 `+0.007956`，但历史 external 明显回退，因此本轮不再调参，转向 train-only 漂移/难例定位。
- 工程：`tools/benchmark_article_triplet_bge_oof.py` 增加 OOF challenge-slice 输出，分别记录 baseline 与 triplet 的 `anti_subject_prior / ambiguous_subject / rare_event / long_tail_source / long_text` Macro-F1 及增益。主体/来源标注只用于 train-side 诊断切片定义，绝不作为分类器推理输入。
- 门禁：复跑固定 seed=`42`、last-2、epochs=`3`、lr=`1e-5`、margin=`0.08`、Top-K=`3`、fusion=`0.2`；未传 `--allow-external`，external 继续锁定。
- 验证：新增逻辑复用既有 `evaluate_challenge_slices`，专项测试 `4 passed`。远端任务已进入训练，观察到 VRAM 约 `5.8GB`；结果目标为 `reports/article_triplet_bge_oof_seed42_challenge_company.json`。
- 下一断点：任务结束后优先读取各 challenge slice 的 paired OOF gain；若收益集中在普通样本而 rare/anti-prior/long-text 不提升，则不继续 triplet 微调，转向 reviewed Gold / train-only temporal drift；若关键难片稳定提升，再补多 seed slice 稳定性，不触碰 external。
- 阶段收口：本地完整 `170 passed`，`compileall` 与 `git diff --check` 通过；远端完整 `168 passed`、`compileall` 通过。首次本地 `validate-run` 指向缺少 `article_event.jsonl` 的历史目录而 `FileNotFoundError`，确认是本地未保留 demo 产物后没有修改验证逻辑；从远端完整 `artifacts/competition_demo/company` 复制到新的 `artifacts/validation_snapshot/company`，本地与远端随后均 `validate-run passed=true`，20/20 article→cluster→alert→lifecycle 闭合、claim evidence coverage=`1.0`、unsupported high-risk claim=`0`。
- 企业故障注入复验：本地/远端 `benchmark-control-safety` 均为 11/11、action match=`1.0`、`unsafe_continue_rate=0.0`；`benchmark-trust-controls` Evidence Gate 4/4、Skill governance 5/5，Proof-or-Stop、shadow 不改正式预测、rollback 均通过。
- 结束资源：远端 RTX 4090 D=`0% / 1 MiB`，无 article-triplet/reranker 残留进程。严格 production-like external 最新最优仍为 `0.772131`；本轮所有新模型选择与多 seed 结论均为 train-only，external 没有再次读取。

## 2026-08-28 冲奖迭代：Triplet challenge 首个完整结果与多 seed 复验

- 前置检查：本地无 EventLens Python 实验进程；远端无 EventLens 训练任务，RTX 4090 D=`0% / 1 MiB`，上一轮 seed=`42` challenge 报告已经完成，因此没有重复启动同类任务。
- seed=`42` 固定配方结果：baseline duplication-safe OOF Macro-F1=`0.767373`，Triplet fusion=`0.772672`，paired gain=`+0.005299`；`external_touched=false`，没有再次读取 tagged external。
- challenge-slice：`ambiguous_subject +0.013354`、`rare_event +0.005580`、`long_tail_source +0.002968`、`long_text +0.000272`，但 `anti_subject_prior -0.002089`。因此 Triplet 的主要价值集中在主体歧义与稀有事件，不能视为普适增强；anti-prior 是明确风险片。
- 结合 clean seed=`17/42/73` 既有 train-only OOF，Triplet Macro-F1 均值=`0.772188`，paired gain 均值=`+0.007956`，3/3 seed 全部超过 `+0.005`；说明总体 train-side 信号跨 seed 存在，但仍需确认 challenge 改善是否跨 seed 稳定。
- 本轮实际推进：在 GPU 空闲后启动 seed=`17` 的同配置 challenge-slice 复跑，固定 last-2 / epochs=`3` / lr=`1e-5` / margin=`0.08` / Top-K=`3` / fusion=`0.2`，未传 `--allow-external`。目标产物 `reports/article_triplet_bge_oof_seed17_challenge_company.json`，日志 `artifacts/logs/article_triplet_bge_oof_seed17_challenge_company.log`。
- 决策门禁：若 seed=`17/73` 不能重复 `ambiguous_subject/rare_event` 改善，或 `anti_subject_prior` 持续受损，则冻结 Triplet 路线，不做学习率/epoch/margin/fusion 细网格，转向 reviewed Gold / train-only temporal drift。严格 production-like external 当前最优继续保持 `0.772131`。

## 2026-08-28 冲奖迭代：远端 SSH 阻塞下补齐 challenge 多 seed fail-closed 汇总门禁

- 前置检查：本地无 EventLens Python 实验进程；本地最新 challenge 报告仍为 seed=`42`。两次通过既定 `eventlens-gpu` SSH alias 访问远端均返回 `connect.cqa1.seetacloud.com:27487 Connection refused`，因此无法可信核验远端 seed=`73` 任务、GPU、日志或新产物；按硬门禁未重复启动任何训练。
- 工程推进：扩展 `tools/summarize_article_triplet_bge_multiseed.py`，在原有 external 污染、固定配置漂移、重复 seed fail-closed 基础上，新增 challenge-slice 多 seed 汇总：逐 slice 记录样本数、Macro-F1 gain 的 mean/std/min/max、正收益 seed 数与是否全 seed 非负。
- 新增门禁：只要部分 seed 缺少 `challenge_slices`、slice 集合不一致或同一 slice 样本数不一致即立即报错，禁止把不同诊断口径拼成稳定性结论；主体/来源标签仍仅用于 train-only challenge 定义，不进入 production 推理输入。
- 验证：`py -m pytest -q tests/test_summarize_article_triplet_multiseed.py` 为 `4 passed`；单文件 `compileall` 与 `git diff --check` 通过。
- 决策：不伪造 seed=`73` challenge 结果，也不因远端暂时不可达改用 external 调参。远端恢复后第一优先读取 seed=`17/73` challenge 产物并直接跑多 seed 汇总；若关键难片不能跨 seed 稳定改善，则冻结 Triplet 微调路线，转 reviewed Gold / temporal drift。
- 当前严格 production-like external 最优仍为 Macro-F1=`0.772131`，本轮 external 未再次读取。

## 2026-08-28 冲奖迭代：train-only 严格时间留出验证确认 temporal/domain drift

- 前置检查：本地未发现 EventLens Python 训练任务；通过既定 `eventlens-gpu` SSH alias 访问远端仍返回 `connect.cqa1.seetacloud.com:27487 Connection refused`，因此无法可信读取 seed=`73` challenge、GPU 或远端日志，按硬门禁未重复启动任何 GPU 训练。
- 新增 `tools/benchmark_company_temporal_holdout.py`，仅使用 tagged train 内部数据做 chronology-first 评估；production-like 文本仅为 `title + source + content[:2400]`，不使用 labeled-only `entity/trading_code/industry` 真值字段，也不读取 external。
- 时间切分门禁：先按发布时间确定 cutoff，再以 duplication group 为单位分配；若任一同源 group 跨越 cutoff，则整组直接丢弃，确保 `train_max_publish_time < holdout_min_publish_time` 且 duplication group overlap=`0`。专项测试 `2 passed`，覆盖同源组不跨侧与缺失 publish_time fail-closed。
- 严格 20% 时间留出结果：cutoff=`2026-03-18T10:00:00`，train=`1220`、holdout=`305`、boundary dropped=`0`；train groups=`1194`、holdout groups=`304`、group overlap=`0`；holdout 无 train 未见事件标签。固定 no-subject SVC Accuracy=`0.806557`、Macro-F1=`0.741846`。
- 对照判断：随机 duplication-safe OOF 基线约 `0.767373`，而严格最近 20% chronology holdout 下降约 `-0.025527`，支持此前 Triplet “train-side 有信号但 external 回退”的主要矛盾是 temporal/domain drift，而不是继续扩大 lr/epoch/margin/fusion 搜索空间。
- 曾先执行较宽松的 group-order 15%/20%/25% 诊断，发现按 group 最早时间排序存在理论上的边界未来信息风险；该结果不作为正式证据。工具已收紧为 cutoff crossing group drop 的 strict protocol，正式结论仅采用上述严格 20% 结果。
- 下一步：远端恢复后先读取 seed=`17/73` challenge 并做三 seed fail-closed 汇总；模型路线优先研究 train-only temporal robustness（例如只基于过去窗口构造 hard negatives / recency-balanced sampling 的低自由度单变量验证）或真实 reviewed Gold tranche，而不是再次读取 external 调参。
- 当前严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`；本轮 `external_touched=false`。

## 2026-08-29 冲奖迭代：temporal holdout 类别级/混淆对漂移诊断

- 前置检查：本地无 EventLens Python/pytest 实验进程；远端 `eventlens-gpu` 仍返回 `Connection refused`，因此没有重复启动 seed=`73` challenge 或其他 GPU 任务，也没有假设远端 GPU 空闲。
- 在严格 train-only 20% chronology + duplication-safe holdout 上补充类别级诊断，评估协议与上一轮保持不变：train=`1220`、holdout=`305`、group overlap=`0`、external_touched=`false`、Macro-F1=`0.741846`。
- 最弱且 support>=3 的类别包括：`新车发布定价` F1=`0.500000`、`产能布局调整` F1=`0.571429`、`关键技术突破` F1=`0.600000`、`海外产能进展` F1=`0.600000`、`合资品牌战略` F1=`0.666667`、`技术研发进展` F1=`0.684211`。
- 最大结构性混淆为 `产能布局调整 -> 关键技术突破` 共 `9` 条；其次有 `产品技术创新 -> 合资品牌战略` `4` 条、`产品技术创新 -> 技术研发进展` `3` 条、`关键技术突破 -> 新业务拓展` `3` 条。说明 temporal drift 不是均匀退化，而是集中在语义边界相邻的事件族。
- 简单 recency 重采样仍为负增益：Macro-F1=`0.733480`，相对 baseline `-0.008366`；类别诊断显示 `新车发布定价` 进一步由 F1=`0.500000` 降至 `0.285714`，`技术研发进展` 从 `0.684211` 降至 `0.571429`，因此冻结“单纯提高近期样本权重”路线。
- 决策：下一高价值算法实验应优先围绕上述 temporal confusion family 做 train-only、低自由度的过去窗口 hard-negative / reviewed Gold tranche 验证；不得将这些事件对硬编码进 production，也不得用 external 反推阈值或权重。
- 工程：`tools/benchmark_company_temporal_holdout.py` 新增 fail-closed 类别级 F1 与 top confusion pair 输出；专项测试 `4 passed`。产物：`reports/company_temporal_holdout_20pct_diagnostics.json`。
- 当前严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`。

## 2026-08-29 冲奖迭代：temporal drift 下人工 Gold 预算上限复验

- 前置检查：本地无既有 EventLens benchmark/triplet 训练任务；远端 `eventlens-gpu` 继续返回 `Connection refused`，无法可信核验远端 GPU/seed=73 challenge，因此未重复启动任何 GPU 训练。
- 目标：验证此前 15%/20% 主动复核预算在严格 chronology + duplication-safe temporal drift 场景下是否仍有足够价值，而不是继续做模型参数细网格。
- 协议：沿用严格 20% temporal holdout，train=`1220`、holdout=`305`、group overlap=`0`、external_touched=`false`；production-like 输入仍只含 title/source/content，不使用 labeled-only entity/trading_code/industry 真值字段。
- 新增选样：`predicted_class_balanced_low_margin`，只使用模型预测类别与 decision margin 轮转抽样，选样过程不读取 Gold；Gold 仅在 oracle 覆盖后的最终指标计算中使用。
- baseline temporal Macro-F1=`0.741846`。15%=`46` 条复核预算时，选中错误率=`0.391304`，oracle Macro-F1=`0.807475`，增益=`+0.065628`；20%=`61` 条时，选中错误率=`0.360656`，oracle Macro-F1=`0.846204`，增益=`+0.104358`。
- 判断：在更困难的 temporal drift 场景下，20% 人工复核预算仍可把理论上限推到 `0.846204`，非常接近 0.85；这与此前随机 OOF 上 15%/20% Gold oracle=`0.861547/0.887480` 的方向一致，进一步支持“受治理 reviewed Gold + 主动学习”是当前最高价值主路线，而不是继续 SVC/Triplet 超参搜索。
- 对照：recency 重采样模型本身 Macro-F1=`0.733480`，但 20% 同类 review oracle=`0.855926`；说明提升主要来自高价值人工标签，而不是近期样本简单加权。
- 工程：`tools/benchmark_company_temporal_holdout.py` 新增 temporal review oracle frontier；专项测试 `5 passed`，并完成 `compileall`、`git diff --check`。新产物 `reports/company_temporal_holdout_20pct_review_oracle.json`。
- 下一步：若无新增人工 reviewed Gold，不应把 oracle 当真实模型成绩；应优先等待/导入现有 229 条 review packet 的首 tranche，再做 train-only group-safe OOF + temporal/challenge-slice 重训验证。远端 SSH 恢复后仍优先读取 seed=17/73 challenge 产物做 fail-closed 多 seed 汇总。
- 邮件：当前自动运行上下文禁用邮件发送能力，因此本轮未发送 Gmail；项目迭代与留痕不受影响。

## 2026-08-29 冲奖迭代：rolling reviewed-Gold 三段时序实证

- 前置检查：本地无既有 EventLens 实验进程；远端 `eventlens-gpu` 仍返回 `Connection refused`，因此无法可信核验 GPU/seed=73 challenge，按门禁未重复启动任何远端或 GPU 训练。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external。
- 关键缺口：此前 temporal review oracle 只证明“人工直接改当前窗口”有上限价值，不能证明 reviewed Gold 会提升**后续未来窗口**。因此新增严格三段 chronology + duplication-safe 协议：历史训练窗口 -> 中间 review 窗口按预测类平衡低 margin 非 Gold 选样 -> 模拟人工审批标签回流 -> 更未来窗口重训评估。
- 协议结果：history=`1067`、review window=`229`、future=`229`、boundary dropped=`0`、三段 duplication group overlap=`0`、`external_touched=false`；review cutoff=`2026-03-12T01:29:18`，future cutoff=`2026-03-23T18:54:00`。
- 不回流 reviewed Gold 时，更未来窗口 baseline Macro-F1=`0.737818`。review-window 15% tranche=`34` 条时，选中错误率=`0.500000`，future Macro-F1=`0.734069`，增益=`-0.003749`；20% tranche=`46` 条时，选中错误率=`0.456522`，future Macro-F1=`0.777870`，增益=`+0.040052`。
- 判断：受治理 reviewed Gold 对真实未来泛化存在明显正向证据，但不是线性“少量标签即提升”；34 条覆盖不足会轻微退化，而 46 条达到更有效的类别/边界覆盖后出现约 `+0.04` 的 future Macro-F1 增益。该 `0.777870` 是 train-only rolling temporal 内部验证成绩，不是 external production-like 新纪录，不能替代冻结的 external=`0.772131`。
- 决策：主动学习/Gold 路线从“oracle 上限高”升级为“对后续未来窗口已有实证收益”；下一步优先把真实 approved Gold 按 tranche 导入并做 rolling temporal + duplication-safe OOF/challenge 验证。仍禁止把 review-window Gold 用于选样，禁止根据 external 调预算/阈值，也不恢复已冻结的 SVC/Triplet 参数细网格。
- 工程：`tools/benchmark_company_temporal_holdout.py` 增加 `temporal_three_way_group_split` 与 `rolling_review_tranche_experiment`，专项测试 `6 passed`。产物：`reports/company_temporal_holdout_20pct_rolling_review.json`。
- 邮件：当前自动运行上下文禁用邮件发送能力，因此无法执行 Gmail profile + send；本轮仅在项目留痕和任务通知中记录结果，未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：rolling reviewed-Gold 未来增益 bootstrap 稳定性门禁

- 前置检查：本地未发现既有 EventLens 同类实验进程；远端 `eventlens-gpu` 仍返回 `connect.cqa1.seetacloud.com:27487 Connection refused`，因此无法可信核验远端 GPU、seed=`73` challenge 或日志，按门禁未重复启动任何 GPU/Triplet 任务。严格 production-like external 最优仍冻结为 Macro-F1=`0.772131`，本轮未读取 external 做选择或调参。
- reviewed Gold 状态：`artifacts/review_packet` 与 `artifacts/review_queue` 中未发现新的 approved/reviewed feedback 产物，因此没有把模拟标签冒充真实人工 Gold，也没有启动真实 Gold 回流训练。
- 工程推进：对同一 future window 的 baseline/retrained 预测增加 deterministic paired bootstrap（`n=2000`, seed=`20260829`），直接量化 Macro-F1 gain 的 95% CI 与正增益概率；该步骤不改变选样策略、模型超参或预算，只用于候选生产稳定性门禁。
- 三段协议保持不变：history=`1067`、review=`229`、future=`229`、boundary dropped=`0`、group overlap=`0`、`external_touched=false`，baseline future Macro-F1=`0.737818`。
- 15% tranche=`34` 条：future Macro-F1=`0.734069`，point gain=`-0.003749`；paired bootstrap 95% CI=`[-0.033606, +0.051757]`，正增益概率=`0.5830`。该预算没有稳定正向证据。
- 20% tranche=`46` 条：future Macro-F1=`0.777870`，point gain=`+0.040052`；paired bootstrap 95% CI=`[-0.030369, +0.095971]`，正增益概率=`0.8235`。方向仍明显优于 15%，但 CI 跨 `0`，尚不能按候选生产门禁宣称“稳定提升”。
- 决策：保留 reviewed Gold / 主动学习为最高价值主路线，但将此前 `+0.040052` 从“明显未来增益”收紧为“有前景、但统计稳定性未过门禁”。下一验证应优先扩展到多个严格 rolling temporal windows / 多 cutoff backtest，或导入真实 approved Gold tranche 后重复同一评估；不通过预算细网格、external 调参或事件对硬编码去缩窄 CI。
- 验证：`tests/test_benchmark_company_temporal_holdout.py` 新增 bootstrap 正增益与 fail-closed 对齐测试，共 `8 passed`；相关 `compileall` / `git diff --check` 通过。产物：`reports/company_temporal_holdout_20pct_rolling_review_bootstrap.json`。
- 邮件：当前运行环境明确禁用邮件发送能力，因此无法执行 Gmail profile + send；本轮有新实验结果但只能在项目留痕和任务通知中记录，未向任何未知地址发送邮件。

## 2026-08-29 冲奖迭代：reviewed-Gold 固定三窗口 rolling backtest

- 前置检查：本地未发现 Python/EventLens 同类实验进程；远端 `eventlens-gpu` 仍返回 `Connection refused`，无法可信核验远端 GPU、seed=`73` challenge 与日志，因此未重复启动任何 GPU/Triplet 任务。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external 做模型选择或调参。
- 动机：上一轮最后一个 rolling window 的 20% reviewed-Gold tranche future gain=`+0.040052`，但 paired bootstrap CI 跨 `0`。为排除单窗口偶然性，本轮固定 `end_fraction=[0.70, 0.85, 1.00]` 做三窗口 train-only chronology backtest；每窗 review/future 各占全量 `15%`，duplication group 严格隔离，结束边界后的样本不参与更早窗口，选样仍只用 predicted-class-balanced low-margin，不读取 Gold。
- 0.70 窗口：history=`609`、review=`229`、future=`229`、group overlap=`0`；baseline future Macro-F1=`0.563960`。15% tranche gain=`+0.033020`，bootstrap positive probability=`0.8600`；20% tranche gain=`+0.055675`，positive probability=`0.9500`，但 95% CI=`[-0.007030,+0.083428]` 仍跨 `0`。
- 0.85 窗口：history=`838`、review=`229`、future=`229`、group overlap=`0`；baseline future Macro-F1=`0.585091`。15% tranche gain=`-0.039312`，positive probability=`0.0480`；20% tranche gain=`-0.030395`，positive probability=`0.1065`。该窗口给出明确反例，说明单一 low-margin/class-balanced acquisition 会在部分时间段伤害未来泛化。
- 1.00 窗口复现上一轮：baseline future Macro-F1=`0.737818`；15% gain=`-0.003749`，20% gain=`+0.040052`。
- 跨窗汇总：15% tranche mean gain=`-0.003347`、positive windows=`1/3`；20% tranche mean gain=`+0.021777`、positive windows=`2/3`、min gain=`-0.030395`，`all_windows_positive=false`。因此 reviewed Gold / 主动学习仍是最高价值方向，但“predicted-class-balanced low-margin + 固定 20% tranche”不能作为稳定生产策略，也不能把单窗口 `+0.040052` 外推成稳定收益。
- 决策：冻结对 low-margin acquisition 的预算细网格；下一步若无真实 approved Gold，应优先验证更符合既有 OOF 证据的 train-only `disagreement + class-balance` rolling acquisition，并要求跨窗口 harmed=0 / all-windows-positive 或更严格的 CI 门禁。不得使用 external 选择 acquisition、预算或阈值。
- 工程：新增 `temporal_three_way_group_split_at_end`、固定三窗口 `rolling_review_backtest` 与对应 fail-closed 测试；专项测试 `9 passed`，`compileall` 与 `git diff --check` 通过。正式产物：`reports/company_temporal_holdout_20pct_rolling_backtest.json`，`external_touched=false`。
- 邮件：当前自动运行环境禁用邮件发送能力，无法执行 Gmail profile + send；本轮有新实验结论，已在项目留痕与任务通知中记录，未向任何未知地址发送邮件。

## 2026-08-29 冲奖迭代：temporal snapshot disagreement acquisition 三窗口复验

- 前置检查：本地未发现既有 EventLens benchmark/triplet 同类实验；远端 `eventlens-gpu` 继续返回 `connect.cqa1.seetacloud.com:27487 Connection refused`，无法可信读取 GPU、seed=`73` challenge 或远端 reports，因此按硬门禁未重复启动任何远端/GPU 训练。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external。
- 资源约束：本地未发现可复用的 `company_event_train` BGE embedding 资产，Ollama `127.0.0.1:11434` 也不可达；本机仅 RTX 3050 Laptop 4GB，故没有伪造或重新计算 SVC-BGE disagreement。为验证“disagreement 是否本身能改善 rolling reviewed-Gold acquisition”，仅新增一个低自由度 train-only 替代：全历史 SVC 与最近半历史 SVC 在 review 窗口的 prediction disagreement，按主模型预测类轮转并以主模型 margin 排序；固定 recent snapshot=`50%`、review budget=`20%`，不做网格。
- 0.70 窗口：history=`609`、recent snapshot=`305`、review=`229`、future=`229`；disagreement=`41`，因此实际只复核 `41` 条；选中错误率=`0.634146`。future Macro-F1=`0.563960 -> 0.581646`，gain=`+0.017687`；paired bootstrap 95% CI=`[-0.014935,+0.056395]`，positive probability=`0.8430`。
- 0.85 窗口：history=`838`、recent snapshot=`419`、review=`229`、future=`229`；disagreement=`47`，实际复核 `46` 条；选中错误率=`0.478261`。future Macro-F1=`0.585091 -> 0.583777`，gain=`-0.001314`；95% CI=`[-0.039448,+0.038896]`，positive probability=`0.4295`。
- 1.00 窗口：history=`1067`、recent snapshot=`534`、review=`229`、future=`229`；disagreement=`42`，实际复核 `42` 条；选中错误率=`0.452381`。future Macro-F1=`0.737818 -> 0.717629`，gain=`-0.020189`；95% CI=`[-0.050678,+0.028425]`，positive probability=`0.2730`。
- 跨窗汇总：mean future Macro-F1 gain=`-0.001272`，min gain=`-0.020189`，positive windows=`1/3`，`all_windows_positive=false`。因此“时间快照 disagreement + class-balance”明确未过稳定门禁，冻结该替代 acquisition，不继续扫 recent fraction、预算或 margin。
- 结论边界：该失败不能否定此前 OOF 上真实 `SVC-BGE disagreement + class balance` 的高 oracle 价值；它只说明**无 BGE 的 temporal-snapshot disagreement 不能作为等价替代**。下一高价值验证仍是远端/BGE 资产恢复后，把真实 SVC-BGE disagreement acquisition 放入同一 fixed-three-window rolling protocol；若仍不能 harmed=0 / all-windows-positive，则主动学习策略需要进一步依赖真实 approved Gold 或更稳健的 acquisition，而不是继续模型/预算网格。
- 工程：`tools/benchmark_company_temporal_holdout.py` 新增 `_disagreement_class_balanced_indices`、`rolling_snapshot_disagreement_experiment/backtest`；专项测试扩展到 `11 passed`。正式产物：`reports/company_temporal_snapshot_disagreement_backtest.json`，`external_touched=false`。
- 邮件：当前运行环境不提供邮件发送能力，本轮有新实验结论但无法执行 Gmail profile + send；项目迭代继续，且未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：history-only 3-fold ensemble disagreement 三窗口复验

- 前置检查：本地未发现既有 EventLens benchmark/triplet 同类实验；远端 `eventlens-gpu` 继续返回 `connect.cqa1.seetacloud.com:27487 Connection refused`，因此无法可信读取 GPU、seed=`73` challenge、日志或最新远端 reports，按硬门禁未重复启动任何远端/GPU 训练。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external。
- 设计理由：上一轮 temporal-snapshot disagreement 失败可能来自“最近半历史模型”本身存在时间偏置。为区分 acquisition 思路失败还是 secondary model 偏置，本轮固定一个更稳健、仍完全 train-only 的单变量方案：仅在每个 rolling history 窗口内部按 duplication group 的稳定 SHA-256 哈希构造 3 个非重叠 fold；训练 3 个 each-heldout 子模型，对 review 窗口多数投票，再与 full-history primary 的 prediction disagreement 做 predicted-class balance + primary margin 排序。fold 数固定为 `3`、review budget 固定 `20%`，不做网格；review Gold 只在模拟审批回流和离线评估阶段使用。
- 0.70 窗口：history=`609`、fold counts=`212/193/204`、review=`229`、future=`229`；primary-vs-ensemble disagreement=`17`，实际选中 `17` 条，错误率=`0.823529`。future Macro-F1=`0.563960 -> 0.570905`，gain=`+0.006945`；paired bootstrap 95% CI=`[-0.007471,+0.018994]`，positive probability=`0.6485`。
- 0.85 窗口：history=`838`、fold counts=`303/254/281`；disagreement=`9`，实际选中 `9` 条，错误率=`0.555556`。future Macro-F1=`0.585091 -> 0.610052`，gain=`+0.024961`；95% CI=`[-0.004567,+0.062033]`，positive probability=`0.9385`。
- 1.00 窗口：history=`1067`、fold counts=`382/329/356`；disagreement=`16`，实际选中 `16` 条，错误率=`0.625000`。future Macro-F1=`0.737818 -> 0.705977`，gain=`-0.031841`；95% CI=`[-0.052005,-0.001596]`，positive probability=`0.0080`。该 CI 完全低于 `0`，是明确 harmed 反例，而非随机波动无法判断。
- 跨窗汇总：mean future Macro-F1 gain=`+0.000022`，min gain=`-0.031841`，positive windows=`2/3`，`all_windows_positive=false`。因此**history-only 3-fold ensemble disagreement + class-balance 也未过生产稳定门禁**；尤其最后窗口存在 bootstrap 显著负收益，正式冻结该替代 acquisition，不继续扫 fold 数、budget、margin 或哈希种子。
- 结论边界：该实验仍不能否定此前 OOF 上真实 `SVC-BGE disagreement + class balance` 的高 oracle 价值，因为 secondary semantic view 不同。下一最高价值算法断点仍是：BGE 资产或远端恢复后，把真实 SVC-BGE disagreement 放入同一 fixed-three-window chronology protocol；若其仍不能 harmed=0 / all-windows-positive，则 acquisition 主路线应转向真实 approved Gold 与持续数据覆盖，而不是继续构造 SVC 内部 disagreement 代理。
- 工程：新增 `_history_group_fold_indices`、`_majority_vote`、`rolling_history_ensemble_disagreement_experiment/backtest`；duplication group 保持同 fold、预测对齐 fail-closed，专项测试扩展到 `13 passed`。正式产物：`reports/company_temporal_holdout_20pct_history_ensemble.json`，其中 `rolling_history_ensemble_disagreement_backtest.external_touched=false`。
- 邮件：当前自动运行环境禁用邮件发送能力，无法执行 Gmail profile + send；本轮有新实验结论，已在项目留痕与任务通知中记录，未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：history OOF class-risk acquisition 五窗口复验

- 前置检查：本地未发现既有 EventLens 同类实验进程；远端 `eventlens-gpu` 仍返回 `Connection refused`，无法可信读取 GPU、seed=`73` challenge、远端日志或最新 reports，因此未重复启动 GPU/Triplet 任务。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external。
- 设计理由：简单 low-margin 等额 class-balance 在五窗口仅 `2/5` 为正，SVC 内部 disagreement 代理也不稳定。本轮固定一个纯 train-only acquisition：每个 rolling history 窗口内部按 duplication group 做 3-fold OOF，以 **OOF 预测类错误率**估计 class risk；固定 review budget=`20%`，预算按 `class_error_rate × review predicted-class availability` 分配，类内仍按 primary margin 从低到高选样。review Gold 只用于模拟人工审批回流与离线评估，不参与选样；不扫 fold/budget/权重。
- 五窗口 future gain（end=`0.65/0.75/0.85/0.95/1.00`）：`-0.008945/+0.026339/-0.007782/-0.008814/+0.025098`。mean=`+0.005179`、median=`-0.007782`、positive windows=`2/5`、`all_windows_positive=false`；虽然均值比简单 low-margin 的 `-0.000396` 更好，但仍存在 3 个 harmed 窗口，未过生产稳定门禁。
- 选中错误率分别为 `0.543478/0.586957/0.565217/0.543478/0.673913`，说明 history OOF class-risk 确实能富集错误样本；但“更会挑错”并不自动等价于“回流后未来 Macro-F1 稳定提升”，仍受 temporal/domain drift 与类别边界迁移影响。
- paired-bootstrap 95% CI 仍全部跨 `0`；例如 end=`0.75` gain=`+0.026339`、positive probability=`0.9145`，但 CI=`[-0.010208,+0.056890]`；end=`1.00` gain=`+0.025098`，CI=`[-0.030279,+0.056730]`。因此不能把正窗口宣传为稳定增益。
- 决策：**冻结 history-OOF class-risk + low-margin 这一 SVC-only acquisition 变体，不继续搜索风险平滑、预算或 OOF fold 数。** 该实验进一步说明当前瓶颈不在“再设计一个 SVC acquisition heuristic”，下一最高价值仍是远端/BGE 资产恢复后，用真正异构的 `SVC-BGE disagreement + class balance` 进入同一 rolling protocol；若仍不能 harmed=0 / all-windows-positive，则主线应收敛到真实 approved Gold 的持续补充与受治理数据飞轮。
- 工程：`tools/benchmark_company_temporal_holdout.py` 新增 history OOF predicted-class error-rate acquisition 与固定五窗口 backtest；专项测试 `16 passed`。正式产物：`reports/company_temporal_history_oof_risk_20pct_five_window.json`，`external_touched=false`。
- 邮件：当前运行环境禁用邮件发送能力，无法执行 Gmail profile + send；本轮有新实验结论，仅在项目留痕与任务通知中记录，未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：100% approved recent Gold refresh 五窗口上限诊断

- 前置检查：本地未发现既有 EventLens benchmark/triplet 同类实验进程；本机 RTX 3050 Laptop 4GB 仅有桌面图形占用，没有 EventLens 计算任务。远端 `eventlens-gpu` 仍返回 `connect.cqa1.seetacloud.com:27487 Connection refused`，因此无法可信读取远端 GPU、seed=`73` challenge、日志或最新 reports，按硬门禁未重复启动任何远端/GPU 训练。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external 做模型选择或调参。
- 设计理由：前序 low-margin / temporal snapshot disagreement / history ensemble disagreement / history OOF class-risk 等低预算 acquisition 均存在 harmed rolling window。为区分“选样策略不稳”与“近期 Gold 本身无效”，本轮不再设计新 heuristic，而是做固定上限诊断：每个 chronology 三段窗口中，假设 **review window 100% 经人工审批成为 Gold**，全部回流 history 后评估下一 future window。review/future 各固定占全量 `15%`，结束分位预声明为 `0.65/0.75/0.85/0.95/1.00`，duplication group overlap=`0`，`external_touched=false`；不做预算/阈值/模型参数搜索。
- 五窗口 baseline -> full-refresh future Macro-F1：`0.617653->0.618310`（gain=`+0.000657`）、`0.581998->0.631766`（`+0.049768`）、`0.585091->0.604852`（`+0.019760`）、`0.692952->0.722893`（`+0.029940`）、`0.737818->0.822714`（`+0.084896`）。每个 review window 均约 `229` 条 approved Gold，全部五个窗口正增益，mean gain=`+0.037004`、median=`+0.029940`、min=`+0.000657`、max=`+0.084896`、`all_windows_positive=true`。
- 稳定性边界：五窗口 point gain 首次全部为正，说明“持续补充近期真实 Gold”本身有比低预算 SVC heuristic 更一致的跨期方向；但 paired-bootstrap 95% CI 仍全部跨 `0`。其中 end=`0.75` positive probability=`0.9625`、gain=`+0.049768`、CI=`[-0.002661,+0.078618]`；end=`0.95` positive probability=`0.9535`、gain=`+0.029940`、CI=`[-0.005136,+0.069705]`；最后窗口 gain=`+0.084896` 但 CI=`[-0.018660,+0.141881]`。因此该实验只作为 **approved Gold refresh 上限/方向性证据**，绝不包装为新的 production-like 成绩。
- 决策：主动学习/数据飞轮主线得到更强 train-only 支撑。下一最高价值不是继续造 SVC acquisition heuristic，而是远端/BGE 资产恢复后，将真实异构 `SVC-BGE disagreement + class balance` 放入同一 rolling protocol，目标是在明显低于 100% review 成本下逼近 full-review refresh，并继续要求 harmed=0 / all-windows-positive 或更严格 CI；若仍不稳，则真实 approved Gold 持续补充本身应作为企业可交付的主要性能演进机制。
- 工程：新增 `rolling_full_review_refresh_experiment/backtest`，协议固定五窗口与 100% review Gold；专项测试扩展到 `17 passed`。正式产物：`reports/company_temporal_full_review_gold_refresh_five_window.json`，`external_touched=false`。
- 邮件：当前自动运行环境明确禁用邮件发送能力，无法执行 Gmail profile + send；本轮有新实验结果但只能在项目留痕和任务通知中记录，未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：recent-75% history group-safe 滑窗五窗口复验

- 前置检查：本地未发现 EventLens 模型实验进程；存在上一轮遗留 Docker build，但与本轮 benchmark 不同类，未触碰。远端 `eventlens-gpu` 仍返回 `Connection refused`，无法可信读取 GPU、seed=`73` challenge、远端日志与 reports，因此未重复启动任何 GPU/Triplet 任务。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external。
- 设计理由：前序 reviewed-Gold temporal coverage 显示近期数据有价值，但尚需区分“需要新增近期 Gold”还是“只保留更近历史即可适应 drift”。因此预声明单一低自由度方案：每个 rolling history 仅保留最近 `75%` 样本，cutoff 跨越的 duplication group 直接丢弃；review/future 完全不进入训练或选择。比例固定，不扫 50/60/80/90%。
- 五窗口 future gain（end=`0.65/0.75/0.85/0.95/1.00`）：`-0.051993/-0.035408/-0.060018/-0.024203/-0.132011`；mean=`-0.060727`、median=`-0.051993`、positive windows=`0/5`、`all_windows_positive=false`。该路线在所有窗口均伤害未来 Macro-F1。
- 最后窗口 baseline=`0.737818`，recent-75% history=`0.605808`，gain=`-0.132011`；paired-bootstrap 95% CI=`[-0.194488,-0.061597]`，positive probability=`0.0`，为统计上明确的 harmed 反例。end=`0.85` 也有 gain=`-0.060018`、positive probability=`0.038`。
- 结论：**冻结简单 recent-history truncation / sliding-window SVC，不继续搜索 keep_fraction。** temporal drift 不能靠丢弃旧 Gold 解决；旧样本仍提供重要类别/事件边界覆盖。结合 100% recent approved-Gold refresh 的 `5/5` 正向证据，下一主线仍应是“保留历史 + 持续补充近期 approved Gold”，以及待远端恢复后用真实 `SVC-BGE disagreement + class balance` 降低人工审核成本，而不是缩短历史窗口。
- 工程：新增 `recent_group_safe_history_indices`、`rolling_recent_history_experiment/backtest` 与 duplication-boundary fail-closed 测试；专项测试 `22 passed`。正式产物：`reports/company_temporal_recent75_history_five_window.json`，`external_touched=false`。
- 邮件：当前自动运行环境明确禁用邮件发送能力，无法执行 Gmail profile + send；本轮新结论仅在项目留痕和任务通知中记录，未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：保留全历史 + recent-25% 重采样五窗口复验

- 前置检查：本地最新产物停在 `company_temporal_recent75_history_five_window.json`，未发现既有 Python/EventLens 模型实验进程。远端 SSH 状态检查在当前执行环境被安全层拦截，因此**本轮未能可信核验远端 GPU/进程/log/reports**，也未把历史 `Connection refused` 当成本轮实时结果；按门禁没有启动任何远端/GPU/Triplet 任务。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external 做选择或调参。
- 设计理由：recent-75% truncation 五窗口 `0/5` 正向，说明“删旧数据适应 drift”明显错误。为保留旧历史事件边界覆盖，同时测试低自由度近期适配，预声明单一方案：每个 rolling history **全部保留**，并仅将最新 `25%` history 额外重复一次；review/future 完全不参与选择，比例和重复次数均固定，不扫 10/20/30/50% 或多次重复。
- 五窗口 future gain（end=`0.65/0.75/0.85/0.95/1.00`）：`+0.037104/+0.013124/-0.008217/+0.006804/+0.012816`；mean=`+0.012326`、median=`+0.012816`、positive windows=`4/5`、`all_windows_positive=false`。相比 recent-75% truncation 的 mean=`-0.060727`、`0/5`，轻度 recent weighting 明显更合理，但仍存在 harmed 窗口。
- 稳定性：end=`0.65` baseline=`0.617653` -> candidate=`0.654757`，gain=`+0.037104`，paired-bootstrap 95% CI=`[+0.007733,+0.062549]`、positive probability=`0.9955`，为明确正向窗口；end=`0.85` gain=`-0.008217`，CI=`[-0.028367,+0.007734]`、positive probability=`0.1635`，因此整体仍不能过 harmed=0 / all-windows-positive 门禁。
- 决策：保留该结果作为“轻度近期权重优于删旧历史”的证据，但**冻结 repeat ratio/次数搜索**；单独 recency weighting 不升级 production，也不触碰 external。为验证其是否能在低人工成本 Gold flywheel 上提供互补增益，下一步只做一个预声明组合：固定 50% temporal coverage reviewed Gold + 同一 recent-25% repeat-once，不改变 acquisition 或预算。
- 工程：新增 `rolling_recent_repeat_experiment/backtest`；专项测试通过。正式产物：`reports/company_temporal_recent25_repeat_once_five_window.json`，`external_touched=false`。

## 2026-08-29 冲奖迭代：50% temporal Gold coverage + recent-25% repeat 固定组合

- 设计：在上一轮已冻结的 `50%` predicted-class-balanced temporal coverage reviewed-Gold 选样上，仅叠加刚验证的“保留全 history + 最新 `25%` history 重复一次”。review 选样仍由 baseline full-history 模型的预测类 + 类内时间覆盖决定，不读取 review/future Gold；同时计算 baseline、coverage-only 与 combined 三者，避免把组合收益误归因于 Gold 本身。
- 五窗口 combined future gain：`-0.017182/+0.059041/+0.021016/+0.017313/+0.057113`；mean=`+0.027460`、median=`+0.021016`、positive windows=`4/5`、`all_windows_positive=false`。最早窗口虽然相较 coverage-only 的 `-0.031451` 改善到 `-0.017182`，但仍 harmed；end=`0.75` 与 `1.00` 的 combined gain 95% CI 分别为 `[+0.005210,+0.099087]`、`[+0.011362,+0.105537]`，局部有稳定正向证据。
- 关键互补性诊断：combined 相对 coverage-only 的五窗口增量=`+0.014269/+0.004043/-0.014837/+0.006500/-0.016169`，平均 **`-0.001239`**，仅 `3/5` 窗口为正。也就是说 recent weighting 虽能单独改善部分 drift，但与 50% reviewed-Gold coverage **没有稳定可加性**，甚至在 end=`0.85/1.00` 抵消部分 Gold refresh 收益。
- 决策：**冻结“50% Gold coverage + recent repeat”组合，不继续扫 repeat ratio、次数、review budget 或组合权重。** 当前企业数据飞轮证据仍以 100% recent approved Gold refresh 的 `5/5` 正向最稳；50%/75% coverage 各 `4/5`，而 recency weighting 不能把低成本方案推到 harmed=0。下一最高价值断点保持不变：远端/BGE 资产恢复后，用真正异构的 `SVC-BGE disagreement + class balance` 进入同一冻结五窗口 rolling protocol；若仍失败，则优先真实 approved Gold 持续覆盖，而不是更多 SVC/recency heuristic。
- 工程与门禁：新增固定组合 experiment/backtest；专项测试 `24 passed`。正式产物：`reports/company_temporal_coverage50_recent25_repeat_five_window.json`，`external_touched=false`。本轮没有新增 production-like external 分数，严格 external 最优仍为 `0.772131`。
- 邮件：当前自动运行上下文禁用邮件发送能力，无法执行 Gmail profile + send；本轮有新实验结果，已在项目留痕和任务通知中记录，未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：duplication-group 单代表 Gold 传播五窗口审计

- 前置检查：本地 `tasklist` 未发现 Python/SSH EventLens 实验进程；远端 `eventlens-gpu` 本轮实时返回 `connect.cqa1.seetacloud.com:27487 Connection refused`，因此无法可信读取 GPU、seed=`73` challenge、远端日志或 reports，按硬门禁未重复启动任何 GPU/Triplet 任务。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮 `external_touched=false`。
- 设计理由：100% recent approved Gold refresh 是当前唯一 `5/5` 正向路线，但人工成本高。为验证 duplication 数据是否可用于降低审批成本，本轮固定测试“每个 review-window duplication group 只人工审批最早 1 篇代表文，然后盲传播该 Gold 标签给同组其余文章”。代表选择只依赖 duplication group、发布时间与 article_id；同组其余 Gold 仅用于事后审计传播错误，不参与选择或传播决策。
- 五窗口结果（end=`0.65/0.75/0.85/0.95/1.00`）：future gain=`-0.000534/+0.039046/+0.022693/+0.031994/+0.084896`，mean=`+0.035619`、median=`+0.031994`、positive windows=`4/5`、`all_windows_positive=false`。首窗出现 harmed，未达到 100% full-review 的 `5/5` 稳定性；所有 bootstrap CI 仍跨 `0`。
- 人工成本并没有实质下降：每个 review window 均为 `229` 条，而单代表审批动作分别仍需 `221/223/225/225/227` 次；五窗口平均人工动作比例=`0.979039`，即仅节省约 `2.1%`。因此这条路线即使性能接近 full-review，也不具备足够 ROI。
- 更关键的安全审计：全 company train 共 `1525` 行、`1498` 个 duplication group，其中 `1474` 个为 singleton，singleton 行占比=`0.966557`；仅 `24` 个 multi-member group（51 行），但其中 `21/24=87.5%` 的组包含多个 event label。说明 `duplication_id` 适合 duplication-safe split/同源处理，却**不是安全的事件标签传播键**。五窗口传播 row-level precision 虽均约 `97.4%–99.1%`，主要是 singleton 主导的表象，不能据此开放自动传播。
- 决策：**正式否决 blind duplication-group Gold propagation**，不再搜索 representative 规则、组大小阈值或传播置信度；企业侧保持“duplication_id 只做泄漏控制/同源聚合，事件 Gold 必须逐条审批或由更可靠证据链确认”。下一最高价值算法断点不变：远端/BGE 恢复后，用真正异构的 `SVC-BGE disagreement + class balance` 进入冻结五窗口 protocol，验证能否在显著低于 100% 人工成本下达到 harmed=0。
- 工程：新增 group representative propagation backtest 与 fail-closed 测试；正式产物 `reports/company_temporal_dupgroup_representative_gold_five_window.json`、`reports/company_duplication_group_gold_propagation_audit.json`。专项测试 `26 passed`。
- 邮件：当前运行环境明确禁用邮件发送能力，无法执行 Gmail profile + send；本轮有新实验与明确否决结论，已在项目留痕与任务通知中记录，未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：三批人工审阅包累计 tranche 稳定性验证

- 前置检查：本地未发现 EventLens 同类模型实验；本机 RTX 3050 仅桌面进程占用。远端 `eventlens-gpu` 本轮 SSH 已完成 TCP/SSH 建链但随后被 `58.144.141.28:27487` 主动关闭，无法可信读取远端 GPU、日志、seed=73 challenge 或最新 reports；因此未重复启动任何远端/GPU/Triplet 任务。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮 `external_touched=false`。
- 设计理由：现有正式人工审阅包固定为 `76+76+77=229` 三批。相比继续扫 40/60/80% 预算，本轮只验证与真实运营流程一一对应的累计 tranche：第一批完成约 `33%`，前两批完成约 `67%`，三批全部完成即 `100%`。选样仍使用既定 predicted-class balance + 类内时间覆盖，不读取 review/future Gold；五个 chronology + duplication-safe rolling window 固定不变。
- 33% tranche：每窗审批 `76/229` 条；future gain=`-0.058934/+0.029612/+0.004347/-0.003833/-0.003739`，mean=`-0.006509`、median=`-0.003739`、positive windows=`2/5`、`all_windows_positive=false`。最早窗口 positive-gain probability=`0.0415`，说明仅完成第一批就重训上线存在明确 harmed 风险。
- 67% tranche：每窗审批 `153/229` 条；future gain=`-0.020203/+0.050693/+0.033522/+0.026549/+0.036152`，mean=`+0.025343`、median=`+0.033522`、positive windows=`4/5`、`all_windows_positive=false`。虽然明显优于第一批，但最早窗口仍 harmed，尚未达到生产候选门禁。
- 与已冻结成本点合并后形成运营前沿：33%=`2/5` 正向、50%=`4/5`、67%=`4/5`、75%=`4/5`、100%=`5/5`；只有 full-review 100% recent approved Gold refresh 满足 `all_windows_positive=true`。因此不得把“完成第一批/第二批”本身作为自动发布条件；部分 tranche 只能进入 shadow/training-candidate，需继续累计 Gold 并重新通过 rolling + bootstrap/challenge 门禁。
- 决策：不继续补 40/60/80/90% 等预算网格。下一最高价值仍是远端/BGE 恢复后，把真实 `SVC-BGE disagreement + class balance` 放进同一冻结五窗口 protocol，检验能否在显著低于全量人工审批的前提下做到 harmed=0；若不能，企业数据飞轮发布策略应明确采用“累计充分 reviewed Gold 后再发布”，而不是按批次到达即重训上线。
- 产物：`reports/company_temporal_coverage_review_33pct_five_window.json`、`reports/company_temporal_coverage_review_67pct_five_window.json`、`reports/company_temporal_review_tranche_operational_frontier.json`。
- 邮件：当前运行环境明确禁用邮件发送能力，无法执行 Gmail profile + send；本轮有新实验结果与明确发布结论，已在项目留痕和任务通知中记录，未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：Triplet class-balanced anchor sampling 预注册与远端阻塞

- 前置检查：本地没有 EventLens/Triplet 训练进程，本机 RTX 3050 仅桌面图形占用（约 `1423/4096 MiB`，GPU util 约 `14%`）。远端 `eventlens-gpu` 首次只读检查成功：仅有 TensorBoard/Jupyter 基础进程，无 EventLens 训练任务；最新报告为 `article_triplet_bge_oof_seed73_challenge_company.json`，说明 seed=`73` challenge 已完成。随后两次 SSH 均被 `58.144.141.28:27487` 主动关闭，因此无法继续可信读取 GPU 或启动新训练；按硬门禁没有重复/盲发 GPU 任务。
- 设计理由：固定 Triplet 的三 seed OOF gain 已达到 `+0.012949/+0.005299/+0.005620`，但 challenge 聚合中 `rare_event` 仅 `2/3` seed 正、`anti_subject_prior` 仅 `1/3` seed 正。当前训练每篇文章等权，导致高频事件类对梯度总量贡献更大。为优先降低 train-side 方差而不是重扫 lr/epoch/margin，本轮预注册一个**无连续超参**的单变量改动：按 anchor 类频次倒数使用 `WeightedRandomSampler`，总采样数仍等于原 triplet 数、epochs/batch/lr/margin/last-2/top3/fusion 全部冻结。
- 工程：`ArticleTripletTrainingConfig` 新增默认关闭的 `class_balanced_sampling`；新增 `inverse_frequency_anchor_weights()`，保证每类期望采样质量相同；`benchmark_article_triplet_bge_oof.py` 新增 `--class-balanced-sampling` 且写入报告 `fixed_config`，external gate 逻辑不变、默认仍不读取 external。专项测试 `7 passed`。
- 当前结论边界：**本轮尚无新的 Macro-F1 数值，不能把该改动宣称为提升。** 由于远端 SSH 在实现完成后持续主动断开，而本地 4GB GPU 低于既有 OOF 单折约 `5.3GB` 峰值，本轮停止在“代码+测试+协议预注册”而没有降级到不等价的本地 CPU/小模型实验。下一次远端稳定可用时，最高价值步骤是先跑固定 seed=`42` 的 class-balanced duplication-safe 3-fold OOF + challenge；只有整体 OOF 不退化且 `rare_event/anti_subject_prior` 稳定性改善，才允许扩到 seed=`17/73`，仍禁止触碰 external。
- 严格 production-like external 最优保持 Macro-F1=`0.772131`；本轮 `external_touched=false`。
- 邮件：当前自动运行环境禁用邮件发送能力，无法执行 Gmail profile + send；本轮阻塞与工程改动已在项目留痕和任务通知中记录，未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：远端 SSH 恢复但 GPU 设备未挂载

- 前置检查：本地未发现 EventLens/Triplet 训练进程，本机 RTX 3050 Laptop 仅桌面占用（约 `1587/4096 MiB`，GPU util 约 `19%`）。远端 `eventlens-gpu` 本轮可建立 SSH，未发现 `benchmark_article_triplet` 或其他 EventLens 训练进程；远端最新 report 仍为已完成的 `article_triplet_bge_oof_seed73_challenge_company.json`，因此不存在需要监控的同类训练任务。
- GPU 阻塞：远端 `nvidia-smi` 可执行但无设备输出，`/dev/nvidia*` 不存在；远端 PyTorch=`2.6.0+cu124`，`torch.cuda.is_available()=False`、`device_count=0`。这说明当前不是“GPU 空闲”，而是容器没有挂载 GPU 设备；因此没有启动预注册的 seed=`42` class-balanced Triplet OOF，避免误退化到 CPU 或盲发任务。
- 同步状态：尝试把本地已测试的 class-balanced Triplet 实现同步到 `/root/autodl-tmp/EventLens`，SSH 在复制过程中再次被远端主动关闭；复核发现仅 `tests/test_benchmark_article_triplet_bge_oof.py` 与本地一致，其余目标文件仍是远端旧版本，因此**不把远端视为已完成部署**。后续 GPU 恢复后应先完成这 4 个文件的原子同步并复核 hash，再启动 seed=`42`。
- 本地验证：重新执行 `py -m pytest -q tests/test_article_contrastive.py tests/test_benchmark_article_triplet_bge_oof.py`，结果 `7 passed`。class-balanced sampler 的本地实现与 external fail-closed 门禁保持可用。
- 决策：本轮没有新的 Macro-F1，严格 production-like external 最优继续冻结为 `0.772131`，`external_touched=false`。下一断点保持为固定 seed=`42` 的 class-balanced duplication-safe 3-fold OOF + challenge；只有整体 OOF 不退化且 `rare_event/anti_subject_prior` 改善，才扩 seed=`17/73`，仍禁止触碰 external。
- 邮件：当前自动运行环境禁用邮件发送能力，本轮无法执行 Gmail profile + send；阻塞与验证结果已写入项目留痕。

## 2026-08-29 冲奖迭代：真实 SVC-BGE disagreement 固定五窗口复验

- 前置检查：本地无 EventLens/Triplet 模型训练进程，仅有既有 submission Docker build；RTX 3050 为桌面图形占用。远端 `eventlens-gpu` 本轮 SSH 可连接，未发现 EventLens/Triplet 训练进程；`nvidia-smi` 仍无 GPU 设备输出，最新远端 reports 仍为已完成的 `article_triplet_bge_oof_seed73_challenge_company.json` 等历史产物，因此未启动任何 GPU 训练。
- 资产发现：本地 `artifacts/remote_archive/embeddings/company_event_train` 已保存完整 frozen BGE-M3 train embeddings（与 1525 条 company train article_id 顺序严格对齐），所以无需 GPU、无需读取 external，即可验证此前长期缺失的真实异构 `SVC-BGE disagreement + class balance` rolling acquisition。
- 协议：新增独立 `tools/benchmark_company_temporal_bge_disagreement.py`，固定 `end_fraction=[0.65,0.75,0.85,0.95,1.00]`、review/future 各 15%、review budget 固定 20%。primary 为 history-only production-like SVC；secondary 仅用 history Gold article 的 frozen BGE Top1 exemplar；只在两者 prediction disagreement 中按 SVC 预测类轮转、类内按 SVC margin 排序。review/future Gold 均不参与选样，external 不读取，`external_touched=false`。
- 五窗口结果：future Macro-F1 gain=`+0.002511/+0.016700/-0.007155/+0.017331/-0.001826`；mean=`+0.005512`、median=`+0.002511`、min=`-0.007155`、positive windows=`3/5`、`all_windows_positive=false`。paired-bootstrap 95% CI 五窗均跨 0；其中 end=`0.95` positive probability=`0.9515`、gain=`+0.017331`，但 CI=`[-0.001737,+0.043198]`，仍不能宣称稳定提升。
- Acquisition 诊断：每窗 SVC-vs-BGE disagreement=`85/83/63/71/74`，20% budget 均可实际选满 `46` 条；选中样本的 primary error rate=`45.65%/54.35%/54.35%/60.87%/63.04%`。这显著高于简单 temporal coverage 的错误富集，证明异构 semantic disagreement 确实更会定位当前模型错误；但回流后仍有 2 个 harmed future window，再次说明“挑错效率”与“跨期泛化增益”不是同一目标。
- 决策：**否决 frozen BGE Top1 disagreement + 固定 20% review 作为自动生产 acquisition**，不继续扫描 10/15/25/30% budget、margin 或 Top-K；这不是否定 BGE 价值，而是说明当前静态 BGE disagreement 对 temporal drift 仍不够稳。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external 做任何选择或调参。
- 下一步：若远端 GPU 恢复，仍优先执行已预注册的 seed=`42` class-balanced Triplet duplication-safe OOF + challenge，验证是否能改善 `rare_event/anti_subject_prior`；数据飞轮侧则继续把 100% recent approved Gold refresh 作为当前唯一五窗口 harmed=0 的稳定参考，并等待真实 reviewed Gold 回流。
- 工程：新增 `tools/benchmark_company_temporal_bge_disagreement.py` 与对应单测；正式产物 `reports/company_temporal_svc_bge_disagreement_20pct_five_window.json`。邮件能力当前禁用，无法执行 Gmail profile + send；未向未知地址发送邮件。

## 2026-08-29 冲奖迭代：SVC-BGE disagreement + temporal coverage 50/50 单点复验

- 前置检查：本地无 EventLens/Triplet 同类训练进程，RTX 3050 仅桌面图形占用（约 `1470/4096 MiB`、GPU util `9%`）；上一轮 hybrid 报告不存在且无残留进程，因此没有重复任务。远端 `eventlens-gpu` 本轮实时返回 `Connection refused`，无法可信读取远端 GPU/进程/reports，未启动任何远端训练。
- 协议：只验证一个预声明单点，固定 review budget=`20%`、其中 `50%` 由 `SVC vs frozen BGE Top1` disagreement + predicted-class balance + margin 选择，剩余 `50%` 用 predicted-class-balanced temporal coverage 补足。end fraction 仍固定 `0.65/0.75/0.85/0.95/1.00`；不扫 share/budget/margin/Top-K，review/future Gold 均不参与选样，`external_touched=false`。
- 五窗口结果：future Macro-F1 gain=`-0.014176/+0.051263/-0.003525/-0.007767/-0.019032`；mean=`+0.001353`、median=`-0.007767`、min=`-0.019032`、positive windows=`1/5`、`all_windows_positive=false`。仅 end=`0.75` 明确正向，paired-bootstrap 95% CI=`[+0.016655,+0.087805]`、positive probability=`1.0`；其余 4 窗均未形成稳定正增益，最后窗口 positive probability 仅 `0.177`。
- Acquisition 诊断：每窗严格选满 `46` 条，固定 `23 disagreement + 23 temporal coverage`；选中样本 primary error rate=`30.43%/39.13%/36.96%/41.30%/47.83%`，明显低于纯 disagreement 的 `45.65%~63.04%`。这说明 temporal coverage 的确增加了时间分布覆盖，但同时稀释了错误富集，且没有换来跨期 harmed=0。
- 决策：**否决并冻结 50/50 hybrid acquisition**。不继续搜索 25/75、40/60、budget、margin 或 Top-K；当前数据已足以否定“简单 temporal coverage 可以修复 frozen-BGE disagreement 稳定性”的假设。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external 做选择或调参。
- 下一步：远端 GPU 恢复后仍优先执行已预注册的 seed=`42` class-balanced Triplet duplication-safe OOF + challenge；只有总体 OOF 不退化且 `rare_event/anti_subject_prior` 改善才扩 seed=`17/73`。数据飞轮侧继续以 100% recent approved Gold refresh 的 `5/5` 正向作为当前稳定参考，不再增加 acquisition 配比网格。
- 产物：`reports/company_temporal_svc_bge_hybrid_20pct_five_window.json`。邮件能力当前禁用，无法执行 Gmail profile + send；未发送邮件。

## 2026-08-29 冲奖迭代：Triplet sampler 多样性/ESS train-only 审计

- 前置检查：本地未发现 EventLens/Triplet 同类 Python 训练进程；RTX 3050 Laptop 为桌面占用（约 `1286/4096 MiB`、GPU util `33%`）。远端 `eventlens-gpu` 本轮实时返回 `Connection refused`，无法读取远端 GPU/进程/reports，因此未启动 seed=`42` Triplet GPU 训练。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮 `external_touched=false`。
- 目的：上一轮已经把 full inverse-frequency anchor sampling 收紧为 sqrt inverse-frequency，但只量化了类级 oversample factor。本轮补充一个不需要 GPU 的 train-only 分析，直接在 seed=`42` 的同一 duplication-safe 3-fold fit split 上计算固定 `3 epochs`、replacement sampling 的期望唯一 anchor 覆盖与 sampling effective sample size（ESS），判断是否仍存在过度重复稀有样本的明显风险。
- 三折结果：uniform 的 ESS fraction=`1.000/1.000/1.000`、expected unique anchor fraction=`0.9503/0.9503/0.9503`；sqrt-inverse 的 ESS fraction=`0.7913/0.7813/0.7864`、unique fraction=`0.8988/0.8983/0.8953`；full-inverse 的 ESS fraction=`0.4554/0.4190/0.4521`、unique fraction=`0.7651/0.7610/0.7572`。
- 单 anchor 期望重复强度：sqrt-inverse 在三折最大约 `9.37/9.43/9.37` 次（3 epochs 总计），full-inverse 为 `23.16/23.16/23.00` 次；sqrt-inverse 仍比 uniform 更偏向长尾，但已经显著降低极端重复与有效样本损失。
- 决策：**full inverse-frequency 正式冻结，不再进入 GPU 实验；sqrt inverse-frequency 保留为唯一 class-balanced 单点。** 该审计只能说明训练采样多样性风险下降，不能宣称 Macro-F1 提升。远端 GPU 恢复后仍先跑固定 seed=`42` duplication-safe 3-fold OOF + challenge；只有总体 OOF 不退化且 `rare_event/anti_subject_prior` 有明确 train-side 改善，才扩 seed=`17/73`，仍不触碰 external。
- 工程：新增 `tools/audit_article_triplet_sampler_diversity.py` 与对应单测；正式产物 `reports/article_triplet_sampler_diversity_audit.json`。专项相关测试 `8 passed`，`compileall -q src tools` 与 `git diff --check` 通过。
- 邮件：当前自动运行环境禁用邮件发送能力，无法执行 Gmail profile + send；本轮有新的 train-only 审计结论，已在项目留痕与任务通知中记录，未发送邮件。

## 2026-08-30 冲奖迭代：production 候选 OOF 稳定性门禁补强与离线模型阻塞

- 前置检查：本地未发现 EventLens/Triplet 同类训练进程；RTX 3050 Laptop 约 `1403/4096 MiB`、GPU util `16%`，均为桌面图形进程。远端 `eventlens-gpu` 本轮实时返回 `connect.cqa1.seetacloud.com:27487 Connection refused`，无法可信读取远端 GPU/进程/reports，因此未启动任何远端训练。严格 production-like external 最优继续冻结为 Macro-F1=`0.772131`，本轮未读取 external。
- 设计理由：当前 production 候选的 duplication-safe OOF `0.767373 -> 0.772805`（`+0.005432`）只有点估计，尚缺 paired-bootstrap CI 与逐 fold 一致性证据，不满足“候选生产模型需要多 seed/OOF 或 bootstrap CI”的企业门禁。本轮没有继续设计 acquisition heuristic，而是优先补这一稳定性缺口。
- 工程：`tools/benchmark_complementary_prototype_fusion.py` 新增 `--skip-external`，在该模式下只运行 train OOF；同时新增固定 seed=`20260830`、`n_bootstrap=2000` 的 paired-bootstrap Macro-F1 gain，以及 3 个 duplication-safe fold 的 baseline/candidate Macro-F1 与 `all_folds_positive` 汇总。external gate 原逻辑保留，但 `--skip-external` 时不会进入 tagged test 读取分支。
- 实际运行结论：train-only OOF 审计已真实启动，但本机 `SentenceTransformer` 需要离线加载 `BAAI/bge-m3` 来重建 subject route / event recall；本机 Hugging Face cache 不含该模型完整 `config.json`，且代码明确 `local_files_only=True`，因此抛出 `LocalEntryNotFoundError/OSError` 并 fail-closed。没有生成 `reports/complementary_prototype_fusion_company_oof_stability.json`，因此**本轮没有新的 Macro-F1/CI，可验证结论是“稳定性门禁仍缺证据”而不是模型提升**。
- 决策：不为绕过该阻塞而改成联网下载、替换模型或取消 production route/retrieval 约束，因为这些都会改变被冻结评估口径。下一次远端恢复后，优先在已有完整 BGE-M3 环境运行同一个 `--skip-external` OOF 稳定性审计；若 CI/逐 fold 不稳，则 `0.772805` 只保留为点估计候选，不追加 external 验证。GPU 训练断点仍保持 seed=`42` sqrt-inverse class-balanced Triplet，且必须在 OOF/challenge train-side 通过后才扩 seed。
- 邮件：当前自动运行环境明确禁用邮件发送能力，无法执行 Gmail profile + send；本轮有明确资源/模型阻塞与工程门禁改动，已在项目留痕和任务通知中记录，未发送邮件。
