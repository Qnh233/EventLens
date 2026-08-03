# 实验日志

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
