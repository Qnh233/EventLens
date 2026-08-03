# EventLens

面向上市公司的事件驱动智能识别、可信评估与脉络追踪系统 MVP。

## 当前边界

- 做：赛题 Excel 读取、数据画像、事件识别 baseline、情感方向、同源聚合、可信度评分、严重性评分、预警 JSON。
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

## 配置中心

所有运行配置统一由 `configs/app.yaml` 管理，包括路径、模型参数、聚类阈值、可信度权重和泛化评测参数。配置通过 Pydantic 启动校验，未知字段或类型错误会立即失败，避免配置静默失效。

旧的 `configs/model.yaml`、`configs/cluster.yaml`、`configs/credibility.yaml` 仅保留为历史记录，不再被运行代码读取。

## 数据放置

建议把赛题 Excel 放到：

```text
data/raw/train.xlsx
data/raw/test.xlsx
```

字段支持中文列名：`文章id`、`文章标题`、`发布日期`、`来源网站`、`正文文本`、`实体`、`行业`、`事件`、`事件情感正负面`、`事件影响分析`、`重复性标志`。

## 常用命令

生成数据画像：

```bash
conda run -n eventlens python -m eventlens.cli profile --input data/raw/train.xlsx --output reports/data_profile.md
```

训练 baseline：

```bash
conda run -n eventlens python -m eventlens.cli train --input data/raw/train.xlsx --model-dir artifacts/models
```

预测并输出事件、聚合和预警：

```bash
conda run -n eventlens python -m eventlens.cli predict --input data/raw/test.xlsx --model-dir artifacts/models --output-dir artifacts/run
```

无训练模型时可用规则启发式跑通链路：

```bash
conda run -n eventlens python -m eventlens.cli predict --input data/raw/test.xlsx --output-dir artifacts/run
```

生成真实泛化评测报告（时间/公司/来源切分，包含 Accuracy、Macro-F1、分类指标和混淆矩阵）：

```bash
conda run -n eventlens python -m eventlens.cli generalization-report --input data/raw/train.xlsx --output reports/generalization_eval.md
```

## 留痕位置

- 环境：`environment.yml`
- 统一配置中心：`configs/app.yaml`
- 实验记录：`reports/experiment_log.md`
- 数据画像：`reports/data_profile.md`
- 错误分析：`reports/error_analysis.md`
- 泛化评测：`reports/generalization_eval.md`
