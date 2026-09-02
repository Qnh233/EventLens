from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from eventlens.baseline import TrainedBaseline
from eventlens.config import load_settings
from eventlens.pipeline import run_pipeline
from eventlens.schema import ArticleRecord
from eventlens.trust_control_benchmark import benchmark_trust_controls


class WebArticle(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=20000)
    source: str = Field(default="演示来源", max_length=200)
    publish_time: datetime | None = None


class AnalyzeRequest(BaseModel):
    scope: Literal["company", "industry"] = "company"
    articles: list[WebArticle] = Field(min_length=1, max_length=20)


FINAL_METRICS = {
    "company_production_like_macro_f1": 0.772131,
    "company_cluster_pairwise_f1": 0.929172,
    "industry_cluster_pairwise_f1": 0.853907,
    "claim_evidence_coverage": 1.0,
    "runtime_unsafe_continue_rate": 0.0,
    "review_15pct_oracle_macro_f1": 0.861547,
}


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_root() -> Path:
    return Path(os.getenv("EVENTLENS_MODEL_ROOT", _root() / "deploy" / "models"))


@lru_cache(maxsize=2)
def load_deploy_model(scope: str) -> TrainedBaseline:
    model_dir = _model_root() / scope
    model_path = model_dir / "baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"部署模型不存在: {model_path}. 请先运行 scripts/build_deploy_models.py"
        )
    return TrainedBaseline.load(model_dir)


def analyze_payload(payload: AnalyzeRequest) -> dict:
    try:
        model = load_deploy_model(payload.scope)
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    # Lite 部署模式刻意不接收 labeled-only 主体真值字段，保证网页输入与
    # production-like 评估口径一致；主体解析/BGE/Agent 属于增强部署能力。
    articles = [
        ArticleRecord(
            article_id=f"WEB-{index + 1:04d}",
            title=row.title,
            source=row.source,
            publish_time=row.publish_time or datetime.now(),
            content=row.content,
            entity="",
            industry="",
            trading_code="",
            industry_code="",
            task_scope=f"web_{payload.scope}",
        )
        for index, row in enumerate(payload.articles)
    ]
    settings = load_settings()
    cluster_config = settings.cluster.model_dump()
    cluster_config["semantic"]["enabled"] = False
    result = run_pipeline(
        articles,
        model=model,
        cluster_config=cluster_config,
        skill_registry_path=_root() / "deploy" / "runtime" / "skills.jsonl",
    )
    return {
        "scope": payload.scope,
        "mode": "cpu_lite",
        "articles": [row.model_dump(mode="json") for row in result["predictions"]],
        "clusters": [row.model_dump(mode="json") for row in result["clusters"]],
        "alerts": [row.model_dump(mode="json") for row in result["alerts"]],
        "lifecycles": [row.model_dump(mode="json") for row in result["lifecycles"]],
        "claim_bindings": [row.model_dump(mode="json") for row in result["claim_bindings"]],
        "evidence_gates": [row.model_dump(mode="json") for row in result["evidence_gates"]],
        "learning_signals": [row.model_dump(mode="json") for row in result["learning_signals"]],
        "summary": {
            "article_count": len(result["predictions"]),
            "cluster_count": len(result["clusters"]),
            "alert_count": len(result["alerts"]),
            "blocked_alert_count": sum(
                1 for row in result["alerts"] if not row.delivery_allowed
            ),
        },
    }


app = FastAPI(
    title="EventLens",
    version="0.7.0-final",
    description="上市公司金融事件识别、可信评估、聚合追踪与预警演示服务",
)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = Path(__file__).with_name("web_index.html")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    model_root = _model_root()
    ready = all((model_root / scope / "baseline.joblib").exists() for scope in ("company", "industry"))
    return {"status": "ok" if ready else "degraded", "models_ready": ready, "version": "0.7.0-final"}


@app.get("/api/info")
def info() -> dict:
    return {
        "project": "EventLens",
        "positioning": "金融事件智能闭环：识别→辨伪→聚合→追踪→预警",
        "metrics": FINAL_METRICS,
        "deployment_mode": "CPU Lite; BGE-M3 / DeepSeek are optional enhanced services",
    }


@app.get("/api/trust-demo")
def trust_demo() -> dict:
    """运行真实 Evidence Gate / Skill governance 故障注入，供 Web 演示安全边界。"""
    settings = load_settings()
    return benchmark_trust_controls(
        settings.evidence_control.model_dump()
    ).model_dump(mode="json")


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict:
    try:
        return analyze_payload(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("eventlens.webapp:app", host="0.0.0.0", port=8000, reload=False)
