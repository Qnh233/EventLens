from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path("docs/assets")


def _setup():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _box(ax, x, y, w, h, text, fontsize=11):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fill=False, linewidth=1.4)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=14, linewidth=1.2))


def architecture():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    xs = [0.4, 2.4, 4.5, 6.6, 8.7, 10.8]
    labels = ["多源新闻/公告", "事件识别\nCPU 主链路", "主体/事件 Top-K\nBGE 增强", "同源聚合\nEvent Cluster", "Claim→Evidence\nProof-or-Stop", "生命周期/预警\nHITL"]
    for x, label in zip(xs, labels): _box(ax, x, 4.3, 1.55, 1.15, label)
    for x in xs[:-1]: _arrow(ax, x + 1.55, 4.88, x + 2.0, 4.88)
    _box(ax, 4.5, 1.5, 1.55, 1.0, "Hard Case\nRouter")
    _box(ax, 6.6, 1.5, 1.55, 1.0, "DeepSeek\nShadow Expert")
    _box(ax, 8.7, 1.5, 1.55, 1.0, "Verifier /\nAbstain")
    _box(ax, 10.8, 1.5, 1.55, 1.0, "Review Queue\nData Flywheel")
    _arrow(ax, 5.28, 4.3, 5.28, 2.5); _arrow(ax, 6.05, 2.0, 6.6, 2.0); _arrow(ax, 8.15, 2.0, 8.7, 2.0); _arrow(ax, 10.25, 2.0, 10.8, 2.0)
    ax.text(6.5, 6.25, "EventLens 生产级事件智能闭环", ha="center", fontsize=18, fontweight="bold")
    ax.text(6.5, 0.55, "大模型只处理 hard case；证据不足不推送；所有决策保留审计与回流路径", ha="center", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "architecture.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def metrics():
    labels = ["Company\nclassification", "Company\nclustering", "Industry\nclustering", "Claim→Evidence", "Review 15%\noracle"]
    values = [0.772131, 0.929172, 0.853907, 1.0, 0.861547]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score"); ax.set_title("EventLens 核心量化指标")
    ax.axhline(0.85, linestyle="--", linewidth=1, label="0.85 target")
    for bar, value in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, value+0.018, f"{value:.3f}", ha="center", fontsize=10)
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "metrics.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def flywheel():
    fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
    points = [(1,3.8,"OOF Hard Case"),(3.2,4.7,"Review Queue"),(5.6,4.7,"Human Approval"),(8,3.8,"FeedbackRecord"),(8,1.7,"Evaluation"),(5.6,0.8,"Shadow / Release"),(3.2,0.8,"Rollback"),(1,1.7,"Next Data")]
    for x,y,t in points: _box(ax,x,y,1.7,.75,t,10)
    for (x1,y1,_),(x2,y2,_) in zip(points, points[1:]+points[:1]): _arrow(ax,x1+0.85,y1+0.38,x2+0.85,y2+0.38)
    ax.text(5.5,3.0,"15% / 229 条高价值复核\noracle Macro-F1 = 0.861547",ha="center",va="center",fontsize=14,fontweight="bold")
    ax.set_title("受治理的数据飞轮：经验必须经过审批、评估和回滚门禁", fontsize=16, pad=15)
    fig.tight_layout(); fig.savefig(OUT / "data_flywheel.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def main():
    _setup(); OUT.mkdir(parents=True, exist_ok=True); architecture(); metrics(); flywheel(); print(OUT)


if __name__ == "__main__":
    main()
