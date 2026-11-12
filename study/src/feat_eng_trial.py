"""Exam 별 월간 성적 변화량·변화율을 survived 값에 따라 비교하는 scatter plot (figures/feat_eng_trial/).

시간 순서는 M3(전전전월) → M2(전전월) → M1(직전월). 변화는 나중 달 - 이전 달로 정의해 양수 = 상승.
  변화량 = M(N-1) - M(N),  변화율(%) = (M(N-1) - M(N)) / M(N) × 100   (N = 2, 3)
이전 달 점수가 0 이하(Exam3 에 음수 존재)면 변화율은 정의하지 않는다(NaN).
Total_M = Exam1 + Exam2 + Exam3 (그 달 세 시험이 모두 있을 때만, 하나라도 없으면 NaN).
StudyIndex_sum = StudyIndex_M1 + M2 + M3 (세 달 모두 있을 때만).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures" / "feat_eng_trial"

TARGET = "survived"
LABELS = {0: "0 (이탈)", 1: "1 (잔류)"}
EXAMS = ["Exam1", "Exam2", "Exam3"]
MONTHS = ["M3", "M2", "M1"]          # 시간 순
STEPS = [("M3", "M2"), ("M2", "M1")]   # (이전 달, 나중 달)
CLIP_Q = (0.005, 0.995)                 # 축 범위만 자른다(통계는 전체 값)

POINT = "#2a78d6"
BOX_FILL = "#cde2fb"
INK = "#1f1f1d"
INK_MUTED = "#6b6a66"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_MUTED,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def add_totals(df: pd.DataFrame) -> pd.DataFrame:
    totals = {f"Total_{m}": df[[f"{e}_{m}" for e in EXAMS]].sum(axis=1, min_count=len(EXAMS))
              for m in MONTHS}
    totals["StudyIndex_sum"] = df[[f"StudyIndex_{m}" for m in MONTHS]].sum(axis=1, min_count=len(MONTHS))
    return df.assign(**totals)


def change_features(df: pd.DataFrame) -> pd.DataFrame:
    out = {f"Total_{m}": df[f"Total_{m}"] for m in MONTHS} | {"StudyIndex_sum": df["StudyIndex_sum"]}
    for e in EXAMS + ["Total"]:
        for prev, curr in STEPS:
            a, b = df[f"{e}_{prev}"], df[f"{e}_{curr}"]
            out[f"{e}_diff_{prev}{curr}"] = b - a
            out[f"{e}_rate_{prev}{curr}"] = (b - a) / a.where(a > 0) * 100
    return pd.DataFrame(out).assign(**{TARGET: df[TARGET]})


def draw_box(ax, pos: float, v: pd.Series) -> None:
    # 수염 = 1.5 IQR, 그 밖은 이상치 점
    ax.boxplot([v], positions=[pos], widths=0.5, patch_artist=True, showfliers=True,
               boxprops=dict(facecolor=BOX_FILL, edgecolor=POINT, linewidth=1.2),
               medianprops=dict(color=INK, linewidth=2),
               whiskerprops=dict(color=POINT, linewidth=1.2), capprops=dict(color=POINT, linewidth=1.2),
               flierprops=dict(marker="o", markersize=2.5, markerfacecolor=POINT, markeredgecolor="none",
                               alpha=0.35))


def strip(ax, feat: pd.DataFrame, col: str, rng: np.random.Generator, signed: bool = True,
          box: bool = False) -> None:
    ticks = []
    for y in sorted(feat[TARGET].unique()):
        g = feat.loc[feat[TARGET] == y, col]
        v = g.dropna()
        if box:
            draw_box(ax, y, v)
        else:
            x = y + rng.uniform(-0.28, 0.28, len(v))
            # 잔류(1)가 이탈(0)의 약 34배라 점 밀도가 달라 보이지 않게 투명도를 표본 수에 맞춘다
            ax.scatter(x, v, s=5, color=POINT, alpha=float(np.clip(300 / max(len(v), 1), 0.05, 0.6)),
                       linewidths=0)
            ax.hlines(v.median(), y - 0.36, y + 0.36, color=INK, lw=2)
        ax.annotate(f"중앙값 {v.median():{'+' if signed else ''}.2f}", (y + 0.37, v.median()), va="center",
                    fontsize=8, color=INK, zorder=5,
                    bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))
        ticks.append(f"{LABELS[y]}\nn={len(v):,} · 결측 {g.isna().mean():.1%}")
    if signed:
        ax.axhline(0, color=INK_MUTED, lw=0.8, ls=(0, (2, 2)))
    lo, hi = feat[col].quantile(CLIP_Q)
    pad = (hi - lo) * 0.05
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(sorted(feat[TARGET].unique()), ticks, fontsize=8.5)
    ax.set_xlim(-0.55, 1.95)


def plot_exam(feat: pd.DataFrame, exam: str, path: Path, rng: np.random.Generator, box: bool = False) -> None:
    fig, axes = plt.subplots(2, len(STEPS), figsize=(11, 8.5))
    for j, (prev, curr) in enumerate(STEPS):
        for i, (kind, unit) in enumerate([("diff", "변화량 (점)"), ("rate", "변화율 (%)")]):
            ax = axes[i, j]
            strip(ax, feat, f"{exam}_{kind}_{prev}{curr}", rng, box=box)
            ax.set_title(f"{prev} → {curr} {unit.split()[0]}", color=INK, fontsize=10.5, loc="left")
            ax.set_ylabel(f"{exam}_{curr} - {exam}_{prev}" if kind == "diff" else unit)
    fig.suptitle(f"{exam} 월간 변화 — survived 별 (양수 = 상승, y축은 {CLIP_Q[0]:.1%}–{CLIP_Q[1]:.1%} 분위로 자름)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_totals(feat: pd.DataFrame, path: Path, rng: np.random.Generator, box: bool = False) -> None:
    fig, axes = plt.subplots(1, len(MONTHS), figsize=(15, 4.8), sharey=True)
    for ax, m in zip(axes, MONTHS):
        strip(ax, feat, f"Total_{m}", rng, signed=False, box=box)
        ax.set_title(f"{m} 총점", color=INK, fontsize=10.5, loc="left")
    axes[0].set_ylabel("Exam1 + Exam2 + Exam3")
    # sharey 라 마지막 축의 분위 범위가 남으므로 세 달 공통 범위로 다시 맞춘다
    lo, hi = np.nanquantile(feat[[f"Total_{m}" for m in MONTHS]].to_numpy(), CLIP_Q)
    pad = (hi - lo) * 0.05
    axes[0].set_ylim(lo - pad, hi + pad)
    fig.suptitle(f"월별 시험 총점 — survived 별 (세 시험 모두 있는 행만, y축은 {CLIP_Q[0]:.1%}–{CLIP_Q[1]:.1%} 분위로 자름)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_single(feat: pd.DataFrame, col: str, ylabel: str, title: str, path: Path,
                rng: np.random.Generator, box: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.8))
    strip(ax, feat, col, rng, signed=False, box=box)
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "Study.공유.csv")
    feat = change_features(add_totals(df))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for e in EXAMS:
        plot_exam(feat, e, FIG_DIR / f"{e}_change.png", rng)
    plot_totals(feat, FIG_DIR / "Total_by_month.png", rng)
    plot_exam(feat, "Total", FIG_DIR / "Total_change.png", rng)
    plot_single(feat, "StudyIndex_sum", "StudyIndex_M1 + M2 + M3",
                "StudyIndex 3개월 합 — survived 별\n(세 달 모두 있는 행만)",
                FIG_DIR / "StudyIndex_sum.png", rng)

    # 같은 그림의 box plot 판 (figures/feat_eng_trial/box/, 파일명 동일)
    box_dir = FIG_DIR / "box"
    box_dir.mkdir(exist_ok=True)
    for e in EXAMS:
        plot_exam(feat, e, box_dir / f"{e}_change.png", rng, box=True)
    plot_totals(feat, box_dir / "Total_by_month.png", rng, box=True)
    plot_exam(feat, "Total", box_dir / "Total_change.png", rng, box=True)
    plot_single(feat, "StudyIndex_sum", "StudyIndex_M1 + M2 + M3",
                "StudyIndex 3개월 합 — survived 별\n(세 달 모두 있는 행만)",
                box_dir / "StudyIndex_sum.png", rng, box=True)

    cols = [c for c in feat.columns if c != TARGET]
    summary = feat.groupby(TARGET)[cols].agg(["median", "mean"]).T.round(2)
    summary["n_nan_0"] = [feat.loc[feat[TARGET] == 0, c].isna().sum() for c, _ in summary.index]
    print(summary.to_string())


if __name__ == "__main__":
    main()
