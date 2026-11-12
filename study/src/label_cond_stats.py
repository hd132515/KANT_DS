"""survived 값별 입력 피처 통계(notes/survived_feature_stats.csv)와
라벨값에 따른 피처 scatter plot(figures/label_cond_stat/)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures" / "label_cond_stat"
NOTE_DIR = ROOT / "notes"

ID_COL = "ID"
TARGET = "survived"
LABELS = {0: "0 (이탈)", 1: "1 (잔류)"}

POINT = "#2a78d6"
BOX_FILL = "#cde2fb"
LABEL_COLOR = {1: "#2a78d6", 0: "#eb6834"}   # 범주 팔레트 1·2번 슬롯
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


def label_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for c in features:
        for y, g in df.groupby(TARGET)[c]:
            rows.append({
                "feature": c,
                TARGET: y,
                "n_rows": len(g),
                "n": int(g.notna().sum()),
                "missing": int(g.isna().sum()),
                "missing_pct": round(g.isna().mean() * 100, 2),
                "mean": round(g.mean(), 3),
                "median": round(g.median(), 3),
                "std": round(g.std(), 3),
            })
    return pd.DataFrame(rows)


def draw_box(ax, pos: float, v: pd.Series) -> None:
    # 수염 = 1.5 IQR, 그 밖은 이상치 점
    ax.boxplot([v], positions=[pos], widths=0.5, patch_artist=True, showfliers=True,
               boxprops=dict(facecolor=BOX_FILL, edgecolor=POINT, linewidth=1.2),
               medianprops=dict(color=INK, linewidth=2),
               whiskerprops=dict(color=POINT, linewidth=1.2), capprops=dict(color=POINT, linewidth=1.2),
               flierprops=dict(marker="o", markersize=2.5, markerfacecolor=POINT, markeredgecolor="none",
                               alpha=0.35))


def plot_label_cond(df: pd.DataFrame, c: str, path: Path, rng: np.random.Generator, box: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ticks = []
    for y in sorted(df[TARGET].unique()):
        g = df.loc[df[TARGET] == y, c]
        v = g.dropna()
        if box:
            draw_box(ax, y, v)
        else:
            x = y + rng.uniform(-0.28, 0.28, len(v))
            # 잔류(1)가 이탈(0)의 약 34배라 점 밀도가 달라 보이지 않게 투명도를 표본 수에 맞춘다
            ax.scatter(x, v, s=6, color=POINT, alpha=float(np.clip(300 / max(len(v), 1), 0.05, 0.6)),
                       linewidths=0)
            ax.hlines(v.median(), y - 0.36, y + 0.36, color=INK, lw=2)
        ax.annotate(f"중앙값 {v.median():.2f}", (y + 0.37, v.median()), va="center",
                    fontsize=8.5, color=INK)
        ticks.append(f"{LABELS[y]}\nn={len(v):,} · 결측 {g.isna().mean():.1%}")
    ax.set_xticks(sorted(df[TARGET].unique()), ticks)
    ax.set_xlim(-0.55, 1.85)
    ax.set_xlabel(TARGET)
    ax.set_ylabel(c)
    ax.set_title(f"{c} — survived 별 분포" + (" (box)" if box else ""), color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_missing(stats: pd.DataFrame, path: Path) -> None:
    pct = stats.pivot(index="feature", columns=TARGET, values="missing_pct")
    pct = pct.loc[stats["feature"].unique()]          # 원본 열 순서 유지
    y = np.arange(len(pct))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, 7.5))
    for k, lab in enumerate([0, 1]):
        ax.barh(y + (k - 0.5) * h, pct[lab], height=h, color=LABEL_COLOR[lab],
                edgecolor="white", linewidth=1, label=f"{LABELS[lab]}")
    ax.set_yticks(y, pct.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("결측률 (%)")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2, labelcolor=INK)
    ax.set_title("변수별 결측률 — survived 별", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "Study.공유.csv")
    features = [c for c in df.columns if c not in (ID_COL, TARGET)]

    NOTE_DIR.mkdir(exist_ok=True)
    stats = label_stats(df, features)
    stats.to_csv(NOTE_DIR / "survived_feature_stats.csv", index=False, encoding="utf-8-sig")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for c in features:
        plot_label_cond(df, c, FIG_DIR / f"{c}.png", rng)
    (FIG_DIR / "box").mkdir(exist_ok=True)
    for c in features:
        plot_label_cond(df, c, FIG_DIR / "box" / f"{c}.png", rng, box=True)

    plot_missing(stats, FIG_DIR / "missing_rate.png")

    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
