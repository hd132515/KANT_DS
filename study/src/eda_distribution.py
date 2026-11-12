"""Study.공유.csv 피처별 분포 그림(figures/)과 결측치·평균·중앙값 통계(feature_stats.csv)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures" / "features"

ID_COL = "ID"                    # 레코드 구분자 — 분포·통계에서 제외
CATEGORICAL = ["survived"]       # 0 이탈 / 1 잔류
CAT_LABELS = {"survived": {0: "0 (이탈)", 1: "1 (잔류)"}}

BAR = "#2a78d6"
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


def missing_note(s: pd.Series) -> str:
    n_miss = int(s.isna().sum())
    return f"n={s.notna().sum():,} · 결측 {n_miss:,} ({n_miss / len(s):.1%})"


def plot_categorical(s: pd.Series, path: Path) -> None:
    counts = s.value_counts(dropna=False).sort_index()
    labels = [CAT_LABELS.get(s.name, {}).get(k, str(k)) for k in counts.index]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=BAR, width=0.6, edgecolor="white", linewidth=2)
    for b, v in zip(bars, counts.values):
        ax.annotate(f"{v:,}\n({v / len(s):.1%})", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    color=INK, fontsize=9)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("빈도")
    ax.set_title(f"{s.name}\n{missing_note(s)}", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def hist_bins(x: pd.Series):
    # 정수값 피처는 값 하나당 막대 하나(값이 많으면 묶음)로 경계를 정수 사이에 둔다
    if np.allclose(x, x.round()):
        lo, hi = int(x.min()), int(x.max())
        step = max(1, int(np.ceil((hi - lo + 1) / 60)))
        return np.arange(lo - 0.5, hi + step + 0.5, step)
    return min(60, len(np.histogram_bin_edges(x, bins="fd")) - 1)


def plot_continuous(s: pd.Series, path: Path) -> None:
    x = s.dropna()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(x, bins=hist_bins(x), color=BAR, edgecolor="white", linewidth=0.6)
    ax.axvline(x.mean(), color=INK, lw=1.6, label=f"평균 {x.mean():.2f}")
    ax.axvline(x.median(), color=INK, lw=1.6, ls=(0, (4, 3)), label=f"중앙값 {x.median():.2f}")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.set_xlabel(s.name)
    ax.set_ylabel("빈도")
    ax.set_title(f"{s.name}\n{missing_note(s)}", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "Study.공유.csv")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    features = [c for c in df.columns if c != ID_COL]

    rows = []
    for c in features:
        s = df[c]
        kind = "범주형" if c in CATEGORICAL else "연속형"
        (plot_categorical if kind == "범주형" else plot_continuous)(s, FIG_DIR / f"{c}.png")
        rows.append({
            "feature": c,
            "type": kind,
            "n": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(s.isna().mean() * 100, 2),
            "mean": round(s.mean(), 3),
            "median": round(s.median(), 3),
        })

    stats = pd.DataFrame(rows)
    stats.to_csv(ROOT / "feature_stats.csv", index=False, encoding="utf-8-sig")
    print(f"rows={len(df):,}  figures -> {FIG_DIR}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
