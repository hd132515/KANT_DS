"""특정 시험 미응시 패턴의 이탈률이 나머지보다 높은지 검정한다.

1. 피처별: took_ExamE_Mm 이 False(미응시)인 행 vs True 인 행의 이탈률.
2. 패턴별: 9 bit 응시 패턴 하나 vs 그 패턴이 아닌 모든 행의 이탈률 (n ≥ MIN_N 패턴만).
검정은 2×2 Fisher exact (양측), 다중 비교는 Benjamini–Hochberg (각 표 안에서).
이탈률 신뢰구간은 Wilson 95%.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, false_discovery_control, fisher_exact

from train import FEATURES, OUT, load

ROOT = Path(__file__).resolve().parents[2]
MIN_N = 30            # 검정할 패턴의 최소 행 수
PLOT_MIN_N = 100      # 그림에 올릴 패턴의 최소 행 수
ALPHA = 0.05

INK = "#1f1f1d"
INK_MUTED = "#6b6a66"
GRID = "#e4e3df"
POINT = "#2a78d6"
GROUP_COLOR = {"응시": "#2a78d6", "미응시": "#eb6834"}   # 범주 팔레트 1·2번 슬롯

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_MUTED,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
})


def short(f: str) -> str:
    return f.replace("Exam", "E").replace("_", "")      # Exam3_M1 → E3M1


def describe(code: int) -> str:
    missed = [short(f) for k, f in enumerate(FEATURES) if not (code >> k) & 1]
    taken = [short(f) for k, f in enumerate(FEATURES) if (code >> k) & 1]
    if not missed:
        return "전부 응시"
    if not taken:
        return "전부 미응시"
    return f"미응시 {','.join(missed)}" if len(missed) <= len(taken) else f"응시만 {','.join(taken)}"


def compare(in_group: np.ndarray, churn: np.ndarray) -> dict:
    a, n1 = int(churn[in_group].sum()), int(in_group.sum())
    c, n0 = int(churn[~in_group].sum()), int((~in_group).sum())
    odds, p = fisher_exact([[a, n1 - a], [c, n0 - c]])
    ci = binomtest(a, n1).proportion_ci(method="wilson")
    return {
        "n": n1, "n_churn": a, "churn_rate": a / n1, "ci_lo": ci.low, "ci_hi": ci.high,
        "rest_rate": c / n0, "risk_ratio": (a / n1) / (c / n0), "odds_ratio": odds, "p": p,
    }


def finish(tab: pd.DataFrame) -> pd.DataFrame:
    tab["q_bh"] = false_discovery_control(tab["p"])
    tab["sig"] = np.where(tab["q_bh"] < ALPHA, np.where(tab["risk_ratio"] > 1, "높음", "낮음"), "")
    return tab


def plot_patterns(tab: pd.DataFrame, overall: float, path: Path) -> None:
    t = tab[tab["n"] >= PLOT_MIN_N].sort_values("churn_rate")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(t) + 1.6))
    ax.hlines(y, t["ci_lo"] * 100, t["ci_hi"] * 100, color=POINT, lw=2)
    sig = (t["sig"] != "").to_numpy()
    ax.scatter(t["churn_rate"][sig] * 100, y[sig], s=48, color=POINT, zorder=3,
               edgecolor=POINT, linewidth=1.8, label=f"BH q < {ALPHA}")
    ax.scatter(t["churn_rate"][~sig] * 100, y[~sig], s=48, color="white", zorder=3,
               edgecolor=POINT, linewidth=1.8, label="유의하지 않음")
    ax.axvline(overall * 100, color=INK_MUTED, lw=1, ls=(0, (3, 3)))
    ax.annotate(f"전체 {overall:.2%}", (overall * 100, len(t) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=8.5, color=INK_MUTED, va="bottom")
    ax.set_yticks(y, [f"{d}  (n={n:,})" for d, n in zip(t["pattern"], t["n"])], fontsize=8.5)
    ax.set_xlabel("이탈률 (%) · Wilson 95% CI")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=2, labelcolor=INK, fontsize=9)
    ax.set_title(f"응시 패턴별 이탈률 (n ≥ {PLOT_MIN_N})", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def wilson(k: int, n: int) -> tuple[float, float]:
    ci = binomtest(k, n).proportion_ci(method="wilson")
    return ci.low, ci.high


def plot_features(x: np.ndarray, churn: np.ndarray, panels: list[tuple[str, np.ndarray, pd.DataFrame]],
                  path: Path) -> None:
    """행 = 대상 집단, 열 = 시험, 가로축 = 달(시간 순). 미응시·응시 이탈률과 Wilson 95% CI."""
    months = ["M3", "M2", "M1"]
    exams = ["Exam1", "Exam2", "Exam3"]
    fig, axes = plt.subplots(len(panels), len(exams), figsize=(12, 4.2 * len(panels)), sharey=True)
    for r, (title, mask, tab) in enumerate(panels):
        base = churn[mask].mean()
        for c, e in enumerate(exams):
            ax = axes[r, c]
            for i, m in enumerate(months):
                k = FEATURES.index(f"{e}_{m}")
                for j, (grp, sel) in enumerate([("응시", x[:, k] == 1), ("미응시", x[:, k] == 0)]):
                    g = mask & sel
                    n, a = int(g.sum()), int(churn[g].sum())
                    lo, hi = wilson(a, n)
                    xp = i + (j - 0.5) * 0.28
                    ax.vlines(xp, lo * 100, hi * 100, color=GROUP_COLOR[grp], lw=2)
                    ax.scatter(xp, a / n * 100, s=46, color=GROUP_COLOR[grp], zorder=3,
                               edgecolor="white", linewidth=1.2,
                               label=grp if (r, c, i) == (0, 0, 0) else None)
                row = tab.loc[tab["feature"] == f"{e}_{m}"].iloc[0]
                mark = "유의" if row["q_bh"] < ALPHA else "n.s."
                ax.annotate(f"×{row['risk_ratio']:.2f}\n{mark}", (i, 0.99), xycoords=("data", "axes fraction"),
                            ha="center", va="top", fontsize=8.5, color=INK)
            ax.axhline(base * 100, color=INK_MUTED, lw=0.9, ls=(0, (3, 3)))
            ax.set_xticks(range(len(months)), months)
            ax.set_xlim(-0.6, len(months) - 0.4)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_title(f"{e}" + (f"  ·  {title}" if c == 0 else ""), color=INK, fontsize=10.5, loc="left")
        axes[r, 0].set_ylabel(f"이탈률 (%)  ·  점선 = 집단 평균 {base:.2%}")
    axes[0, 0].set_ylim(0, None)
    lo, hi = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(0, hi * 1.18)                  # 위쪽 RR 주석 자리
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=2,
               labelcolor=INK, fontsize=9.5)
    fig.suptitle(f"시험 미응시 vs 응시 이탈률 (Wilson 95% CI, ×위험비, 유의 = BH q < {ALPHA})",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    x, code, churn = load()
    month = pd.read_csv(ROOT / "data" / "Study.공유.csv", usecols=["Month"])["Month"].to_numpy()
    overall = churn.mean()

    feat = finish(pd.DataFrame([{"feature": f, **compare(x[:, k] == 0, churn)}
                                for k, f in enumerate(FEATURES)]))
    feat.to_csv(OUT / "test_missing_by_feature.csv", index=False, encoding="utf-8-sig")

    # 신규 회원은 M2·M3 기록이 없어 시험도 '미응시'로 잡히고 이탈률도 낮다(가입 1–2개월 패턴).
    # 이 교란을 빼려고 3개월 기록(StudyCnt_M3)이 모두 있는 회원만으로 다시 검정한다.
    hist = pd.read_csv(ROOT / "data" / "Study.공유.csv", usecols=["StudyCntCnt_M3"])["StudyCntCnt_M3"].notna().to_numpy()
    feat_hist = finish(pd.DataFrame([{"feature": f, **compare(x[hist, k] == 0, churn[hist])}
                                     for k, f in enumerate(FEATURES)]))
    feat_hist.to_csv(OUT / "test_missing_by_feature_3m_history.csv", index=False, encoding="utf-8-sig")

    plot_features(x, churn, [(f"전체 {len(churn):,}명", np.ones(len(churn), bool), feat),
                             (f"3개월 기록 회원 {hist.sum():,}명", hist, feat_hist)],
                  OUT / "missing_by_feature.png")

    rows = []
    for c in np.unique(code):
        g = code == c
        if g.sum() >= MIN_N:
            rows.append({"code": c, "pattern": describe(c), "n_missed": 9 - bin(c).count("1"),
                         "month_median": float(np.median(month[g])), **compare(g, churn)})
    pat = finish(pd.DataFrame(rows)).sort_values("churn_rate", ascending=False)
    pat.to_csv(OUT / "test_missing_by_pattern.csv", index=False, encoding="utf-8-sig")
    plot_patterns(pat, overall, OUT / "pattern_churn_rate.png")

    pd.set_option("display.width", 200)
    fmt = {"churn_rate": "{:.2%}".format, "rest_rate": "{:.2%}".format, "p": "{:.1e}".format,
           "q_bh": "{:.1e}".format, "risk_ratio": "{:.2f}".format}
    cols = ["n", "n_churn", "churn_rate", "rest_rate", "risk_ratio", "p", "q_bh", "sig"]
    print(f"전체 이탈률 {overall:.2%}\n\n[피처별: 미응시 vs 응시]")
    print(feat.set_index("feature")[cols].to_string(formatters=fmt))
    print(f"\n[피처별, 3개월 기록 회원만: {hist.sum():,}행, 이탈률 {churn[hist].mean():.2%}]")
    print(feat_hist.set_index("feature")[cols].to_string(formatters=fmt))
    print(f"\n[패턴별 vs 나머지, n ≥ {MIN_N}: {len(pat)}개 패턴, 행 {pat['n'].sum():,}]")
    print(pat.set_index("pattern")[["month_median"] + cols].to_string(formatters=fmt))


if __name__ == "__main__":
    main()
