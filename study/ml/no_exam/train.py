"""시험 응시 여부(Exam1~3 × M1~3, 9개 T/F)와 잔류기간 Month 로 이탈을 예측한다.

손실: 미분 가능한 AUC (soft-AUC).
  AUC      = P(s_pos > s_neg)            = mean_{i∈pos, j∈neg} 1[s_i > s_j]
  soft-AUC = mean_{i∈pos, j∈neg} σ((s_i − s_j) / τ)
  loss     = 1 − soft-AUC + λ‖θ‖²
  양성 = 이탈(survived == 0). 동점 쌍은 σ(0) = 0.5 로 AUC 의 동점 처리와 같다.

입력이 이산값이라 서로 다른 입력 조합(셀)이 적다: 9 bit 패턴 512개(관측 195), × Month 39 → 관측 1,865.
모든 (양성, 음성) 행 쌍을 펼치는 대신 셀별 양성·음성 수를 세어 셀 쌍 가중치로 soft-AUC 를 정확히(근사 없이) 계산한다.

피처 세트 (FEATURE_SETS):
  exam         9 bit 만
  exam+month   9 bit + Month. 선형 계열 입력은 [Month/39, log(Month)/log(39)] 두 열
               (가입 초기 1–2개월의 낮은 이탈률처럼 앞쪽이 굽은 형태를 표현하려고 log 항을 둔다).

모델 (모두 같은 손실로 학습):
  linear       s = w·x + b
  interaction  s = w·x + Σ_{k<l} v_kl x_k x_l + b      (2차 상호작용, Month 열 포함)
  mlp          d → 32 → 32 → 1 (ReLU)
  table        s = θ[pattern] (+ φ[Month])              (패턴마다 점수 1개, Month 는 가법 항)
비교용 참고값(경사 학습 아님): rate = 학습 폴드의 셀별 이탈률(가법 평활).

평가: StratifiedKFold(5, shuffle) 를 시드 0~4 로 5번 반복 = 25 fit, 검증 폴드 AUC-ROC.
두 피처 세트는 같은 폴드를 쓰므로 폴드끼리 짝지어 비교할 수 있다.
"""
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"

EXAMS = ["Exam1", "Exam2", "Exam3"]
MONTHS = ["M1", "M2", "M3"]
FEATURES = [f"{e}_{m}" for e in EXAMS for m in MONTHS]
N_PAT = 2 ** len(FEATURES)
MAX_MONTH = 39
FEATURE_SETS = {"exam": False, "exam+month": True}     # 이름 → Month 사용 여부

N_SPLITS = 5
N_REPEATS = 5
TAU = 1.0
LAMBDA = 1e-4
LR = 0.05
STEPS = 500
RATE_PRIOR = 1.0          # rate 참고값의 가법 평활 강도(가상 표본 수)

torch.set_num_threads(4)


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(ROOT / "data" / "Study.공유.csv")
    taken = df[FEATURES].notna()                      # True = 응시(값 있음)
    taken.columns = [f"took_{c}" for c in FEATURES]
    taken.assign(Month=df["Month"], survived=df["survived"]).to_csv(OUT / "took_exam_features.csv", index=False)
    x = taken.to_numpy(dtype=np.int64)
    code = x @ (1 << np.arange(len(FEATURES)))        # 9 bit → 패턴 번호 0..511
    churn = (df["survived"] == 0).to_numpy(dtype=np.int64)
    return x, code, churn


def load_month() -> np.ndarray:
    return pd.read_csv(ROOT / "data" / "Study.공유.csv", usecols=["Month"])["Month"].to_numpy()


@dataclass
class Cells:
    """서로 다른 입력 조합. 행 → 셀 번호는 `row_cell`."""
    x: torch.Tensor        # (n_cell, d) 선형 계열 입력
    code: torch.Tensor     # (n_cell,) 9 bit 패턴 번호
    month: torch.Tensor    # (n_cell,) Month (미사용 시 0)
    row_cell: np.ndarray   # (n_row,)


def make_cells(code: np.ndarray, month: np.ndarray, use_month: bool) -> Cells:
    m = month if use_month else np.zeros_like(month)
    keys, row_cell = np.unique(np.stack([code, m], 1), axis=0, return_inverse=True)
    c, mm = keys[:, 0], keys[:, 1]
    bits = ((c[:, None] >> np.arange(len(FEATURES))) & 1).astype(np.float32)
    cols = [bits]
    if use_month:
        cols.append(np.stack([mm / MAX_MONTH, np.log(mm) / np.log(MAX_MONTH)], 1).astype(np.float32))
    return Cells(torch.tensor(np.concatenate(cols, 1)), torch.tensor(c), torch.tensor(mm), row_cell.ravel())


class Linear(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.f = torch.nn.Linear(d, 1)

    def forward(self, cells: Cells):
        return self.f(cells.x).squeeze(-1)


class Interaction(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.pairs = list(itertools.combinations(range(d), 2))
        self.f = torch.nn.Linear(d + len(self.pairs), 1)

    def forward(self, cells: Cells):
        x = cells.x
        cross = torch.stack([x[:, k] * x[:, l] for k, l in self.pairs], dim=1)
        return self.f(torch.cat([x, cross], dim=1)).squeeze(-1)


class MLP(torch.nn.Module):
    def __init__(self, d: int, hidden: int = 32):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, cells: Cells):
        return self.f(cells.x).squeeze(-1)


class Table(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(N_PAT))
        self.phi = torch.nn.Parameter(torch.zeros(MAX_MONTH + 1))   # Month 미사용 시 phi[0] 만 쓰여 상수항

    def forward(self, cells: Cells):
        return self.theta[cells.code] + self.phi[cells.month]


MODELS = {"linear": Linear, "interaction": Interaction, "mlp": MLP, "table": Table}


def soft_auc(scores: torch.Tensor, n_pos: torch.Tensor, n_neg: torch.Tensor) -> torch.Tensor:
    # 셀 i 의 양성 n_pos[i] 개 × 셀 j 의 음성 n_neg[j] 개가 모두 같은 차이 s_i − s_j 를 갖는다.
    # 양성이 있는 셀만 행으로 둔다(가중치 0 인 행 제거).
    p = n_pos > 0
    w = n_pos[p][:, None] * n_neg[None, :]
    return (w * torch.sigmoid((scores[p][:, None] - scores[None, :]) / TAU)).sum() / w.sum()


def fit(name: str, cells: Cells, n_pos: np.ndarray, n_neg: np.ndarray, seed: int) -> np.ndarray:
    """학습 폴드의 셀별 양성·음성 수로 학습하고 모든 셀의 점수를 돌려준다."""
    torch.manual_seed(seed)
    model = MODELS[name](cells.x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    npos, nneg = torch.tensor(n_pos, dtype=torch.float32), torch.tensor(n_neg, dtype=torch.float32)
    for _ in range(STEPS):
        opt.zero_grad()
        l2 = sum((p ** 2).sum() for p in model.parameters())
        loss = 1 - soft_auc(model(cells), npos, nneg) + LAMBDA * l2
        loss.backward()
        opt.step()
    with torch.no_grad():
        return model(cells).numpy()


def rate_scores(n_pos: np.ndarray, n_neg: np.ndarray) -> np.ndarray:
    prior = n_pos.sum() / (n_pos.sum() + n_neg.sum())
    return (n_pos + RATE_PRIOR * prior) / (n_pos + n_neg + RATE_PRIOR)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    _, code, churn = load()
    month = load_month()
    all_cells = {fs: make_cells(code, month, use) for fs, use in FEATURE_SETS.items()}
    print(f"rows={len(code):,}  이탈={churn.sum()} ({churn.mean():.2%})  "
          + "  ".join(f"{fs}: 셀 {len(c.code):,}" for fs, c in all_cells.items()))

    rows = []
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep)
        for fold, (tr, va) in enumerate(skf.split(code, churn)):
            for fs, cells in all_cells.items():
                rc = cells.row_cell
                n_cell = len(cells.code)
                n_pos = np.bincount(rc[tr][churn[tr] == 1], minlength=n_cell)
                n_neg = np.bincount(rc[tr][churn[tr] == 0], minlength=n_cell)
                scores = {name: fit(name, cells, n_pos, n_neg, seed=rep * N_SPLITS + fold) for name in MODELS}
                scores["rate (참고)"] = rate_scores(n_pos, n_neg)
                for name, s in scores.items():
                    rows.append({
                        "feature_set": fs, "model": name, "repeat": rep, "fold": fold,
                        "train_auc": roc_auc_score(churn[tr], s[rc[tr]]),
                        "val_auc": roc_auc_score(churn[va], s[rc[va]]),
                    })
        print(f"repeat {rep} done")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cv_scores.csv", index=False)

    # 반복별 5-fold 평균 → 그 5개 값의 평균 ± 표준편차
    keys = ["feature_set", "model"]
    per_rep = res.groupby(keys + ["repeat"])[["train_auc", "val_auc"]].mean()
    summary = per_rep.groupby(keys).agg(["mean", "std"]).round(4)
    summary[("val_auc_fold", "std")] = res.groupby(keys)["val_auc"].std().round(4)
    summary = summary.sort_values(("val_auc", "mean"), ascending=False)
    summary.to_csv(OUT / "cv_summary.csv")
    print(summary.to_string())

    # 같은 폴드끼리 짝지은 Month 추가 효과
    wide = res.pivot_table(index=["model", "repeat", "fold"], columns="feature_set", values="val_auc")
    gain = (wide["exam+month"] - wide["exam"]).groupby("model").agg(
        mean="mean", std="std", wins=lambda d: f"{(d > 0).sum()}/{len(d)}")
    print("\n[Month 추가 효과: val AUC (exam+month − exam), 같은 폴드 짝]")
    print(gain.round(4).to_string())

    # 전체 데이터 패턴별 이탈률 (해석용)
    pat = pd.DataFrame({"code": code, "churn": churn}).groupby("code")["churn"].agg(["size", "sum", "mean"])
    pat.columns = ["n", "n_churn", "churn_rate"]
    bits = pd.DataFrame([[(c >> k) & 1 for k in range(len(FEATURES))] for c in pat.index],
                        index=pat.index, columns=[f"took_{f}" for f in FEATURES])
    pat = bits.join(pat).sort_values("n", ascending=False)
    pat.to_csv(OUT / "pattern_stats.csv")


if __name__ == "__main__":
    main()
