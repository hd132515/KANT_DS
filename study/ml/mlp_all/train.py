"""AGE 를 뺀 모든 피처로 MLP 를 soft-AUC 로 학습해 이탈을 예측한다.

입력 (ID·survived·AGE 제외 16개 원 피처):
  Month, StudyCnt_M1/StudyCntCnt_M2/M3, StudyIndex_M1~M3, Exam1~3_M1~M3
  → 값: 학습 폴드의 관측값 평균·표준편차로 표준화한 뒤 결측은 0 으로 채운다 (= 원 척도 평균 대체).
  → 결측 여부: 결측이 있는 피처마다 0/1 지시 변수 하나.
  원 척도 0 으로 채우지 않는 이유: Exam1 에는 실제 0점이 있어 결측과 섞인다.

손실: soft-AUC = mean_{i∈이탈, j∈잔류} σ((s_i − s_j)/τ), loss = 1 − soft-AUC. 양성 = 이탈(survived == 0).
  학습 폴드의 모든 (이탈, 잔류) 쌍으로 전체 배치 계산 (≈ 700 × 24,000 쌍).

조기 종료: 학습 폴드에서 10% 를 층화 추출해 inner 검증셋으로 떼고, inner AUC 가 가장 높은 step 의 가중치를 쓴다.
평가: StratifiedKFold(5, shuffle) × 시드 0~4 = 25 fit, 검증 폴드 AUC-ROC (ml/no_exam 과 같은 폴드).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"

DROP = ["ID", "survived", "AGE"]
N_SPLITS = 5
N_REPEATS = 5
INNER_FRAC = 0.1
HIDDEN = 64
TAU = 1.0
LR = 3e-3
WEIGHT_DECAY = 1e-4
MAX_STEPS = 600
EVAL_EVERY = 10
PATIENCE = 10            # 평가 횟수 기준 (= 100 step)

torch.set_num_threads(8)


def load() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(ROOT / "data" / "Study.공유.csv")
    churn = (df["survived"] == 0).to_numpy(dtype=np.int64)
    return df.drop(columns=DROP), churn


def encode(x: pd.DataFrame, fit_rows: np.ndarray, miss_cols: list[str]) -> np.ndarray:
    """fit_rows 의 관측값으로 표준화 → 결측 0 대체, 결측 지시 변수를 붙인다."""
    mu = x.iloc[fit_rows].mean()
    sd = x.iloc[fit_rows].std().replace(0, 1)
    z = ((x - mu) / sd).fillna(0.0)
    ind = x[miss_cols].isna().astype(float).add_prefix("miss_")
    return pd.concat([z, ind], axis=1).to_numpy(dtype=np.float32)


class MLP(torch.nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Linear(d, HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN, 1),
        )

    def forward(self, x):
        return self.f(x).squeeze(-1)


def soft_auc(s_pos: torch.Tensor, s_neg: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid((s_pos[:, None] - s_neg[None, :]) / TAU).mean()


def fit(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[MLP, int, list[dict]]:
    tr, inner = train_test_split(np.arange(len(y)), test_size=INNER_FRAC, stratify=y, random_state=seed)
    xt = torch.tensor(x[tr])
    pos, neg = torch.tensor(y[tr] == 1), torch.tensor(y[tr] == 0)
    xi = torch.tensor(x[inner])

    torch.manual_seed(seed)
    model = MLP(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best, best_step, best_state, bad, curve = -1.0, 0, None, 0, []
    for step in range(1, MAX_STEPS + 1):
        model.train()
        opt.zero_grad()
        s = model(xt)
        sa = soft_auc(s[pos], s[neg])
        (1 - sa).backward()
        opt.step()
        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                inner_auc = roc_auc_score(y[inner], model(xi).numpy())
            curve.append({"step": step, "train_soft_auc": sa.item(), "inner_auc": inner_auc})
            if inner_auc > best:
                best, best_step, bad = inner_auc, step, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_step, curve


def main() -> None:
    OUT.mkdir(exist_ok=True)
    x_raw, churn = load()
    miss_cols = [c for c in x_raw.columns if x_raw[c].isna().any()]
    print(f"rows={len(churn):,}  이탈={churn.sum()} ({churn.mean():.2%})  "
          f"피처 {x_raw.shape[1]} + 결측 지시 {len(miss_cols)} = {x_raw.shape[1] + len(miss_cols)}")

    rows, curves = [], []
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep)
        for fold, (tr, va) in enumerate(skf.split(x_raw, churn)):
            t0 = time.time()
            x = encode(x_raw, tr, miss_cols)
            seed = rep * N_SPLITS + fold
            model, best_step, curve = fit(x[tr], churn[tr], seed)
            with torch.no_grad():
                s = model(torch.tensor(x)).numpy()
            rows.append({
                "repeat": rep, "fold": fold, "best_step": best_step,
                "train_auc": roc_auc_score(churn[tr], s[tr]),
                "inner_auc_best": max(c["inner_auc"] for c in curve),
                "val_auc": roc_auc_score(churn[va], s[va]),
                "sec": round(time.time() - t0, 1),
            })
            curves += [{"repeat": rep, "fold": fold, **c} for c in curve]
            r = rows[-1]
            print(f"rep {rep} fold {fold}: best_step {best_step:3d}  train {r['train_auc']:.4f}  "
                  f"inner {r['inner_auc_best']:.4f}  val {r['val_auc']:.4f}  ({r['sec']}s)", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cv_scores.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "learning_curves.csv", index=False)

    per_rep = res.groupby("repeat")[["train_auc", "val_auc"]].mean()
    print(f"\ntrain AUC {per_rep['train_auc'].mean():.4f} ± {per_rep['train_auc'].std():.4f}")
    print(f"val   AUC {per_rep['val_auc'].mean():.4f} ± {per_rep['val_auc'].std():.4f} "
          f"(반복 간), 폴드 std {res['val_auc'].std():.4f}, best_step 중앙값 {res['best_step'].median():.0f}")


if __name__ == "__main__":
    main()
