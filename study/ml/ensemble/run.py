"""지금까지의 모델을 같은 폴드에서 다시 학습해 OOF 예측을 모으고, 순위 평균 앙상블을 평가한다.

구성원 (모두 양성 = 이탈, 점수가 높을수록 이탈):
  mlp      ml/mlp_all  MLP (soft-AUC), 초기화 시드 5개 평균 (시드 배깅)
  logreg   ml/mlp_all  과 같은 입력(표준화 + 결측 0 + 결측 지시), 로지스틱 회귀 (BCE, class_weight=balanced)
  hgb      원 피처(AGE 제외, 결측 그대로) HistGradientBoosting (class_weight=balanced, 내부 10% 조기 종료)
  no_exam  ml/no_exam  응시 여부 9 bit + Month, interaction 모델 (soft-AUC)

앙상블: 검증 폴드 안에서 각 구성원 점수를 순위(0–1)로 바꿔 동일 가중 평균한다.
  AUC 는 순위만 보므로 척도가 다른 점수를 섞을 때 순위 평균이 안전하다. 가중치는 학습하지 않는다.
평가: StratifiedKFold(5, shuffle) × 시드 0–4 = 25 폴드, 이전 실험과 같은 폴드.
"""
import importlib.util
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
N_SPLITS = 5
N_REPEATS = 5
MLP_SEEDS = 5


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mlp_all = import_path("mlp_all_train", HERE.parent / "mlp_all" / "train.py")
no_exam = import_path("no_exam_train", HERE.parent / "no_exam" / "train.py")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    x_raw, churn = mlp_all.load()
    miss_cols = [c for c in x_raw.columns if x_raw[c].isna().any()]
    _, code, _ = no_exam.load()
    cells = no_exam.make_cells(code, no_exam.load_month(), use_month=True)
    members = ["mlp", "logreg", "hgb", "no_exam"]

    rows, oof = [], []
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep)
        for fold, (tr, va) in enumerate(skf.split(x_raw, churn)):
            t0 = time.time()
            seed = rep * N_SPLITS + fold
            x = mlp_all.encode(x_raw, tr, miss_cols)
            pred = {}

            s = np.zeros(len(va))
            for k in range(MLP_SEEDS):
                model, _, _ = mlp_all.fit(x[tr], churn[tr], seed * 100 + k)
                with torch.no_grad():
                    s += rankdata(model(torch.tensor(x[va])).numpy()) / len(va)
            pred["mlp"] = s / MLP_SEEDS

            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(x[tr], churn[tr])
            pred["logreg"] = lr.decision_function(x[va])

            hgb = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=500, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=30, class_weight="balanced", random_state=seed,
            ).fit(x_raw.iloc[tr], churn[tr])
            pred["hgb"] = hgb.predict_proba(x_raw.iloc[va])[:, 1]

            rc = cells.row_cell
            n_pos = np.bincount(rc[tr][churn[tr] == 1], minlength=len(cells.code))
            n_neg = np.bincount(rc[tr][churn[tr] == 0], minlength=len(cells.code))
            pred["no_exam"] = no_exam.fit("interaction", cells, n_pos, n_neg, seed)[rc[va]]

            ranks = {m: rankdata(pred[m]) / len(va) for m in members}
            for r in range(1, len(members) + 1):
                for combo in itertools.combinations(members, r):
                    s = np.mean([ranks[m] for m in combo], axis=0)
                    rows.append({"combo": "+".join(combo), "size": r, "repeat": rep, "fold": fold,
                                 "val_auc": roc_auc_score(churn[va], s)})
            oof.append(pd.DataFrame({"repeat": rep, "row": va, "churn": churn[va], **ranks}))
            print(f"rep {rep} fold {fold} ({time.time() - t0:.0f}s): "
                  + "  ".join(f"{m} {roc_auc_score(churn[va], pred[m]):.4f}" for m in members)
                  + f"  | all {rows[-1]['val_auc']:.4f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cv_scores.csv", index=False)
    pd.concat(oof).to_csv(OUT / "oof_rank_preds.csv", index=False)

    per_rep = res.groupby(["combo", "size", "repeat"])["val_auc"].mean()
    summary = per_rep.groupby(["combo", "size"]).agg(["mean", "std"])
    summary["fold_std"] = res.groupby(["combo", "size"])["val_auc"].std()
    best_single = summary.xs(1, level="size")["mean"].idxmax()
    wide = res.pivot_table(index=["repeat", "fold"], columns="combo", values="val_auc")
    summary["vs_best_single"] = [(wide[c] - wide[best_single]).mean() for c, _ in summary.index]
    summary["wins"] = [f"{(wide[c] > wide[best_single]).sum()}/{len(wide)}" for c, _ in summary.index]
    summary = summary.sort_values("mean", ascending=False).round(4)
    summary.to_csv(OUT / "cv_summary.csv")
    print(f"\n[val AUC, 반복 간 mean ± std / 폴드 std / 최고 단일({best_single}) 대비 짝 차이·우위 폴드 수]")
    print(summary.to_string())


if __name__ == "__main__":
    main()
