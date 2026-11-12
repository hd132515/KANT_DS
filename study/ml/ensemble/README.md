# ensemble — 순위 평균 앙상블

- 구성원 (같은 25 폴드에서 다시 학습):
  - `mlp`: ml/mlp_all MLP (soft-AUC), 초기화 시드 5개 순위 평균 (시드 배깅)
  - `logreg`: 같은 입력 (표준화 + 결측 0 + 결측 지시), 로지스틱 회귀 BCE, class_weight=balanced
  - `hgb`: 원 피처 (AGE 제외, 결측 그대로) HistGradientBoosting, class_weight=balanced, 내부 10% 조기 종료
  - `no_exam`: ml/no_exam 응시 여부 9 bit + Month, interaction (soft-AUC)
- 결합: 검증 폴드 안에서 점수를 순위(0–1)로 바꿔 동일 가중 평균. 가중치 학습 없음.
- 평가: StratifiedKFold(5, shuffle) × 시드 0–4 = 25 폴드. 15개 조합 전부.
- 실행: `../../../.venv/bin/python run.py` (약 7분, CPU). `results/cv_summary.csv`, `results/cv_scores.csv`, `results/oof_rank_preds.csv`.

## 결과 (val AUC, 반복 간 mean ± std, 짝 차이는 최고 단일 mlp 대비)

| 조합 | val AUC | 폴드 std | mlp 대비 | 우위 폴드 |
|---|---:|---:|---:|---:|
| **mlp + logreg + hgb** | **0.687 ± 0.004** | 0.014 | +0.007 | 22/25 |
| mlp + hgb | 0.686 ± 0.004 | 0.014 | +0.007 | 22/25 |
| logreg + hgb | 0.686 ± 0.004 | 0.014 | +0.006 | 17/25 |
| 4개 전부 | 0.684 ± 0.004 | 0.014 | +0.005 | 18/25 |
| mlp + logreg | 0.681 ± 0.004 | 0.015 | +0.002 | 15/25 |
| mlp (시드 5개) | 0.680 ± 0.005 | 0.016 | – | – |
| hgb | 0.678 ± 0.007 | 0.014 | −0.002 | 12/25 |
| logreg | 0.677 ± 0.004 | 0.016 | −0.003 | 10/25 |
| no_exam | 0.621 ± 0.005 | 0.017 | −0.059 | 0/25 |

- 시드 배깅만으로 MLP 가 0.673 (단일 시드, ml/mlp_all) → 0.680.
- 이득은 hgb 를 섞을 때 나온다 (트리 vs 신경망·선형의 다양성). mlp + logreg 는 거의 같은 정보라 +0.002.
- no_exam 은 넣으면 오히려 떨어진다: 결측 지시로 같은 정보를 이미 가진 데다 단독 성능이 0.06 낮아 동일 가중에서 끌어내린다.
- 15개 조합 중 최고를 같은 CV 로 골랐으므로 0.687 은 약간 낙관적이다. 다만 mlp+logreg+hgb 는 22/25 폴드에서 단일 최고보다 높아 방향은 안정적이다.
