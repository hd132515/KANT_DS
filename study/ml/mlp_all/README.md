# mlp_all — AGE 제외 전체 피처 MLP (soft-AUC)

- 입력 31개 = 원 피처 16개 (Month, StudyCnt M1–M3, StudyIndex M1–M3, Exam1–3 × M1–M3) + 결측 지시 15개 (Month 외 전부).
  원 피처는 학습 폴드 관측값으로 표준화 후 결측을 0 으로 채운다 (= 원 척도 평균 대체).
  원 척도 0 대체를 쓰지 않은 이유: Exam1 에 실제 0점이 있어 결측과 섞인다.
- 모델: MLP 31 → 64 → 64 → 1 (ReLU), AdamW lr 3e-3, wd 1e-4.
- 손실: 1 − soft-AUC, 학습 폴드의 모든 (이탈, 잔류) 쌍으로 전체 배치.
- 조기 종료: 학습 폴드의 10% 를 층화 추출한 inner 셋 AUC 최고 step (10 step 마다 평가, patience 100 step).
- 평가: StratifiedKFold(5, shuffle) × 시드 0–4 = 25 fit. `ml/no_exam` 과 같은 폴드.
- 실행: `../../../.venv/bin/python train.py` (약 1분, CPU). 결과 `results/cv_scores.csv`, `results/learning_curves.csv`.

## 결과

| 모델 | train AUC | val AUC | 폴드 std |
|---|---:|---:|---:|
| **MLP (soft-AUC)** | 0.718 | **0.673 ± 0.008** | 0.018 |
| 로지스틱 회귀 (같은 입력, BCE, class_weight=balanced, 참고) | – | 0.677 | 0.015 |
| −StudyIndex_M1 하나 (학습 없음, 참고) | – | 0.661 | – |
| no_exam 최고 (응시 여부 + Month, interaction) | 0.642 | 0.621 ± 0.005 | 0.017 |

- 응시 여부만 쓴 모델보다 +0.05. 이득 대부분은 StudyIndex_M1 하나로 설명된다 (단독 0.661).
- MLP 는 선형 로지스틱 회귀를 넘지 못한다. best step 이 20–110 (중앙값 50) 으로 일찍 멈추고,
  train 0.72 vs val 0.67 로 과적합이 빠르다. 이탈 868명 규모에서는 비선형 용량의 이득이 없다.
- inner 셋의 이탈자는 약 70명뿐이라 조기 종료 기준이 흔들린다 (inner AUC 0.62–0.75).
