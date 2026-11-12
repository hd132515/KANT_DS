"""비전문가용 HTML 보고서를 만든다: src/report_template.html + 그림(base64 내장) → notes/report.html.

`--fragment PATH` 를 주면 <!doctype>/<html> 껍데기 없는 본문 조각도 함께 쓴다(웹 게시용).
그림을 새로 그린 뒤 다시 실행하면 보고서에 반영된다.
"""
import argparse
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "src" / "report_template.html"
OUT = ROOT / "notes" / "report.html"

IMAGES = {
    "survived": ("figures/features/survived.png", "이탈 868명, 잔류 29,796명 막대 그래프"),
    "studyindex_m1": ("figures/label_cond_stat/box/StudyIndex_M1.png", "직전 달 학력 지수 상자 그림, 이탈 중앙값 49.5, 잔류 63"),
    "total_by_month": ("figures/feat_eng_trial/box/Total_by_month.png", "세 달의 시험 총점 상자 그림"),
    "missing_rate": ("figures/label_cond_stat/missing_rate.png", "항목별 빈칸 비율, 이탈과 잔류 비교 막대 그래프"),
    "missing_by_feature": ("ml/no_exam/results/missing_by_feature.png", "시험별 달별 미응시와 응시 회원의 이탈률 비교"),
}

# (이름, AUC) — 각 실험 README 의 25 폴드 평균
MODELS = [
    ("동전 던지기 (기준)", 0.500),
    ("1 · 시험 응시 여부만", 0.609),
    ("2 · + 이용 기간", 0.621),
    ("3 · 전체 기록, 신경망 하나", 0.673),
    ("4 · 신경망 5개 평균", 0.680),
    ("5 · 세 모델 앙상블", 0.687),
]


def img_tag(key: str) -> str:
    path, alt = IMAGES[key]
    data = base64.b64encode((ROOT / path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}" loading="lazy">'


def model_chart() -> str:
    lo, hi = 0.5, 0.70
    w, label_w, right_pad, row_h, top = 640, 210, 52, 34, 10
    x0, x1 = label_w, w - right_pad
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    h = top + row_h * len(MODELS) + 28
    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="단계별 AUC 막대 그래프">']
    for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
        x = sx(t)
        parts.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 4}" y2="{h - 22}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{h - 6}" text-anchor="middle">{t:.2f}</text>')
    best = max(v for _, v in MODELS)
    for i, (name, v) in enumerate(MODELS):
        y = top + i * row_h
        cy = y + row_h / 2
        parts.append(f'<text x="{x0 - 10}" y="{cy + 4:.1f}" text-anchor="end">{name}</text>')
        if v > lo:
            cls = "bar best" if v == best else "bar"
            parts.append(f'<rect class="{cls}" x="{x0}" y="{y + 7}" width="{sx(v) - x0:.1f}" height="{row_h - 14}" rx="3"/>')
        else:
            parts.append(f'<line class="base" x1="{x0}" x2="{x0}" y1="{y + 5}" y2="{y + row_h - 5}"/>')
        parts.append(f'<text class="val" x="{sx(v) + 6:.1f}" y="{cy + 4:.1f}">{v:.3f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def build() -> str:
    html = TEMPLATE.read_text()
    for key in IMAGES:
        html = html.replace(f"@@img:{key}@@", img_tag(key))
    html = html.replace("@@chart:models@@", model_chart())
    assert "@@" not in html, "채우지 않은 자리표시자가 남았다"
    return html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment", type=Path, help="껍데기 없는 본문 조각을 쓸 경로(웹 게시용)")
    args = ap.parse_args()
    body = build()
    OUT.write_text('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
                   '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                   "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    print(f"{OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    if args.fragment:
        args.fragment.write_text(body)
        print(f"{args.fragment}")


if __name__ == "__main__":
    main()
