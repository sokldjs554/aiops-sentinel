"""학습 파이프라인: 로그 로드 -> 피처 생성 -> 모델 학습 -> 평가 -> 저장."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector import AnomalyDetector, evaluate
from src.features import load_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=Path, default=ROOT / "data" / "raw_logs.log")
    p.add_argument("--labels", type=Path, default=ROOT / "data" / "anomaly_labels.json")
    p.add_argument("--model-out", type=Path, default=ROOT / "models" / "detector.joblib")
    p.add_argument("--report-out", type=Path, default=ROOT / "reports" / "evaluation.json")
    p.add_argument("--contamination", type=float, default=0.02)
    args = p.parse_args()

    print("[train] 피처 생성 중...")
    feat = load_features(args.logs)
    print(f"[train] {len(feat)}개 윈도우, {feat.shape[1]}개 피처")

    det = AnomalyDetector(contamination=args.contamination).fit(feat)
    scored = det.score(feat)
    det.save(args.model_out)

    # 스코어링 결과 저장 (대시보드에서 사용)
    scored_path = ROOT / "reports" / "scored.csv"
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scored_path)

    metrics = evaluate(scored, args.labels)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[train] 모델 저장: {args.model_out}")
    print(f"[train] 분 단위  precision={metrics['minute_level']['precision']} "
          f"recall={metrics['minute_level']['recall']} f1={metrics['minute_level']['f1']}")
    print(f"[train] 이벤트 단위 탐지율: {metrics['event_level']['detected']}/{metrics['event_level']['total']} "
          f"({metrics['event_level']['detection_rate']:.1%})")


if __name__ == "__main__":
    main()
