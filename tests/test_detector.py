"""핵심 로직 단위 테스트: 파싱, 피처, 탐지."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector import FEATURE_COLS, AnomalyDetector
from src.features import build_features, parse_lines

SAMPLE = [
    "2026-07-01T00:00:10 auth-api /login 200 120ms ip=10.0.1.5",
    "2026-07-01T00:00:20 gateway /health 503 900ms ip=10.0.2.9",
    "2026-07-01T00:01:05 billing-api /api/v1/pay 200 95ms ip=10.0.3.1",
    "malformed line should be skipped",
]


def test_parse_lines():
    df = parse_lines(SAMPLE)
    assert len(df) == 3
    assert df["status"].tolist() == [200, 503, 200]
    assert df["latency"].tolist() == [120, 900, 95]


def test_build_features():
    df = parse_lines(SAMPLE)
    feat = build_features(df)
    assert set(FEATURE_COLS) <= set(feat.columns)
    assert feat.iloc[0]["req_count"] == 2
    assert 0.0 <= feat.iloc[0]["error_rate"] <= 1.0


def test_detector_flags_injected_outlier():
    # 정상 데이터 200분 + 명백한 이상치 주입
    import numpy as np
    rng = np.random.default_rng(0)
    n = 200
    feat = pd.DataFrame({
        "req_count": rng.normal(60, 5, n),
        "error_rate": rng.uniform(0, 0.03, n),
        "latency_avg": rng.normal(120, 10, n),
        "latency_p95": rng.normal(220, 15, n),
        "unique_ips": rng.normal(40, 4, n),
        "error_concentration": rng.uniform(0, 0.2, n),
        "req_count_chg": rng.normal(1, 0.05, n),
        "latency_avg_chg": rng.normal(1, 0.05, n),
        "error_rate_chg": rng.normal(1, 0.1, n),
    }, index=pd.date_range("2026-07-01", periods=n, freq="1min"))
    feat.iloc[100] = [60, 0.6, 2500, 4000, 40, 0.9, 1.0, 15.0, 30.0]  # 명백한 장애

    det = AnomalyDetector(contamination=0.02).fit(feat)
    scored = det.score(feat)
    assert scored.iloc[100]["is_anomaly"], "주입된 이상치를 탐지해야 함"
    assert scored["is_anomaly"].mean() < 0.1, "오탐이 과도하면 안 됨"


def test_save_load_roundtrip(tmp_path):
    import numpy as np
    rng = np.random.default_rng(1)
    feat = pd.DataFrame(rng.normal(1, 0.1, (50, len(FEATURE_COLS))), columns=FEATURE_COLS,
                        index=pd.date_range("2026-07-01", periods=50, freq="1min"))
    det = AnomalyDetector().fit(feat)
    p = tmp_path / "m.joblib"
    det.save(p)
    det2 = AnomalyDetector.load(p)
    assert det2.threshold_ == det.threshold_
