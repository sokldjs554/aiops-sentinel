"""IsolationForest 기반 로그 이상탐지 모델."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

FEATURE_COLS = [
    "req_count", "error_rate", "latency_avg", "latency_p95",
    "unique_ips", "error_concentration",
    "req_count_chg", "latency_avg_chg", "error_rate_chg",
]


class AnomalyDetector:
    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        self.contamination = contamination
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.threshold_ = None

    def fit(self, feat: pd.DataFrame):
        X = self.scaler.fit_transform(feat[FEATURE_COLS])
        self.model.fit(X)
        scores = -self.model.score_samples(X)  # 높을수록 이상
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def score(self, feat: pd.DataFrame) -> pd.DataFrame:
        X = self.scaler.transform(feat[FEATURE_COLS])
        scores = -self.model.score_samples(X)
        out = feat.copy()
        out["anomaly_score"] = scores
        out["is_anomaly"] = scores >= self.threshold_
        return out

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model, "threshold": self.threshold_}, path)

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetector":
        obj = joblib.load(path)
        det = cls()
        det.scaler, det.model, det.threshold_ = obj["scaler"], obj["model"], obj["threshold"]
        return det


def evaluate(scored: pd.DataFrame, labels_path) -> dict:
    """주입된 이상 구간 라벨과 비교해 구간 단위 성능 평가."""
    labels = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    start = pd.Timestamp(labels["start"])
    truth = np.zeros(len(scored), dtype=bool)
    idx = scored.index
    for a in labels["anomalies"]:
        s = start + pd.Timedelta(minutes=a["start_min"])
        e = start + pd.Timedelta(minutes=a["end_min"])
        truth |= (idx >= s) & (idx < e)

    pred = scored["is_anomaly"].to_numpy()
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # 이벤트(구간) 단위 탐지율: 구간 내 1분이라도 탐지하면 성공
    detected_events = 0
    for a in labels["anomalies"]:
        s = start + pd.Timedelta(minutes=a["start_min"])
        e = start + pd.Timedelta(minutes=a["end_min"])
        if scored.loc[(idx >= s) & (idx < e), "is_anomaly"].any():
            detected_events += 1

    return {
        "minute_level": {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
                          "tp": tp, "fp": fp, "fn": fn},
        "event_level": {"detected": detected_events, "total": len(labels["anomalies"]),
                         "detection_rate": round(detected_events / len(labels["anomalies"]), 3)},
    }
