"""AIOps Sentinel — 이상탐지 대시보드 + AI 챗봇 웹 서버 (Flask)."""
import json
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.chatbot import answer  # noqa: E402

app = Flask(__name__, template_folder="static", static_folder="static")


def _scored():
    path = ROOT / "reports" / "scored.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=[0])


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/summary")
def api_summary():
    df = _scored()
    if df is None:
        return jsonify({"error": "run train.py first"}), 400
    metrics = {}
    report = ROOT / "reports" / "evaluation.json"
    if report.exists():
        metrics = json.loads(report.read_text(encoding="utf-8"))
    anomalies = df[df["is_anomaly"]]
    return jsonify({
        "total_minutes": len(df),
        "anomaly_minutes": int(df["is_anomaly"].sum()),
        "avg_error_rate": round(float(df["error_rate"].mean()), 4),
        "avg_latency": round(float(df["latency_avg"].mean()), 1),
        "worst": anomalies["anomaly_score"].idxmax().isoformat() if len(anomalies) else None,
        "metrics": metrics,
    })


@app.route("/api/timeline")
def api_timeline():
    df = _scored()
    if df is None:
        return jsonify({"error": "run train.py first"}), 400
    # 렌더링 부담을 줄이기 위해 5분 리샘플 + 이상 플래그는 any()
    agg = df.resample("5min").agg({
        "req_count": "mean", "error_rate": "mean", "latency_avg": "mean",
        "anomaly_score": "max", "is_anomaly": "mean",
    }).fillna(0)
    # 5분 중 40% 이상이 이상으로 판정된 구간만 표시 (단발성 오탐 노이즈 억제)
    agg["is_anomaly"] = agg["is_anomaly"] >= 0.4
    return jsonify({
        "ts": [t.isoformat() for t in agg.index],
        "req_count": agg["req_count"].round(1).tolist(),
        "error_rate": (agg["error_rate"] * 100).round(2).tolist(),
        "latency_avg": agg["latency_avg"].round(1).tolist(),
        "anomaly_score": agg["anomaly_score"].round(3).tolist(),
        "is_anomaly": agg["is_anomaly"].astype(bool).tolist(),
    })


@app.route("/api/anomalies")
def api_anomalies():
    df = _scored()
    if df is None:
        return jsonify({"error": "run train.py first"}), 400
    top = df[df["is_anomaly"]].nlargest(12, "anomaly_score")
    rows = []
    for ts, r in top.iterrows():
        if r["error_rate"] > 0.1:
            cause = f"에러 폭증 ({r['error_rate']:.0%})"
        elif r["latency_avg"] > 500:
            cause = f"지연 급증 ({r['latency_avg']:.0f}ms)"
        elif r["req_count_chg"] > 2.5:
            cause = f"트래픽 스파이크 (x{r['req_count_chg']:.1f})"
        else:
            cause = "복합 패턴"
        rows.append({"ts": ts.isoformat(), "score": round(float(r["anomaly_score"]), 3), "cause": cause})
    return jsonify(rows)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    q = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not q:
        return jsonify({"error": "question required"}), 400
    return jsonify(answer(q))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
