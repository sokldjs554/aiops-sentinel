"""합성 인프라 로그 생성기.

실제 서버 접근 로그와 유사한 패턴(일중 트래픽 곡선 + 노이즈)을 만들고,
운영 장애 시나리오(에러 폭증, 지연 급증, 트래픽 스파이크)를 구간 단위로 주입한다.
주입된 구간은 라벨(ground truth)로 저장되어 모델 평가에 사용된다.
"""
import argparse
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

SERVICES = ["auth-api", "billing-api", "gateway", "search-api", "user-api"]
STATUS_OK = [200, 200, 200, 200, 201, 204, 301, 302]
STATUS_ERR = [500, 502, 503, 504]
PATHS = ["/login", "/api/v1/users", "/api/v1/orders", "/health", "/api/v1/search", "/api/v1/pay"]

ANOMALY_TYPES = ["error_burst", "latency_spike", "traffic_spike"]


def base_rate(minute_of_day: int) -> float:
    """일중 트래픽 곡선: 새벽에 낮고 오후에 피크."""
    h = minute_of_day / 60.0
    return 40 + 35 * math.sin((h - 6) / 24 * 2 * math.pi) + 15 * math.sin(h / 12 * 2 * math.pi)


def generate(hours: int, anomaly_count: int, seed: int, out_dir: Path):
    rng = random.Random(seed)
    start = datetime(2026, 7, 1, 0, 0, 0)
    total_minutes = hours * 60

    # 이상 구간(10~25분) 랜덤 배치 — 서로 겹치지 않게
    anomalies = []
    used = set()
    while len(anomalies) < anomaly_count:
        s = rng.randint(60, total_minutes - 40)
        length = rng.randint(10, 25)
        window = set(range(s - 10, s + length + 10))
        if window & used:
            continue
        used |= set(range(s, s + length))
        anomalies.append({
            "start_min": s,
            "end_min": s + length,
            "type": rng.choice(ANOMALY_TYPES),
            "service": rng.choice(SERVICES),
        })
    anomalies.sort(key=lambda a: a["start_min"])

    def anomaly_at(m):
        for a in anomalies:
            if a["start_min"] <= m < a["end_min"]:
                return a
        return None

    lines = []
    for m in range(total_minutes):
        ts_base = start + timedelta(minutes=m)
        anom = anomaly_at(m)
        rate = max(5, base_rate(m % 1440) + rng.gauss(0, 6))
        err_p, lat_mu = 0.015, 120.0
        if anom:
            if anom["type"] == "traffic_spike":
                rate *= rng.uniform(3.5, 6.0)
            elif anom["type"] == "error_burst":
                err_p = rng.uniform(0.25, 0.55)
            elif anom["type"] == "latency_spike":
                lat_mu = rng.uniform(900, 2500)

        for _ in range(int(rate)):
            svc = anom["service"] if anom and rng.random() < 0.7 else rng.choice(SERVICES)
            is_err = rng.random() < (err_p if not anom or svc == anom["service"] else 0.015)
            status = rng.choice(STATUS_ERR if is_err else STATUS_OK)
            latency = max(3, rng.gauss(lat_mu if not anom or svc == anom["service"] else 120.0, lat_mu * 0.35))
            ts = ts_base + timedelta(seconds=rng.uniform(0, 59))
            lines.append(
                f'{ts.isoformat()} {svc} {rng.choice(PATHS)} {status} {latency:.0f}ms '
                f'ip=10.0.{rng.randint(0, 20)}.{rng.randint(1, 254)}'
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_logs.log").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "anomaly_labels.json").write_text(
        json.dumps({"start": start.isoformat(), "total_minutes": total_minutes, "anomalies": anomalies},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[generate_logs] {len(lines):,} lines, {len(anomalies)} anomaly windows -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, default=72)
    p.add_argument("--anomalies", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = p.parse_args()
    generate(args.hours, args.anomalies, args.seed, args.out)
