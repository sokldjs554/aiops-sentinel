"""로그 파싱 및 시계열 피처 추출.

원시 로그 라인을 1분 단위 윈도우로 집계해 이상탐지 모델의 입력 피처를 만든다.
피처: 요청 수, 에러율, 평균/최대 지연시간, 고유 IP 수, 서비스별 에러 집중도.
"""
import math
import re

import pandas as pd

LOG_RE = re.compile(
    r"^(?P<ts>\S+) (?P<svc>\S+) (?P<path>\S+) (?P<status>\d{3}) (?P<latency>\d+)ms ip=(?P<ip>\S+)$"
)


def parse_lines(lines):
    """로그 라인 iterable -> DataFrame(ts, svc, status, latency, ip)."""
    rows = []
    for line in lines:
        m = LOG_RE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        rows.append((d["ts"], d["svc"], int(d["status"]), int(d["latency"]), d["ip"]))
    df = pd.DataFrame(rows, columns=["ts", "svc", "status", "latency", "ip"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """1분 윈도우 집계 피처 생성."""
    df = df.set_index("ts").sort_index()
    g = df.groupby(pd.Grouper(freq="1min"))

    feat = pd.DataFrame({
        "req_count": g.size(),
        "error_rate": g["status"].apply(lambda s: (s >= 500).mean() if len(s) else 0.0),
        "latency_avg": g["latency"].mean(),
        "latency_p95": g["latency"].quantile(0.95),
        "unique_ips": g["ip"].nunique(),
    }).fillna(0.0)

    # 서비스별 에러 집중도: 특정 서비스에 에러가 몰리면 높아짐 (허핀달 지수)
    def error_concentration(grp):
        errs = grp[grp["status"] >= 500]["svc"].value_counts()
        if errs.sum() == 0:
            return 0.0
        p = errs / errs.sum()
        return float((p ** 2).sum())

    feat["error_concentration"] = g.apply(error_concentration)

    # 단기 변화율 피처 (직전 5분 평균 대비)
    for col in ["req_count", "latency_avg", "error_rate"]:
        roll = feat[col].rolling(5, min_periods=1).mean().shift(1)
        feat[f"{col}_chg"] = (feat[col] / roll.replace(0, 1e-9)).clip(0, 50).fillna(1.0)

    # 일중 주기 인코딩 — 트래픽의 시간대별 패턴을 정상으로 학습하게 함
    minute_of_day = feat.index.hour * 60 + feat.index.minute
    feat["tod_sin"] = [math.sin(2 * math.pi * m / 1440) for m in minute_of_day]
    feat["tod_cos"] = [math.cos(2 * math.pi * m / 1440) for m in minute_of_day]

    return feat.fillna(0.0)


def load_features(log_path) -> pd.DataFrame:
    with open(log_path, encoding="utf-8") as f:
        df = parse_lines(f)
    return build_features(df)
