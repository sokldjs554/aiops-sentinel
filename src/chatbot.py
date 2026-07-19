"""AI 운영 어시스턴트 챗봇.

탐지된 이상 구간과 로그 통계를 컨텍스트로 삼아 운영자의 질문에 답한다.
- ANTHROPIC_API_KEY 또는 OPENAI_API_KEY가 설정되어 있으면 LLM으로 답변 생성
- 키가 없으면 내장 분석 엔진(rule-based)으로 동작해 데모가 항상 가능
"""
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = """당신은 인프라 운영팀의 AIOps 어시스턴트입니다.
아래 이상탐지 결과 컨텍스트를 근거로 운영자의 질문에 한국어로 간결하고 정확하게 답하세요.
근거 없는 추측은 하지 말고, 데이터에 있는 수치를 인용하세요."""


def _load_context():
    scored_path = ROOT / "reports" / "scored.csv"
    if not scored_path.exists():
        return None, None
    scored = pd.read_csv(scored_path, index_col=0, parse_dates=[0])
    anomalies = scored[scored["is_anomaly"]]
    return scored, anomalies


def _summarize_anomalies(scored, anomalies, top_n=8):
    if anomalies is None or anomalies.empty:
        return "탐지된 이상 구간이 없습니다."
    top = anomalies.nlargest(top_n, "anomaly_score")
    lines = []
    for ts, row in top.iterrows():
        cause = []
        if row["error_rate"] > 0.1:
            cause.append(f"에러율 {row['error_rate']:.0%}")
        if row["latency_avg"] > 500:
            cause.append(f"평균지연 {row['latency_avg']:.0f}ms")
        if row["req_count_chg"] > 2.5:
            cause.append(f"트래픽 {row['req_count_chg']:.1f}배 급증")
        lines.append(f"- {ts:%m/%d %H:%M} score={row['anomaly_score']:.3f} ({', '.join(cause) or '복합 패턴'})")
    return (
        f"전체 {len(scored)}분 중 이상 {len(anomalies)}분 탐지.\n주요 이상 시점(상위 {len(top)}건):\n"
        + "\n".join(lines)
    )


def _classify_cause(row) -> str:
    if row["error_rate"] > 0.1:
        return f"에러 폭증 (5xx 비율 {row['error_rate']:.0%}) — 배포 롤백 또는 의존 서비스 장애 여부 확인 권장"
    if row["latency_avg"] > 500:
        return f"지연 급증 (평균 {row['latency_avg']:.0f}ms, p95 {row['latency_p95']:.0f}ms) — DB 슬로우쿼리/GC/네트워크 점검 권장"
    if row["req_count_chg"] > 2.5:
        return f"트래픽 스파이크 (평시 대비 {row['req_count_chg']:.1f}배) — DDoS 또는 이벤트성 트래픽 여부 확인 권장"
    return "복합 패턴 — 상세 로그 드릴다운 필요"


def _rule_based_answer(question: str, scored, anomalies) -> str:
    q = question.lower()
    if scored is None:
        return "아직 학습된 결과가 없습니다. 먼저 `python src/train.py`를 실행해 주세요."

    if any(k in q for k in ["몇", "개수", "count", "how many"]):
        return f"분석 구간 {len(scored)}분 중 총 {len(anomalies)}분이 이상으로 탐지되었습니다."

    if any(k in q for k in ["원인", "왜", "이유", "cause", "why"]):
        if anomalies.empty:
            return "탐지된 이상이 없어 원인 분석 대상이 없습니다."
        worst = anomalies.nlargest(3, "anomaly_score")
        parts = [f"- {ts:%m/%d %H:%M}: {_classify_cause(row)}" for ts, row in worst.iterrows()]
        return "가장 심각한 이상 3건의 추정 원인입니다:\n" + "\n".join(parts)

    if any(k in q for k in ["심각", "worst", "가장", "최근", "언제"]):
        if anomalies.empty:
            return "탐지된 이상이 없습니다."
        ts = anomalies["anomaly_score"].idxmax()
        row = anomalies.loc[ts]
        return (f"가장 심각한 이상은 {ts:%m/%d %H:%M} (score={row['anomaly_score']:.3f})입니다.\n"
                f"진단: {_classify_cause(row)}")

    if any(k in q for k in ["요약", "정리", "summary", "상태", "현황"]):
        return _summarize_anomalies(scored, anomalies)

    return (_summarize_anomalies(scored, anomalies, top_n=3)
            + "\n\n'원인', '가장 심각한 이상', '요약' 등으로 질문하면 더 자세히 분석해 드립니다.")


def _llm_answer(question: str, context: str) -> str | None:
    """API 키가 있으면 LLM 호출, 없거나 실패하면 None."""
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=700,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"[이상탐지 컨텍스트]\n{context}\n\n[질문]\n{question}"}],
            )
            return msg.content[0].text
        if os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI()
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"[이상탐지 컨텍스트]\n{context}\n\n[질문]\n{question}"},
                ],
            )
            return r.choices[0].message.content
    except Exception:
        return None
    return None


def answer(question: str) -> dict:
    scored, anomalies = _load_context()
    context = _summarize_anomalies(scored, anomalies) if scored is not None else "데이터 없음"
    llm = _llm_answer(question, context)
    if llm:
        return {"answer": llm, "engine": "llm"}
    return {"answer": _rule_based_answer(question, scored, anomalies), "engine": "rule-based"}


if __name__ == "__main__":
    print(json.dumps(answer("현재 상태 요약해줘"), ensure_ascii=False, indent=2))
