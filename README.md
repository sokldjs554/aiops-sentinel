# AIOps Sentinel — 로그 이상탐지 + AI 운영 어시스턴트

> 서버 로그를 머신러닝으로 분석해 장애 징후(에러 폭증·지연 급증·트래픽 스파이크)를 자동
> 탐지하고, 탐지 결과를 근거로 답하는 AI 챗봇을 결합한 AIOps 데모 프로젝트입니다.
> 네트워크·클라우드 인프라 운영 환경에서 AI를 어떻게 실무에 적용할 수 있는지 보여주는 것을
> 목표로 만들었습니다.

![dashboard](reports/dashboard_demo.png)

## 주요 기능

**1. 로그 파싱 & 시계열 피처 엔지니어링** — 원시 접근 로그를 정규식으로 파싱해 1분 윈도우
단위로 집계합니다. 요청 수, 5xx 에러율, 평균/p95 지연시간, 고유 IP 수, 서비스별 에러
집중도(허핀달 지수), 직전 5분 대비 변화율 등 9개 피처를 생성합니다.

**2. 비지도 이상탐지 (IsolationForest)** — 라벨 없이 정상 패턴을 학습해 이상 구간을
탐지합니다. RobustScaler로 이상치에 강건하게 정규화하고, contamination 기반 분위수로
탐지 임계값을 결정합니다.

**3. 정량 평가** — 로그 생성기가 주입한 이상 구간(ground truth)과 비교해 분 단위
precision/recall/F1과 이벤트(장애 구간) 단위 탐지율을 산출합니다.

| 지표 | 결과 |
|---|---|
| 이벤트 단위 탐지율 | **11/12 (91.7%)** |
| 분 단위 F1 | 0.489 (precision 0.472 / recall 0.507) |
| 분석 구간 | 72시간 (4,320분, 약 18.5만 로그 라인) |

**4. AI 운영 어시스턴트 챗봇** — 탐지 결과를 컨텍스트로 "가장 심각한 이상은 언제야?",
"원인 분석해줘" 같은 자연어 질문에 답합니다. `ANTHROPIC_API_KEY` 또는 `OPENAI_API_KEY`가
있으면 LLM으로 답변을 생성하고, 없으면 내장 rule-based 분석 엔진으로 자동 폴백해
API 키 없이도 데모가 가능합니다. 이상 유형별로 대응 가이드(롤백 확인, 슬로우쿼리 점검,
DDoS 확인 등)를 함께 제시합니다.

**5. 실시간 대시보드** — Flask + 순수 JS(SVG)로 구현한 다크 테마 대시보드. KPI 타일,
이상 지점이 하이라이트된 3개 시계열 차트(크로스헤어 툴팁 지원), 이상 Top 12 테이블,
챗봇 패널을 제공합니다. 외부 차트 라이브러리 없이 동작합니다.

## 아키텍처

```
raw logs ──▶ features.py ──▶ detector.py ──▶ scored.csv / evaluation.json
 (파싱)      (1분 윈도우       (IsolationForest         │
              피처 9종)         + 임계값)               ▼
                                            app.py (Flask API + 대시보드)
                                                  │
                                            chatbot.py (LLM ↔ rule-based 폴백)
```

## 빠른 시작

```bash
pip install -r requirements.txt

# 1) 합성 로그 생성 (72시간, 이상 12건 주입)
python data/generate_logs.py --hours 72 --anomalies 12

# 2) 학습 + 평가
python src/train.py --contamination 0.05

# 3) 대시보드 실행 → http://localhost:5000
python app.py

# (선택) 테스트
pytest tests/
```

## 프로젝트 구조

```
aiops-sentinel/
├── app.py                  # Flask 웹 서버 (대시보드 + REST API)
├── data/generate_logs.py   # 합성 로그 생성기 (이상 시나리오 주입 + 라벨 저장)
├── src/
│   ├── features.py         # 로그 파싱, 1분 윈도우 피처 추출
│   ├── detector.py         # IsolationForest 모델, 평가 로직
│   ├── train.py            # 학습 파이프라인 CLI
│   └── chatbot.py          # AI 어시스턴트 (LLM + rule-based 폴백)
├── static/dashboard.html   # 대시보드 UI (순수 JS/SVG)
├── tests/test_detector.py  # 단위 테스트
└── reports/                # 평가 결과, 스코어링 CSV, 데모 스크린샷
```

## API

| Method | Path | 설명 |
|---|---|---|
| GET | `/api/summary` | KPI + 평가 지표 |
| GET | `/api/timeline` | 5분 리샘플 시계열 (차트용) |
| GET | `/api/anomalies` | 이상 Top 12 + 추정 원인 |
| POST | `/api/chat` | 챗봇 질의 `{"question": "..."}` |

## 기술적 의사결정

- **비지도 학습 선택**: 실제 운영 환경에서는 장애 라벨이 거의 없으므로, 라벨 없이 동작하는
  IsolationForest를 선택했습니다. 합성 데이터의 주입 라벨은 오직 평가에만 사용합니다.
- **변화율 피처**: 절대값만 쓰면 일중 트래픽 곡선을 이상으로 오탐하기 쉬워, 직전 5분 대비
  변화율 피처를 추가해 "평소와 다른 정도"를 학습하게 했습니다.
- **이벤트 단위 평가 병행**: 운영 관점에서는 "장애 구간을 놓치지 않았는가"가 핵심이므로,
  분 단위 지표와 별도로 구간 단위 탐지율(91.7%)을 산출했습니다.
- **LLM 폴백 설계**: 외부 API 의존성을 선택 사항으로 두어, 키가 없는 환경에서도 전체
  데모가 동작하도록 했습니다.

## 향후 개선 방향

시계열 모델(Prophet/LSTM) 기반 예측 잔차 탐지 병행, Drain3를 이용한 로그 템플릿 마이닝,
Kafka 스트리밍 실시간 파이프라인 전환, 알림 연동(Slack/Webhook) 등을 고려하고 있습니다.
