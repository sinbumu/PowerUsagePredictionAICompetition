# Power Usage Prediction – 프로젝트 실행 지도서 (PLAN.md)

## 0) 한눈에 보기

* **목표 지표**: SMAPE ↓
* **기간/룰 핵심**

  * Train: 2024-06-01 \~ 2024-08-24 관측(100건물, 1시간) + 기상
  * Test: 2024-08-25 \~ 2024-08-31 예보 기상
  * **시점 고정**(2024-08-24 23:59:59) 이후 정보 사용 금지
  * 외부 데이터/원격 API 금지, 공개 가중치만 사용 가능, pseudo labeling 금지
* **모델링 전략 요약**

  1. **주간 기준선 + 잔차 예측**(메인): `ŷ = lag168 + f(features)`
  2. **LGBM(메인) + CatBoost(보조)** → 가중 앙상블
  3. (선택) **xPatch/PDMLP** 잔차 예측으로 얹어 소폭 앙상블
  4. (선택) 저부하 시간대 **Amplifier 전처리** A/B 테스트
* **검증(CV)**: 7일 블록, 시작 전 24–48h gap, SMAPE로 통일
* **재현성**: 상대경로, 버전 고정, `run_train.sh` / `run_infer.sh`

---

## 1) 저장소 표준 구조

```
.
├── data/                      # CSV들 (train, test, building_info, sample_submission)
│   ├── train.csv
│   ├── test.csv
│   ├── building_info.csv
│   └── sample_submission.csv
├── src/
│   ├── config.py              # 경로/시드/상수
│   ├── io.py                  # 로딩/저장 유틸
│   ├── features.py            # 누수-안전 피처 생성
│   ├── metrics.py             # smape()
│   ├── cv.py                  # 블록 시계열 CV 분할
│   ├── models_lgbm.py         # 잔차 LGBM 학습/추론
│   ├── models_cat.py          # 잔차 CatBoost
│   ├── models_xpatch.py       # (선택) 잔차 xPatch
│   ├── models_pdmlp.py        # (선택) 잔차 PDMLP
│   ├── blend.py               # NNLS/단순 가중 앙상블
│   ├── infer.py               # test 예측(단일/앙상블)
│   └── baseline.py            # lag168/rolling 베이스라인
├── scripts/
│   ├── run_train.sh
│   └── run_infer.sh
├── outputs/
│   ├── models/                # .txt/.bin 등
│   ├── logs/                  # 학습/검증 로그
│   └── submissions/
├── requirements.txt
├── README.md                  # 대회/실행 요약
└── PLAN.md                    # 본 문서
```

---

## 2) 데이터 계약 & 안전장치

* **키 열**(예시):

  * `building_id`, `timestamp`, `load(kWh)`
  * 기상: `temp, rain, wind, humid, sunshine, irradiance`
  * 메타: `type, total_area, cooling_area, pv_capacity, ess_capacity, pcs_capacity`
* **누수 방지 규칙**

  * 모든 rolling은 `groupby(building).shift(1)` **이후** 적용
  * 라벨/인코더 fit은 **train만** 사용, test는 transform만
  * test 특징 중 **lag168**은 train에서만 참조(재귀 금지)
  * **pseudo labeling 금지**
* **SMAPE**

  ```
  SMAPE = 200 * mean( |y - ŷ| / (|y| + |ŷ| + eps) )
  ```

---

## 3) 모델링 프레임 & 특징 설계 (요약)

* **프레임**: 잔차 예측

  * `r_t = y_t - y_{t-168}`
  * 모델 `f(X_t)`가 `r_t`를 예측 → 최종 `ŷ_t = clip(y_{t-168} + r̂_t, 0, ∞)`
* **시간 특징**: `hour, dow, is_weekend, dayofyear, month`, (sin/cos 주기)
* **부하 라그/롤링**: `lag_1, lag_24, lag_168`, (shift1 후) `roll_mean_24/168, roll_std_24`
* **기상 특징**: 현재(예보) + 1주전(관측 대리) + **차분**(`*_diff = now - lag168`)
* **냉난방 민감도**: `cdd=max(temp-24,0)`, `hdd=max(18-temp,0)`, 주차 차분 포함
* **메타 상호작용**: `pv_capacity*irradiance`, `area` 정규화(선택), `type` 카테고리
* **선택(주파수 분해 피처)**: 24/48/168h 이동평균/잔차로 **저·중·고주파 밴드** 생성

---

## 4) 검증 설계

* **블록 시계열 CV**: 검증구간 = **연속 7일(168h)**, 시작 전 **gap=24–48h**
* 폴드 예:

  * F1: Train ≤ T−21d, Valid = T−14 \~ T−8d
  * F2: Train ≤ T−14d, Valid = T−7 \~ T−1d
  * F3: Train ≤ T−7d,  Valid = T \~ T+6d
* **평가**: 내부도 **SMAPE**로 통일

---

## 5) 실행 시나리오 (제출까지)

1. **EDA 체크**(결측/중복/범위/0비율/유형별 분포)
2. **baseline 제출 파일** 1개 생성 (lag168/roll blend)
3. **LGBM 잔차모델** 학습 → OOF SMAPE 확인
4. **CatBoost** 추가 → **가중 앙상블**
5. (선택) **xPatch/PDMLP** 잔차 모델 얹어 소폭 앙상블
6. (선택) **Amplifier**(잔차 전처리) A/B → 저부하 개선되면 채택
7. **최종 제출 1개** 선택 & README에 재현 절차 명기

---

## 6) 개발 가이드 – Cursor 프롬프트 카드 (짧게, 단계별)

> 각 카드는 **작업 범위가 작고 독립적**입니다.
> 카드 실행 순서대로 PR/커밋하세요. 각 카드에는 **DoD(완료 기준)** 포함.

### Card 01 — 프로젝트 스캐폴딩

**프롬프트**
“`src/`와 `scripts/` 기본 파일을 위 구조로 생성. `requirements.txt`에 pandas, numpy, scikit-learn, lightgbm, catboost, scipy, joblib, tqdm, python-dateutil, pytz를 버전 고정으로 명시. `README.md`에 실행 요약 추가.”
**DoD**

* 폴더/빈 파일 생성, requirements 버전 고정, README 초안 커밋

---

### Card 02 — config & io

**프롬프트**
“`src/config.py`에 상수(시드=42, 경로 상대, eps=1e-6). `src/io.py`에 CSV 로딩/저장 함수, 타입 캐스팅, timestamp 파싱(UTC or local 일관) 구현.”
**DoD**

* train/test/building\_info 로딩 함수 정상 동작
* 모든 timestamp `pd.to_datetime(..., utc=False)` 일관 처리

---

### Card 03 — metrics

**프롬프트**
“`src/metrics.py`에 smape(y, yhat, eps=1e-6) 구현 + 단위테스트(간단) 추가.”
**DoD**

* 0/양수/음수 클립 케이스에서 안정 동작

---

### Card 04 — baseline

**프롬프트**
“`src/baseline.py` 구현: 건물별 정렬 → `lag168`, (shift1 후) `roll_mean_168` → 0.5/0.5 블렌드 후 클립(0,∞). `scripts/run_infer.sh`에서 baseline 제출 CSV 생성.”
**DoD**

* `outputs/submissions/sub_baseline.csv` 생성
* sample\_submission과 인덱스/정렬 일치

---

### Card 05 — features (누수-안전)

**프롬프트**
“`src/features.py`에 시간, 라그/롤링(shift1→rolling), 기상 차분(now−lag168), cdd/hdd, 메타 상호작용 구현. 모든 변환은 **train에서 fit, test는 transform만** 하도록 함수 분리.”
**DoD**

* 학습/추론 경로에서 동일 함수로 재현
* 누수 관련 유닛 체크(현재 시점 미포함)

---

### Card 06 — CV 분할

**프롬프트**
“`src/cv.py`에 7일 블록 검증 분할자 구현(gap=24/48 옵션). fold 마다 train/valid 인덱스 반환. 유닛테스트로 기간 길이/겹침/누수 검증.”
**DoD**

* 기간 정확, fold 간 겹침 없음, gap 준수

---

### Card 07 — LGBM 잔차모델 (학습)

**프롬프트**
“`src/models_lgbm.py`에 잔차 `r = load - lag168` 학습 파이프라인: Dataset 생성, 파라미터(객체=mae/huber), early\_stop, feature importance 저장. OOF SMAPE 출력/로그 저장.”
**DoD**

* `outputs/models/lgbm_*.txt`/`*.bin` 저장
* OOF SMAPE, fold별 SMAPE 로그

---

### Card 08 — 추론 & 단일 제출

**프롬프트**
“`src/infer.py`에 저장된 LGBM 모델로 test 잔차 예측 → 최종 `ŷ = clip(lag168 + r̂, 0,∞)`. `scripts/run_infer.sh`로 단일 제출 생성.”
**DoD**

* `sub_lgbm.csv` 생성, 파일 규격/정렬 준수

---

### Card 09 — CatBoost 보조 모델

**프롬프트**
“`src/models_cat.py` 구현: 카테고리(`type`, `building_id`) 지정, 잔차 학습/추론 동일 파이프라인. 로그/모델 저장.”
**DoD**

* `sub_cat.csv` 생성, OOF SMAPE 로그

---

### Card 10 — 앙상블

**프롬프트**
“`src/blend.py`에 단순 가중/NNLS(비음수) 가중 산출. 입력: fold별 예측/정답. 출력: 최적 가중과 CV SMAPE. `infer.py`에서 가중 적용.”
**DoD**

* 가중/SMAPE 리포트, `sub_ens.csv` 생성

---

### Card 11 — (선택) xPatch 잔차모델

**프롬프트**
“`src/models_xpatch.py`에 xPatch 구조 접목(공식 코드 참고). 입력 채널: weather, lag168(weather), diff, 시간, 메타(시간축 반영). 잔차 예측 학습/추론.”
**DoD**

* xPatch 단일 제출 생성, CV 로그
* LGBM/Cat과 앙상블 비교표

---

### Card 12 — (선택) PDMLP 잔차모델

**프롬프트**
“`src/models_pdmlp.py`에 PDMLP 구성(패치 분할 + 이동평균 분해 → 추세(채널 믹싱)/잔차(채널 독립)). 잔차 예측 학습/추론.”
**DoD**

* PDMLP 단일 제출, CV 로그, 앙상블 비교

---

### Card 13 — (선택) 저부하 보정(A/B)

**프롬프트**
“Amplifier 전처리 아이디어로 잔차의 저에너지 성분 증폭→예측→복원 A/B. 야간/주말 구간 SMAPE 세그먼트 리포트.”
**DoD**

* 세그먼트별 SMAPE 개선 수치, 채택/보류 결정

---

### Card 14 — 리스크/품질 체크

**프롬프트**
“누수 검증 스크립트: (1) test 라인에서 미래 정보 참조 여부 검사, (2) rolling 사용 시 shift1 보장, (3) 제출 음수 클립/NaN 없음.”
**DoD**

* 체크 통과 리포트 저장, 실패 시 CI 실패

---

### Card 15 — 문서/재현 스크립트

**프롬프트**
“`README.md`에 재현 절차(설치→학습→추론→제출)와 환경(OS/버전) 명시. `scripts/run_train.sh` / `scripts/run_infer.sh` 완비.”
**DoD**

* 클린 클론 후 1-command로 재현 확인

---

## 7) 하이퍼파라미터 & 실험 메모

* LGBM: `objective='mae'`(Huber 후보), `num_leaves`(32–128), `lr`(0.02–0.07), `feature/bagging_fraction`(0.7–0.9), `early_stop=200`
* CatBoost: `loss_function=MAE`, `depth`(6–10), `l2_leaf_reg`, `learning_rate`
* xPatch/PDMLP: **잔차 타깃** 기준, 패치 길이(24/48/168) 탐색
* 앙상블: fold OOF 기준 **NNLS** 가중, 과적합 방지로 baseline(lag168) 0.1\~0.3 섞기 실험

---

## 8) 제출/검증 체크리스트

* [ ] 상대경로만 사용(코드/데이터)
* [ ] UTF-8 인코딩, NaN/Inf 무
* [ ] 음수 클립(0 이상)
* [ ] sample\_submission과 ID 정렬/행수 일치
* [ ] 내부 CV SMAPE, 세그먼트(요일/시간/유형) 리포트
* [ ] 최종 1개 제출 파일 선택

---

## 9) 협업/버전 전략(선택)

* 브랜치: `feat/*`, `exp/*`, `fix/*`
* 커밋 태그: `[data]`, `[feat]`, `[model]`, `[exp]`, `[docs]`
* PR 템플릿: 목적/변경점/검증/리스크/재현

---

## 10) 빠른 실행 명령 예시

```bash
# 0) 설치
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) 베이스라인 제출
bash scripts/run_infer.sh --mode baseline \
  --train data/train.csv --test data/test.csv \
  --info data/building_info.csv \
  --out outputs/submissions/sub_baseline.csv

# 2) 학습
bash scripts/run_train.sh --model lgbm \
  --train data/train.csv --info data/building_info.csv \
  --save_dir outputs/models/lgbm/

# 3) 추론(단일 또는 앙상블)
bash scripts/run_infer.sh --mode model \
  --test data/test.csv --info data/building_info.csv \
  --models outputs/models/lgbm/,outputs/models/cat/ \
  --out outputs/submissions/sub_ens.csv
```

---

### 부록: 안전 규칙 리마인더

* 외부 데이터/원격 API 불가, 공개 가중치만 사용 → **로컬 재현** 필수
* 시점 누수 금지(shift→rolling), pseudo labeling 금지
* PPT/코드 제출 시 **Private Score 복원 가능**해야 통과

---