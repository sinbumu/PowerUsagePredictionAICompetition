# REFERENCE.md

본 문서는 `PLAN.md`에서 제안한 모델링/학습 설계를 뒷받침하는 최신 레퍼런스를 정리합니다. 각 항목에는 **원문 링크**와 **프로젝트 내 활용 포인트**를 덧붙였습니다.

---

## 1) 핵심 평가 지표 & 베이스라인

* **SMAPE 정의**
  Wikipedia: *Symmetric mean absolute percentage error* — 구현 시 실무에서 자주 쓰는 분모 `|y|+|ŷ|` 형태 언급. 프로젝트의 `metrics.smape()` 정의 근거. ([위키백과][1])

* **Seasonal Naive(주기적 나이브) 베이스라인**
  Hyndman 교재(FPP2/FPP3)의 *Seasonal naïve method* 정리. 본 과제의 `lag168` 주간 기준선 설계 근거. ([Otexts][2])

---

## 2) Gradient Boosting (메인 백본)

* **LightGBM 공식 문서**
  파라미터/파이썬 API 레퍼런스. 본 저장소의 LGBM 설정·튜닝 기준 문서. ([LightGBM][3])

* **CatBoost 문서**
  회귀용 API와 카테고리 처리 방식 확인용. 보조 모델로 채택한 이유(카테고리 안정 처리)의 근거. ([catboost.ai][4])

* **Bisdoulis (2024/2025). Assets Forecasting with Feature Engineering and Transformation Methods for LightGBM**
  EMA·비율/차분·타깃 변환 등 **경량 특성 공학 레시피**를 LGBM에 체계적으로 적용한 연구. 본 프로젝트의 **잔차 예측 + 특성/타깃 변환** 설계에 직접 참고. ([arXiv][5])

---

## 3) 최신 LTSF(장기 시계열 예측) 모델 – 실용 적용

* **xPatch (AAAI 2025)** — *Dual-Stream TS Forecasting with Exponential Seasonal–Trend Decomposition*
  EMA 기반 계절·추세 분해 + 패치화 + 이중 스트림(선형/비선형). **경량 구조로 장주기 패턴 포착**. 잔차 예측용 보조 DL로 제안. (논문/프로시딩/공식 코드) ([arXiv][6], [AAAI][7], [GitHub][8])

* **PDMLP / PatchMLP (AAAI 2025)** — *Patch-based (Decomposed) MLP for LTSF*
  이동평균 분해 후 **추세(채널 믹싱)/잔차(채널 독립)** 처리하는 단순·강력 MLP. 잔차 예측 보조 모델로 적합. (논문/프로시딩) ([arXiv][9], [AAAI][10])

* **TimeKAN (ICLR 2025)** — *KAN-based Frequency Decomposition Learning*
  **주파수 대역 분해(CFD)** → 저/중/고주파 성분 학습 후 혼합. 본 프로젝트에는 **주파수 밴드형 파생특성**(24/48/168h 이동평균·잔차 등) 설계 아이디어로 차용. (논문/오픈리뷰/코드) ([arXiv][11], [OpenReview][12], [GitHub][13])

* **Amplifier (AAAI 2025)** — *Bringing Attention to Neglected Low-Energy Components in TSF*
  **저에너지(야간·저부하) 성분 증폭→예측→복원** 전처리 모듈. SMAPE 분모 이슈가 큰 구간 개선 목적의 A/B 실험 아이디어. (논문/프로시딩/공식 코드) ([arXiv][14], [AAAI][15], [GitHub][16])

---

## 4) 사전학습 TS + 외생변수 주입 (연구 옵션)

* **ChronosX (AISTATS 2025)** — *Adapting Pretrained Time Series Models with Exogenous Variables*
  사전학습 TS 모델에 \*\*외생변수 어댑터(과거/미래 공변량 주입)\*\*를 붙이는 경량 모듈식 접근. 본 과제에서는 연구형 실험/확장 아이디어로 분류. (논문/공식 페이지/학회 채택 리스트) ([arXiv][17], [assets.amazon.science][18], [virtual.aistats.org][19])

---

## 5) 앙상블 가중 추정

* **NNLS (Non-Negative Least Squares)** — SciPy 문서
  비음수 제약 하의 최소제곱. 폴드 OOF 예측을 모아 **가중치 ≥0 제약의 선형 블렌딩**에 사용. ([docs.scipy.org][20])

---

## 6) 에너지·기후 파생 특성 참고

* **Degree Days (CDD/HDD) 개념**
  냉방/난방 도일의 정의 및 활용. `cdd/hdd` 파생특성 정의의 근거. ([위키백과][21])

---

## 부록) 바로가기 모음

* LightGBM Docs(파라미터/파이썬 API) ([LightGBM][22])
* CatBoost Docs/사이트 ([catboost.ai][4])
* xPatch GitHub ([GitHub][8])
* TimeKAN GitHub ([GitHub][13])
* Amplifier GitHub ([GitHub][16])

---

### 인용 가이드

* 구현/설계 문구에는 위 출처를 각 섹션 첫 등장에 1회만 인용했습니다.
* 자세한 재현(코드/하이퍼파라미터)은 각 서브모듈의 README(또는 주석)에서 해당 논문/문서 항목을 **다시** 인용해 주세요.

---

필요하면 `REFERENCE.md`에 “실험 로그/재현 체크리스트” 섹션도 덧붙여 줄게요 (각 논문별 환경/버전, 데이터 사용 범위 등).

[1]: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error?utm_source=chatgpt.com "Symmetric mean absolute percentage error"
[2]: https://otexts.com/fpp2/simple-methods.html?utm_source=chatgpt.com "3.1 Some simple forecasting methods"
[3]: https://lightgbm.readthedocs.io/?utm_source=chatgpt.com "Welcome to LightGBM's documentation! — LightGBM 4.6.0 ..."
[4]: https://catboost.ai/docs/en/concepts/python-reference_catboostregressor?utm_source=chatgpt.com "CatBoostRegressor"
[5]: https://arxiv.org/abs/2501.07580?utm_source=chatgpt.com "Assets Forecasting with Feature Engineering and Transformation Methods for LightGBM"
[6]: https://arxiv.org/pdf/2412.17323?utm_source=chatgpt.com "arXiv:2412.17323v3 [cs.LG] 11 Feb 2025"
[7]: https://ojs.aaai.org/index.php/AAAI/article/view/34270/36425?utm_source=chatgpt.com "xPatch: Dual-Stream Time Series Forecasting with ..."
[8]: https://github.com/stitsyuk/xPatch?utm_source=chatgpt.com "stitsyuk/xPatch: [AAAI 2025] Official implementation of \" ..."
[9]: https://arxiv.org/abs/2405.13575?utm_source=chatgpt.com "Patch-Based MLP for Long-Term Time Series Forecasting"
[10]: https://ojs.aaai.org/index.php/AAAI/article/view/33378/35533?utm_source=chatgpt.com "Patch-Based MLP for Long-Term Time Series Forecasting"
[11]: https://arxiv.org/abs/2502.06910?utm_source=chatgpt.com "TimeKAN: KAN-based Frequency Decomposition Learning ..."
[12]: https://openreview.net/forum?id=wTLc79YNbh&utm_source=chatgpt.com "TimeKAN: KAN-based Frequency Decomposition Learning ..."
[13]: https://github.com/huangst21/TimeKAN?utm_source=chatgpt.com "An offical implementation of \"TimeKAN: KAN-based ..."
[14]: https://arxiv.org/abs/2501.17216?utm_source=chatgpt.com "Amplifier: Bringing Attention to Neglected Low-Energy ..."
[15]: https://ojs.aaai.org/index.php/AAAI/article/view/33267/35422?utm_source=chatgpt.com "Amplifier: Bringing Attention to Neglected Low-Energy ..."
[16]: https://github.com/aikunyi/amplifier?utm_source=chatgpt.com "aikunyi/Amplifier: Official implementation of the paper \" ..."
[17]: https://arxiv.org/abs/2503.12107?utm_source=chatgpt.com "ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables"
[18]: https://assets.amazon.science/d8/62/ef3b58ee4856a8b068d80379e2d8/chronosx-adapting-pretrained-time-series-models-with-exogenous-variables.pdf?utm_source=chatgpt.com "[PDF] ChronosX: Adapting Pretrained Time Series Models with Exogenous ..."
[19]: https://virtual.aistats.org/Conferences/2025/AcceptedPapersList?utm_source=chatgpt.com "AISTATS 2025 Accepted Papers"
[20]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html?utm_source=chatgpt.com "nnls — SciPy v1.16.1 Manual"
[21]: https://en.wikipedia.org/wiki/Degree_day?utm_source=chatgpt.com "Degree day"
[22]: https://lightgbm.readthedocs.io/en/latest/Parameters.html?utm_source=chatgpt.com "Parameters — LightGBM 4.6.0.99 documentation"
