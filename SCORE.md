# SCORE 기록표

- 레더보드 지표: SMAPE (낮을수록 성능 우수)
- 제출 정책: 하루 3회 제한. 오래된 제출부터 순차 업로드하며 점수 기록
- 아래 표의 Score/Notes는 제출 후 채워 넣으세요

| 순번 | 파일 | 생성시각(로컬) | Score | Notes |
|---:|---|---|---:|---|
| 1 | outputs/submissions/sub_baseline.csv | 2024-08-18 21:28 | 18.1558674197 | 0.5*lag168 + 0.5*roll_mean_168 베이스라인 |
| 2 | outputs/submissions/sub_linear.csv | 2024-08-18 21:39 | 14.2267167935 | 잔차 릿지 모델 OOF≈10.62 |
| 3 | outputs/submissions/sub_gbm.csv | 2024-08-18 21:53 | 21.1654366059 | 잔차 GBM(HGB) OOF≈11.71 |
| 4 | outputs/submissions/sub_ens.csv | 2024-08-18 22:00 | 15.1766973185 | linear+gbm 평균 앙상블 |
| 5 | outputs/submissions/sub_ens_weighted.csv | 2024-08-18 22:07 | 16.6923054153 | linear+gbm OOF 가중(그리드) |
| 6 | outputs/submissions/sub_cat.csv | 2024-08-18 22:12 | 11.2086699547 | 잔차 CatBoost(폴백 HGB) OOF≈11.79 |
| 7 | outputs/submissions/sub_ens_weighted_3.csv | 2024-08-18 22:14 | 12.3305017468 | 3모델 OOF 가중(예: [0.48,0.18,0.34], CV≈9.66) |
| 8 | outputs/submissions/sub_cat_smape_anchor_bias.csv | 2025-08-20 10:23 | 11.8684615387 | CatBoost: SMAPE 가중+앵커드CV+바이어스 보정, OOF≈10.23 |
| 9 | outputs/submissions/sub_cat_anchor.csv | 2025-08-20 10:29 | 14.5174723908 | CatBoost: 앵커드CV만(가중/바이어스 없음), OOF≈12.06 |
| 10 | outputs/submissions/sub_cat_smape_anchor.csv | 2025-08-20 10:30 | 9.8885150109 | CatBoost: 앵커드CV+SMAPE 가중(바이어스 없음), OOF≈9.82 |
| 11 | outputs/submissions/sub_cat_smape_anchor_nowhat.csv | 2025-08-21 14:16 | 9.7389765023 | CatBoost: 앵커드CV+SMAPE 가중+now_hat(일조/일사), OOF≈9.76 |
| 12 | outputs/submissions/sub_cat_tune_d6_lr005_l2_3.csv | 2025-08-21 14:47 | 9.7142185196 | Cat 튠 A(depth=6, lr=0.05, l2=3), now_hat 포함 |
| 13 | outputs/submissions/sub_cat_tune_d10_lr003_l2_6.csv | 2025-08-21 14:50 | 9.366959767 | Cat 튠 B(depth=10, lr=0.03, l2=6), now_hat 포함 |
| 14 | outputs/submissions/sub_cat_type_ensemble.csv | 2025-08-21 15:08 | 10.6225078129	 | 타입별 소모델 앙상블(depth=8, lr=0.05, l2=3), now_hat 포함 |
| 15 | outputs/submissions/sub_cat_ft_depth9_lr003_l2_6_decay.csv | 2025-08-23 21:30 | 9.6493305678 | Cat ft1(depth=9, lr=0.03, l2=6), time-decay+now_hat |
| 16 | outputs/submissions/sub_cat_ft_depth10_lr0025_l2_6_decay.csv | 2025-08-23 21:36 | 10.0073275212 | Cat ft2(depth=10, lr=0.025, l2=6), time-decay+now_hat |
| 17 | outputs/submissions/sub_cat_ft_depth8_lr0035_l2_4_decay.csv | 2025-08-23 21:40 | 10.0072784267 | Cat ft3(depth=8, lr=0.035, l2=4), time-decay+now_hat |
| 18 | outputs/submissions/sub_cat_tuneb_nodecay_sc150.csv | 2025-08-23 22:17 | 9.5993265207 | Cat tuneB no-decay, smape_c=150 |
| 19 | outputs/submissions/sub_cat_smape_anchor_nowhat_nodecay_sc150.csv | 2025-08-23 22:21 | 9.7844695479 | Cat anchor+SMAPE+now_hat(no-decay), smape_c=150 |
| 20 | outputs/submissions/sub_xgb.csv | 2025-08-24 22:39 | 11.9557977306 | XGBoost 잔차모델(앵커 3주 CV), OOF≈8.50 |
| 21 | outputs/submissions/sub_cat_tuneb_lr0028.csv | 2025-08-24 22:52 | 9.7699871108 | Cat tuneB lr=0.028(no-decay), smape_c=150 |
| 22 | outputs/submissions/sub_cat_tuneb_lr0032.csv | 2025-08-24 22:56 | 9.9657743314 | Cat tuneB lr=0.032(no-decay), smape_c=150 |
| 23 | outputs/submissions/sub_cat_tuneb_l2_7.csv | 2025-08-24 23:00 | 9.9721134586 | Cat tuneB l2=7(no-decay), smape_c=150 |

가이드
- 점수 입력 형식 예: `18.1558674197` -> 해당 사이에서 submission csv 넣을 시 단일 점수만줌. 
- 현재 상위권 유저들은 6~5점 사이 점수, 최상위권은 5.02... 처럼 5에 근접.
- 비고에는 수정사항/관찰사항 기록(예: 야간구간 과대예측, 주말↑ 등)
