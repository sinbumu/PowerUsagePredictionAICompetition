# SCORE 기록표

- 레더보드 지표: SMAPE (낮을수록 성능 우수)
- 제출 정책: 하루 3회 제한. 오래된 제출부터 순차 업로드하며 점수 기록
- 아래 표의 Score/Notes는 제출 후 채워 넣으세요

| 순번 | 파일 | 생성시각(로컬) | Score | Notes |
|---:|---|---|---:|---|
| 1 | outputs/submissions/sub_baseline.csv | 2024-08-18 21:28 | 18.1558674197 | 0.5*lag168 + 0.5*roll_mean_168 베이스라인 |
| 2 | outputs/submissions/sub_linear.csv | 2024-08-18 21:39 | 14.2267167935 | 잔차 릿지 모델 OOF≈10.62 |
| 3 | outputs/submissions/sub_gbm.csv | 2024-08-18 21:53 |  | 잔차 GBM(HGB) OOF≈11.71 |
| 4 | outputs/submissions/sub_ens.csv | 2024-08-18 22:00 |  | linear+gbm 평균 앙상블 |
| 5 | outputs/submissions/sub_ens_weighted.csv | 2024-08-18 22:07 |  | linear+gbm OOF 가중(그리드) |
| 6 | outputs/submissions/sub_cat.csv | 2024-08-18 22:12 |  | 잔차 CatBoost(폴백 HGB) OOF≈11.79 |
| 7 | outputs/submissions/sub_ens_weighted_3.csv | 2024-08-18 22:14 |  | 3모델 OOF 가중(예: [0.48,0.18,0.34], CV≈9.66) |

가이드
- 점수 입력 형식 예: `Public=9.87 / Private=?` -> 최종 확정 시 Private 갱신
- 비고에는 수정사항/관찰사항 기록(예: 야간구간 과대예측, 주말↑ 등)
