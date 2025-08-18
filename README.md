# 2025 전력사용량 예측 AI 경진대회


https://dacon.io/competitions/official/236531/overview/description


DACON 에서 진행하는 대회용 


## 일부 파일 설명

_outline.txt : 대회 개요

_rule.txt : 대회 룰

_dataset_info.txt : data/ 경로에 있는 csv들 정보

##

---

## 프로젝트 개요(요약)
- 목표: 100개 건물의 2024-08-25 ~ 08-31 전력사용량(kWh) 예측
- 지표: SMAPE
- 데이터 제약: `test.csv`에는 일조/일사(now) 미포함 → 과거(lag/rolling)만 사용, 필요 시 프록시(now_hat)로 보강(선택)
- 규칙: 외부 데이터/원격 API 금지, 공개 가중치만 허용, 상대경로/UTF-8, pseudo labeling 금지

## 환경
- Python 3.13 (venv 권장)
- 설치: 
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```

## 디렉토리 구조(핵심)
- `data/` : `train.csv`, `test.csv`, `building_info.csv`, `sample_submission.csv`
- `src/` : `config.py`, `io.py`, `features.py`, `cv.py`, `baseline.py`,
  - 모델: `models_linear.py`, `models_lgbm.py`, `models_cat.py`
  - 앙상블: `blend.py`
- `scripts/` : `run_train.sh`, `run_infer.sh`, `check_quality.py`
- `outputs/` : `models/`, `submissions/`

---

## 실행 가이드

### 0) 베이스라인 제출 생성
```bash
source .venv/bin/activate
bash scripts/run_infer.sh --mode baseline \
  --train data/train.csv --test data/test.csv \
  --sample data/sample_submission.csv \
  --out outputs/submissions/sub_baseline.csv
```

### 1) 모델 학습 (잔차 프레임)
- 선형(릿지) + GBM(HistGBR/LGBM 폴백) + CatBoost(없으면 HGB 폴백)
```bash
bash scripts/run_train.sh --model both \
  --train data/train.csv --info data/building_info.csv \
  --save_dir outputs/models
# CatBoost까지 별도 수행 시
bash scripts/run_train.sh --model cat --train data/train.csv --info data/building_info.csv --save_dir outputs/models
```
- 산출물: 각 모델 폴더(`outputs/models/*_resid/`)에 `model.*`, `feature_names.json`, `cv_metrics.json`, `oof.csv`

### 2) 단일 모델 제출
```bash
# GBM 우선, 없으면 선형으로 추론
bash scripts/run_infer.sh --mode model \
  --train data/train.csv --test data/test.csv \
  --sample data/sample_submission.csv \
  --models_root outputs/models \
  --out outputs/submissions/sub_gbm_or_linear.csv

# CatBoost 단일 제출
python3 - << 'PY'
from src.io import load_train_csv, load_test_csv, load_building_info, align_sample_index
from src.models_cat import infer_cat_residual
import pandas as pd

train=load_train_csv('data/train.csv'); test=load_test_csv('data/test.csv'); info=load_building_info('data/building_info.csv')
preds=infer_cat_residual(train,test,info,'outputs/models/cat_resid')
sample=align_sample_index('data/sample_submission.csv')
parts=sample['num_date_time'].str.split('_',n=1,expand=True); bid=parts[0].astype(int); ts=pd.to_datetime(parts[1],format='%Y%m%d %H')
keys=pd.DataFrame({'building_id':bid.values,'timestamp':ts.values,'num_date_time':sample['num_date_time'].values})
keys.merge(preds,on=['building_id','timestamp'],how='left').rename(columns={'pred':'answer'})[['num_date_time','answer']].to_csv('outputs/submissions/sub_cat.csv',index=False)
print('Saved outputs/submissions/sub_cat.csv')
PY
```

### 3) 앙상블 제출
- Equal weights(자동):
```bash
bash scripts/run_infer.sh --mode ensemble \
  --train data/train.csv --test data/test.csv \
  --sample data/sample_submission.csv \
  --models_root outputs/models \
  --out outputs/submissions/sub_ens.csv
```
- OOF 기반 가중 탐색(그리드) → 2모델 또는 3모델 가중 제출
```bash
# 2/3모델 OOF 존재 시, 가중 탐색/저장
python3 - << 'PY'
import os, json, pandas as pd, numpy as np
from src.blend import find_weights_smape_grid

lin=pd.read_csv('outputs/models/linear_resid/oof.csv'); lin['timestamp']=pd.to_datetime(lin['timestamp'])
gbm=pd.read_csv('outputs/models/gbm_resid/oof.csv'); gbm['timestamp']=pd.to_datetime(gbm['timestamp'])
cat=pd.read_csv('outputs/models/cat_resid/oof.csv'); cat['timestamp']=pd.to_datetime(cat['timestamp'])

base=lin.rename(columns={'oof_pred':'oof_lin','load':'load_lin'}).merge(
    gbm.rename(columns={'oof_pred':'oof_gbm','load':'load_gbm'}), on=['building_id','timestamp'], how='inner').merge(
    cat.rename(columns={'oof_pred':'oof_cat','load':'load_cat'}), on=['building_id','timestamp'], how='inner')

y=base['load_lin'].values
preds=[base[c].values for c in ['oof_lin','oof_gbm','oof_cat']]
w,s=find_weights_smape_grid(preds,y,step=0.02)
os.makedirs('outputs/models/blend',exist_ok=True)
json.dump({'models':['lin','gbm','cat'],'weights':w.tolist(),'cv_smape':s}, open('outputs/models/blend/weights_3.json','w'))
print('weights_3:',w,'cv',s)
PY

# 튜닝 가중 앙상블 제출
bash scripts/run_infer.sh --mode ensemble \
  --train data/train.csv --test data/test.csv \
  --sample data/sample_submission.csv \
  --models_root outputs/models \
  --out outputs/submissions/sub_ens_weighted_3.csv
```

### 4) 제출 품질 체크
```bash
python3 scripts/check_quality.py outputs/submissions/sub_ens_weighted_3.csv data/sample_submission.csv
```

---

## 구현 핵심
- 잔차 프레임: `ŷ = clip(lag168 + r̂, 0, ∞)`
- 누수 방지: 모든 rolling은 `shift(1)` 이후, 테스트 라그는 학습 구간으로만 계산
- 일조/일사 now 미포함 대응: 과거 기반(lag/rolling) 유지, (선택) 프록시로 보강 가능
- 시간·기상·메타(PV×irradiance lag/rolling) 피처 설계
- 블록 시계열 CV(7일+gap)로 OOF 산출 및 앙상블 가중 튜닝

---

## 제출 계획 및 SCORE 기록
- 제출 지표: SMAPE(낮을수록 좋음). 하루 제출 3회 제한
- 제출 순서(오래된 → 최신):
  - Day 1: `sub_baseline.csv` → `sub_linear.csv` → `sub_gbm.csv`
  - Day 2: `sub_ens.csv` → `sub_ens_weighted.csv` → `sub_cat.csv`
  - Day 3: `sub_ens_weighted_3.csv`
- 제출 절차:
  1) 제출 전 점검: `python3 scripts/check_quality.py <제출csv> data/sample_submission.csv`
  2) 데이콘 업로드 후 Public 점수 확인
  3) `SCORE.md`의 해당 행에 `Score`와 `Notes` 업데이트(관찰/이상/개선 아이디어)
- 권장: 매일 비슷한 시간대 제출(리더보드 변동 비교 용이)
- 7개 제출 완료 후 `SCORE.md` 기반으로 3일 뒤 고도화 계획 수립(피처/가중/모델 보강)

### SCORE.md란?
- 제출 파일별 스코어/비고를 기록하는 표
- 열: `순번 | 파일 | 생성시각 | Score | Notes`
- 경로: `SCORE.md` (저장소 루트)

---

## 참고
- 데이터/룰 상세는 `_dataset_info.txt`, `_rule.txt`, `_outline.txt` 참고
- 재현성: 상대경로/UTF-8, 요구 패키지는 `requirements.txt` 고정
