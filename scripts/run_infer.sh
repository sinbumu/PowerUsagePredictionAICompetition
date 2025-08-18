#!/bin/bash
set -euo pipefail

MODE="baseline" # baseline|model|ensemble
TRAIN="data/train.csv"
TEST="data/test.csv"
SAMPLE="data/sample_submission.csv"
OUT="outputs/submissions/sub_baseline.csv"
MODELS_ROOT="outputs/models"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --train) TRAIN="$2"; shift 2;;
    --test) TEST="$2"; shift 2;;
    --sample) SAMPLE="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --models_root) MODELS_ROOT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

export TRAIN TEST SAMPLE OUT MODELS_ROOT

if [[ "$MODE" == "baseline" ]]; then
python3 - << 'PY'
import os
import pandas as pd
from src.io import load_train_csv, load_test_csv, align_sample_index
from src.baseline import build_lag_features, predict_baseline

train = load_train_csv(os.environ["TRAIN"]) 
test = load_test_csv(os.environ["TEST"]) 

train_feat, test_feat = build_lag_features(train, test)
yhat = predict_baseline(train_feat, test_feat)

# attach predictions to keys
pred_df = test_feat[["building_id", "timestamp"]].copy()
pred_df["answer"] = yhat.values

# sample keys
sample = align_sample_index(os.environ["SAMPLE"]) 
parts = sample['num_date_time'].str.split('_', n=1, expand=True)
bid = parts[0].astype(int)
dt = pd.to_datetime(parts[1], format='%Y%m%d %H')
sample_keys = pd.DataFrame({
    'building_id': bid.values,
    'timestamp': dt.values,
    'num_date_time': sample['num_date_time'].values,
})

# merge to align order
out = sample_keys.merge(pred_df, on=['building_id','timestamp'], how='left')
assert out['answer'].notna().all(), 'Missing predictions after merge'
sub_out = out[['num_date_time','answer']]
sub_out.to_csv(os.environ["OUT"], index=False)
print(f"Saved: {os.environ['OUT']}, rows={len(sub_out)}")
PY
elif [[ "$MODE" == "model" ]]; then
python3 - << 'PY'
import os
import pandas as pd
from src.io import load_train_csv, load_test_csv, align_sample_index, load_building_info
from src.models_linear import infer_residual_model as infer_linear
from src.models_lgbm import infer_gbm_residual as infer_gbm

root = os.environ['MODELS_ROOT']
train = load_train_csv(os.environ['TRAIN'])
test = load_test_csv(os.environ['TEST'])
from src.io import load_building_info
info = load_building_info('data/building_info.csv')

# pick a model (gbm preferred if exists)
model_dir = os.path.join(root, 'gbm_resid')
use_gbm = os.path.exists(os.path.join(model_dir, 'feature_names.json'))
if use_gbm:
    preds = infer_gbm(train, test, info, model_dir)
else:
    preds = infer_linear(train, test, info, os.path.join(root, 'linear_resid', 'model.npz'))

sample = align_sample_index(os.environ['SAMPLE'])
parts = sample['num_date_time'].str.split('_', n=1, expand=True)
bid = parts[0].astype(int)
ts = pd.to_datetime(parts[1], format='%Y%m%d %H')
keys = pd.DataFrame({'building_id': bid.values, 'timestamp': ts.values, 'num_date_time': sample['num_date_time'].values})
sub = keys.merge(preds, on=['building_id','timestamp'], how='left').rename(columns={'pred':'answer'})
assert sub['answer'].notna().all()
sub[['num_date_time','answer']].to_csv(os.environ['OUT'], index=False)
print('Saved:', os.environ['OUT'], 'rows=', len(sub))
PY
elif [[ "$MODE" == "ensemble" ]]; then
python3 - << 'PY'
import os, json
import pandas as pd
from src.io import load_train_csv, load_test_csv, load_building_info, align_sample_index
from src.models_linear import infer_residual_model as infer_linear
from src.models_lgbm import infer_gbm_residual as infer_gbm
from src.models_cat import infer_cat_residual as infer_cat

root = os.environ['MODELS_ROOT']
train = load_train_csv(os.environ['TRAIN'])
info = load_building_info('data/building_info.csv')
test = load_test_csv(os.environ['TEST'])

# both preds (fall back if missing)
frames = []
lin_path = os.path.join(root, 'linear_resid', 'model.npz')
if os.path.exists(lin_path):
    frames.append(('lin', infer_linear(train, test, info, lin_path)))
model_dir = os.path.join(root, 'gbm_resid')
if os.path.exists(os.path.join(model_dir, 'feature_names.json')):
    frames.append(('gbm', infer_gbm(train, test, info, model_dir)))
cat_dir = os.path.join(root, 'cat_resid')
if os.path.exists(os.path.join(cat_dir, 'feature_names.json')):
    frames.append(('cat', infer_cat(train, test, info, cat_dir)))
assert frames, 'No models available to ensemble'

# merge predictions
pred = frames[0][1].rename(columns={'pred': f'pred_{frames[0][0]}'})
for name, df in frames[1:]:
    pred = pred.merge(df.rename(columns={'pred': f'pred_{name}'}), on=['building_id','timestamp'], how='inner')

# load tuned weights if exists
weights_path = os.path.join(root, 'blend', 'weights_3.json')
if os.path.exists(weights_path):
    meta = json.load(open(weights_path))
    order = meta['models']; w = meta['weights']
    # reorder columns accordingly
    cols = [f'pred_{k}' for k in order if f'pred_{k}' in pred.columns]
    pred['answer'] = sum(w[i]*pred[cols[i]] for i in range(len(cols)))
else:
    cols = [c for c in pred.columns if c.startswith('pred_')]
    pred['answer'] = pred[cols].mean(axis=1)

# align to sample order
sample = align_sample_index(os.environ['SAMPLE'])
parts = sample['num_date_time'].str.split('_', n=1, expand=True)
bid = parts[0].astype(int)
ts = pd.to_datetime(parts[1], format='%Y%m%d %H')
keys = pd.DataFrame({'building_id': bid.values, 'timestamp': ts.values, 'num_date_time': sample['num_date_time'].values})
sub = keys.merge(pred[['building_id','timestamp','answer']], on=['building_id','timestamp'], how='left')
assert sub['answer'].notna().all()
sub[['num_date_time','answer']].to_csv(os.environ['OUT'], index=False)
print('Saved:', os.environ['OUT'], 'rows=', len(sub))
PY
else
  echo "Unknown mode: $MODE"
  exit 1
fi
