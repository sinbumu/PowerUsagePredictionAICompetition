#!/bin/bash
set -euo pipefail

MODEL="both" # linear|gbm|both
TRAIN="data/train.csv"
INFO="data/building_info.csv"
OUT_DIR="outputs/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --train) TRAIN="$2"; shift 2;;
    --info) INFO="$2"; shift 2;;
    --save_dir) OUT_DIR="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

export TRAIN INFO OUT_DIR MODEL

python3 - << 'PY'
import os, json
from src.io import load_train_csv, load_building_info
from src.models_linear import train_residual_model
from src.models_lgbm import train_gbm_residual

train = load_train_csv(os.environ['TRAIN'])
info = load_building_info(os.environ['INFO'])
save_root = os.environ['OUT_DIR']
model_sel = os.environ['MODEL']

if model_sel in ('linear','both'):
    m = train_residual_model(train, info, alpha=5.0, save_dir=os.path.join(save_root, 'linear_resid'))
    print('linear:', json.dumps(m, ensure_ascii=False))
if model_sel in ('gbm','both'):
    m = train_gbm_residual(train, info, save_dir=os.path.join(save_root, 'gbm_resid'))
    print('gbm:', json.dumps(m, ensure_ascii=False))
PY
