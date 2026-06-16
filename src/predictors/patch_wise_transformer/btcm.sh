#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

LOG_DIR="$SCRIPT_DIR/logs/LongForecasting"
mkdir -p "$LOG_DIR"

model_name="PatchTST"
model_id_name="BTCM"
data_name="custom"                         # use custom data loader path in training pipeline

# BTC minute data setup
root_path_name="$REPO_ROOT"
data_path_name="data/btc_minutes_100000_with_features.csv"
target_name="future_log_return_60"
freq_name="min"                            # minute frequency

# 240-minute context -> 60-minute horizon -> 1-minute forecast
seq_len=240
label_len=1
pred_len=1
horizon=0

random_seed=2021

python -u "$SCRIPT_DIR/run_longExp.py" \
  --random_seed "$random_seed" \
  --is_training 1 \
  --root_path "$root_path_name" \
  --data_path "$data_path_name" \
  --model_id "${model_id_name}_${seq_len}_${pred_len}" \
  --model "$model_name" \
  --data "$data_name" \
  --features MS \
  --target "$target_name" \
  --freq "$freq_name" \
  --seq_len "$seq_len" \
  --label_len "$label_len" \
  --pred_len "$pred_len" \
  --horizon "$horizon" \
  --enc_in 20 \
  --dec_in 20 \
  --c_out 1 \
  --e_layers 4 \
  --n_heads 4 \
  --d_model 128 \
  --d_ff 512 \
  --dropout 0.05 \
  --fc_dropout 0.05 \
  --head_dropout 0.1 \
  --patch_len 4 \
  --stride 2 \
  --des "BTCM_240to1_mse_more_features_and_log_return_60" \
  --train_epochs 10 \
  --itr 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --log_return True \
  --loss "mse" \
  --patience 5 \
  > "$LOG_DIR/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"