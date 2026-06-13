param(
    [int]$RandomSeed = 2021
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
$LogDir = Join-Path $ScriptDir "logs\LongForecasting"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$modelName = "PatchTST"
$modelIdName = "BTCM"
$dataName = "custom"

# BTC minute data setup
$rootPath = $RepoRoot
$dataPath = "data/btc_minutes_100000.csv"
$targetName = "close"
$freqName = "min"

# 120-minute context -> 60-minute forecast
$seqLen = 120
$labelLen = 60
$predLen = 60

$logFile = Join-Path $LogDir "${modelName}_${modelIdName}_${seqLen}_${predLen}.log"
$runFile = Join-Path $ScriptDir "run_longExp.py"

python -u $runFile `
  --random_seed $RandomSeed `
  --is_training 1 `
  --root_path $rootPath `
  --data_path $dataPath `
  --model_id "${modelIdName}_${seqLen}_${predLen}" `
  --model $modelName `
  --data $dataName `
  --features MS `
  --target $targetName `
  --freq $freqName `
  --seq_len $seqLen `
  --label_len $labelLen `
  --pred_len $predLen `
  --enc_in 5 `
  --dec_in 5 `
  --c_out 1 `
  --e_layers 3 `
  --n_heads 4 `
  --d_model 64 `
  --d_ff 256 `
  --dropout 0.1 `
  --fc_dropout 0.1 `
  --head_dropout 0.1 `
  --patch_len 30 `
  --stride 15 `
  --des BTCM_120to60 `
  --train_epochs 10 `
  --itr 1 `
  --batch_size 128 `
  --learning_rate 0.0001 *>&1 | Tee-Object -FilePath $logFile

Write-Host "Done. Log saved to: $logFile"
