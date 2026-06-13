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

# Keep these exactly aligned with training config.
$seqLen = 120
$labelLen = 60
$predLen = 60
$desName = "BTCM_120to60"

$setting = "${modelIdName}_${seqLen}_${predLen}_${modelName}_${dataName}_ftMS_sl${seqLen}_ll${labelLen}_pl${predLen}_dm64_nh4_el3_dl1_df256_fc1_ebtimeF_dtTrue_${desName}_0"
$checkpointPath = Join-Path $ScriptDir "checkpoints\$setting\checkpoint.pth"
if (-not (Test-Path $checkpointPath)) {
    throw "Checkpoint not found: $checkpointPath"
}

$logFile = Join-Path $LogDir "${modelName}_${modelIdName}_${seqLen}_${predLen}_test_only.log"
$runFile = Join-Path $ScriptDir "run_longExp.py"

python -u $runFile `
  --random_seed $RandomSeed `
  --is_training 0 `
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
  --des $desName `
  --itr 1 `
  --batch_size 128 `
  --learning_rate 0.0001 `
  *>&1 | Tee-Object -FilePath $logFile

Write-Host "Test-only run complete. Log saved to: $logFile"
