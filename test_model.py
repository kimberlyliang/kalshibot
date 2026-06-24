"""Load a trained PatchTST checkpoint and run a prediction from a CSV."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
PREDICTORS_DIR = REPO_ROOT / "src" / "predictors"
sys.path.insert(0, str(PREDICTORS_DIR))

from exp.exp_main import Exp_Main  # noqa: E402

FUTURE_RETURN_SHIFTS = (
    ("future_log_return_1", 1),
    ("future_log_return_5", 5),
    ("future_log_return_15", 15),
    ("future_log_return_30", 30),
    ("future_log_return_60", 60),
)

MODEL_NAMES = (
    "PatchTST",
    "Autoformer",
    "Informer",
    "Transformer",
    "DLinear",
    "NLinear",
    "Linear",
)

SETTING_RE = re.compile(
    r"_ft(?P<features>MS|M|S)_"
    r"sl(?P<seq_len>\d+)_"
    r"ll(?P<label_len>\d+)_"
    r"pl(?P<pred_len>\d+)_"
    r"dm(?P<d_model>\d+)_"
    r"nh(?P<n_heads>\d+)_"
    r"el(?P<e_layers>\d+)_"
    r"dl(?P<d_layers>\d+)_"
    r"df(?P<d_ff>\d+)_"
    r"fc(?P<factor>\d+)_"
    r"eb(?P<embed>timeF|fixed|learned)_"
    r"dt(?P<distil>True|False)_"
    r"(?P<des>.+)_(?P<itr>\d+)$"
)

# Defaults aligned with BTCM_480_1 training (btcm.sh).
DEFAULT_TRAIN_CONFIG = {
    "seq_len": 480,
    "label_len": 1,
    "pred_len": 1,
    "horizon": 0,
    "enc_in": 30,
    "patch_len": 8,
    "stride": 4,
    "target": "future_log_return_60",
    "feature_ref_csv": "data/btc_minutes_100000_with_features.csv",
    "extra_feature_cols": ("SPY_return_1", "QQQ_return_1", "VIX_return_1"),
}

TIME_COLUMNS = ("open_time", "date", "timestamp", "time", "datetime")


def _pick_time_column(df: pd.DataFrame) -> str:
    for col in TIME_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(f"No timestamp column found. Expected one of: {', '.join(TIME_COLUMNS)}")


def _resolve_checkpoint_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_dir():
        return path / "checkpoint.pth"
    return path


def _patch_num(seq_len: int, patch_len: int, stride: int, padding_patch: str = "end") -> int:
    patch_num = int((seq_len - patch_len) / stride + 1)
    if padding_patch == "end":
        patch_num += 1
    return patch_num


def infer_patchtst_params(state_dict: dict, seq_len_hint: int) -> dict:
    """Read patch_len / stride from checkpoint weights (folder name can be wrong)."""
    w_p_key = next(k for k in state_dict if k.endswith("W_P.weight"))
    w_pos_key = next(k for k in state_dict if k.endswith("W_pos"))

    patch_len = int(state_dict[w_p_key].shape[1])
    patch_num = int(state_dict[w_pos_key].shape[0])

    candidates: list[tuple[int, int, int]] = []
    for stride in (1, 2, 3, 4, 6, 8, 12, 15, 30):
        for seq_len in dict.fromkeys((seq_len_hint, 120, 240, 480)):
            if _patch_num(seq_len, patch_len, stride) == patch_num:
                candidates.append((seq_len, patch_len, stride))

    if not candidates:
        raise ValueError(
            f"Could not infer stride for patch_len={patch_len}, patch_num={patch_num}, "
            f"seq_len_hint={seq_len_hint}"
        )

    seq_len, patch_len, stride = candidates[0]
    return {"seq_len": seq_len, "patch_len": patch_len, "stride": stride}


def count_enc_in(data_path: str, root_path: str, target: str) -> int:
    df = pd.read_csv(Path(root_path) / data_path)
    time_col = _pick_time_column(df)
    cols = [col for col in df.columns if col not in (time_col, target)]
    df = df[[time_col] + cols + [target]]
    return sum(1 for col in df.columns[1:] if pd.api.types.is_numeric_dtype(df[col]))


def _numeric_feature_columns(df: pd.DataFrame, time_col: str, target: str) -> list[str]:
    cols = [col for col in df.columns if col not in (time_col, target)]
    ordered = [time_col] + cols + [target]
    return [col for col in ordered[1:] if pd.api.types.is_numeric_dtype(df[col])]


def align_features_to_training(
    df: pd.DataFrame,
    *,
    root_path: str,
    target: str,
    feature_ref_csv: str,
    enc_in: int,
    extra_feature_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Reorder/pad predict CSV columns to match training schema (enc_in channels)."""
    df = df.copy()
    time_col = _pick_time_column(df)
    ref = pd.read_csv(Path(root_path) / feature_ref_csv, nrows=1)
    ref_time = _pick_time_column(ref)
    ref_features = _numeric_feature_columns(ref, ref_time, target)

    for col in extra_feature_cols:
        if col not in ref_features and len(ref_features) < enc_in:
            ref_features.append(col)

    if len(ref_features) < enc_in:
        for i in range(enc_in - len(ref_features)):
            ref_features.append(f"_pad_{i}")

    ref_features = ref_features[:enc_in]

    out = df[[time_col]].copy()
    for col in ref_features:
        out[col] = df[col] if col in df.columns else 0.0
    out[target] = df[target] if target in df.columns else 0.0
    return out


def add_known_future_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward log returns from close; fill unknown tail with expanding mean."""
    df = df.copy()
    for col, minutes in FUTURE_RETURN_SHIFTS:
        df[col] = np.log(df["close"].shift(-minutes) / df["close"])
    for col, _ in FUTURE_RETURN_SHIFTS:
        df[col] = df[col].fillna(df[col].expanding(min_periods=1).mean())
        df[col] = df[col].fillna(0.0)
    return df


def fit_training_scaler(
    *,
    root_path: str,
    feature_ref_csv: str,
    target: str,
    enc_in: int,
    extra_feature_cols: tuple[str, ...],
) -> StandardScaler:
    """Fit StandardScaler on the first 70% of the aligned training CSV (matches Dataset_Custom)."""
    ref_df = pd.read_csv(Path(root_path) / feature_ref_csv)
    aligned = align_features_to_training(
        ref_df,
        root_path=root_path,
        target=target,
        feature_ref_csv=feature_ref_csv,
        enc_in=enc_in,
        extra_feature_cols=extra_feature_cols,
    )
    time_col = _pick_time_column(aligned)
    feature_cols = [
        col
        for col in aligned.columns
        if col != time_col and pd.api.types.is_numeric_dtype(aligned[col])
    ]
    num_train = int(len(aligned) * 0.7)
    scaler = StandardScaler()
    scaler.fit(aligned.iloc[:num_train][feature_cols].values)
    return scaler


def prepare_predict_csv(
    df: pd.DataFrame,
    *,
    root_path: str,
    target: str,
    feature_ref_csv: str,
    enc_in: int,
    extra_feature_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Backfill known future returns and align columns for inference."""
    if "close" in df.columns:
        df = add_known_future_returns(df)
    return align_features_to_training(
        df,
        root_path=root_path,
        target=target,
        feature_ref_csv=feature_ref_csv,
        enc_in=enc_in,
        extra_feature_cols=extra_feature_cols,
    )


def parse_setting_folder(folder_name: str) -> dict:
    """Parse the training `setting` string encoded in the checkpoint folder name."""
    match = SETTING_RE.search(folder_name)
    if not match:
        raise ValueError(f"Could not parse checkpoint folder name: {folder_name}")

    prefix = folder_name[: match.start()]
    model = next((name for name in MODEL_NAMES if f"_{name}_" in prefix), None)
    if model is None:
        raise ValueError(f"No known model name found in folder: {folder_name}")

    head, _, tail = prefix.partition(f"_{model}_")
    model_id = head
    data = tail.split("_")[0] if tail else "custom"

    parsed = match.groupdict()
    parsed.update(
        {
            "model_id": model_id,
            "model": model,
            "data": data,
            "itr": int(parsed["itr"]),
            "seq_len": int(parsed["seq_len"]),
            "label_len": int(parsed["label_len"]),
            "pred_len": int(parsed["pred_len"]),
            "d_model": int(parsed["d_model"]),
            "n_heads": int(parsed["n_heads"]),
            "e_layers": int(parsed["e_layers"]),
            "d_layers": int(parsed["d_layers"]),
            "d_ff": int(parsed["d_ff"]),
            "factor": int(parsed["factor"]),
            "distil": parsed["distil"] == "True",
        }
    )
    return parsed


def build_args(parsed: dict, checkpoint_path: Path, overrides: dict) -> SimpleNamespace:
    """Build an args namespace compatible with Exp_Main / PatchTST."""
    root_path = overrides.get("root_path", str(REPO_ROOT))
    data_path = overrides.get("data_path", "data/btc_minutes_100000_with_features.csv")
    target = overrides.get("target", "future_log_return_60")

    return SimpleNamespace(
        random_seed=2021,
        is_training=0,
        model_id=parsed["model_id"],
        model=parsed["model"],
        data=parsed["data"],
        root_path=root_path,
        data_path=data_path,
        features=parsed["features"],
        target=target,
        freq=overrides.get("freq", "min"),
        checkpoints=str(checkpoint_path.parent.parent),
        checkpoint_path=str(checkpoint_path),
        seq_len=parsed["seq_len"],
        label_len=parsed["label_len"],
        pred_len=parsed["pred_len"],
        horizon=int(overrides.get("horizon", 0)),
        log_return=overrides.get("log_return", True),
        plot_chunk_size=100,
        plot_chunk_starts="0,1000",
        fc_dropout=float(overrides.get("fc_dropout", 0.05)),
        head_dropout=float(overrides.get("head_dropout", 0.1)),
        patch_len=int(overrides.get("patch_len", DEFAULT_TRAIN_CONFIG["patch_len"])),
        stride=int(overrides.get("stride", DEFAULT_TRAIN_CONFIG["stride"])),
        padding_patch="end",
        revin=int(overrides.get("revin", 1)),
        affine=int(overrides.get("affine", 0)),
        subtract_last=int(overrides.get("subtract_last", 0)),
        decomposition=int(overrides.get("decomposition", 0)),
        kernel_size=int(overrides.get("kernel_size", 25)),
        individual=int(overrides.get("individual", 0)),
        embed_type=0,
        enc_in=int(overrides.get("enc_in", DEFAULT_TRAIN_CONFIG["enc_in"])),
        dec_in=int(overrides.get("dec_in", DEFAULT_TRAIN_CONFIG["enc_in"])),
        c_out=int(overrides.get("c_out", 1)),
        d_model=parsed["d_model"],
        n_heads=parsed["n_heads"],
        e_layers=parsed["e_layers"],
        d_layers=parsed["d_layers"],
        d_ff=parsed["d_ff"],
        moving_avg=25,
        factor=parsed["factor"],
        distil=parsed["distil"],
        dropout=float(overrides.get("dropout", 0.05)),
        embed=parsed["embed"],
        activation="gelu",
        output_attention=False,
        do_predict=False,
        num_workers=0,
        itr=1,
        train_epochs=1,
        batch_size=1,
        patience=5,
        learning_rate=1e-4,
        des=parsed["des"],
        loss="mse",
        lradj="type3",
        pct_start=0.3,
        use_amp=False,
        use_gpu=torch.cuda.is_available(),
        gpu=0,
        use_multi_gpu=False,
        devices="0",
        test_flop=False,
    )


class ForecastModel:
    def __init__(
        self,
        args: SimpleNamespace,
        checkpoint_path: Path,
        split: str = "pred",
        state_dict: dict | None = None,
    ):
        self.args = args
        self.checkpoint_path = checkpoint_path
        self.split = split
        target = args.target
        enc_in = args.enc_in
        feature_ref = getattr(args, "feature_ref_csv", DEFAULT_TRAIN_CONFIG["feature_ref_csv"])
        extra_cols = getattr(args, "extra_feature_cols", DEFAULT_TRAIN_CONFIG["extra_feature_cols"])
        self.raw_df = prepare_predict_csv(
            pd.read_csv(Path(args.root_path) / args.data_path),
            root_path=args.root_path,
            target=target,
            feature_ref_csv=feature_ref,
            enc_in=enc_in,
            extra_feature_cols=extra_cols,
        )
        aligned_path = Path(args.root_path) / "data" / "_predict_aligned_tmp.csv"
        self.raw_df.to_csv(aligned_path, index=False)
        args.data_path = str(aligned_path.relative_to(Path(args.root_path)))
        if split == "pred":
            args.inference_scaler = fit_training_scaler(
                root_path=args.root_path,
                feature_ref_csv=feature_ref,
                target=target,
                enc_in=enc_in,
                extra_feature_cols=extra_cols,
            )
        if len(self.raw_df) < args.seq_len:
            raise ValueError(
                f"CSV has {len(self.raw_df)} rows but the model needs seq_len={args.seq_len}. "
                "Collect more candles (e.g. raise --n_candles in get_btc_price.py)."
            )

        self.exp = Exp_Main(args)
        state = state_dict if state_dict is not None else torch.load(checkpoint_path, map_location=self.exp.device)
        self.exp.model.load_state_dict(state)
        self.exp.model.eval()
        self.dataset, _ = self.exp._get_data(flag=split)
        self.timestamps = self._load_timestamps()

    def _load_timestamps(self) -> pd.Series:
        time_col = _pick_time_column(self.raw_df)
        ts = pd.to_datetime(self.raw_df[time_col], utc=True)
        if self.split == "pred":
            return ts.iloc[-self.args.seq_len :].reset_index(drop=True)
        return ts.iloc[self.dataset.border1 : self.dataset.border2].reset_index(drop=True)

    def _inverse_target(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if hasattr(self.dataset, "inverse_transform_target"):
            return self.dataset.inverse_transform_target(values)

        target_idx = getattr(self.dataset, "target_idx", -1)
        flat = values.reshape(-1)
        scale = self.dataset.scaler.scale_[target_idx]
        mean = self.dataset.scaler.mean_[target_idx]
        return (flat * scale + mean).reshape(values.shape)

    def _last_close(self) -> float | None:
        if "close" not in self.raw_df.columns:
            return None
        return float(self.raw_df["close"].iloc[-1])

    @torch.no_grad()
    def predict(self) -> dict:
        ds_index = len(self.dataset) - 1
        seq_x, seq_y, _, _ = self.dataset[ds_index]

        batch_x = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0).to(self.exp.device)
        outputs = self.exp.model(batch_x)

        f_dim = -1 if self.args.features == "MS" else 0
        pred = outputs[:, -self.args.pred_len :, f_dim:].detach().cpu().numpy()
        pred = self._inverse_target(pred)

        actual = None
        if self.split != "pred" and seq_y is not None and len(seq_y) > self.args.label_len:
            actual_scaled = seq_y[-self.args.pred_len :, f_dim if f_dim >= 0 else -1]
            actual = self._inverse_target(actual_scaled.reshape(1, -1, 1))[0, :, 0]

        last_close = self._last_close()
        if getattr(self.args, "log_return", False) and last_close is not None:
            pred_price = last_close * np.exp(pred[0, 0, 0])
        else:
            pred_price = float(pred[0, 0, 0])

        anchor_time = self.timestamps.iloc[-1]
        return {
            "time": str(anchor_time),
            "prediction": float(pred[0, 0, 0]),
            "predicted_price": float(pred_price),
            "actual": None if actual is None else float(actual[0]),
            "last_close": last_close,
            "checkpoint": str(self.checkpoint_path),
            "data_path": self.args.data_path,
        }


def load_model(model_path: str, **overrides) -> ForecastModel:
    checkpoint_path = _resolve_checkpoint_path(model_path)
    folder_name = checkpoint_path.parent.name
    print("loading model", folder_name)

    parsed = parse_setting_folder(folder_name)
    state = torch.load(checkpoint_path, map_location="cpu")
    inferred = infer_patchtst_params(state, parsed["seq_len"])

    if overrides.get("patch_len") is None:
        overrides["patch_len"] = inferred["patch_len"]
    if overrides.get("stride") is None:
        overrides["stride"] = inferred["stride"]
    if inferred["seq_len"] != parsed["seq_len"]:
        print(
            f"note: checkpoint matches seq_len={inferred['seq_len']} "
            f"(folder says {parsed['seq_len']})"
        )
        parsed["seq_len"] = inferred["seq_len"]

    root_path = overrides.get("root_path", str(REPO_ROOT))
    data_path = overrides.get("data_path", "data/btc_minutes_100000_with_features.csv")
    target = overrides.get("target", DEFAULT_TRAIN_CONFIG["target"])
    if overrides.get("enc_in") is None:
        overrides["enc_in"] = DEFAULT_TRAIN_CONFIG["enc_in"]
    if overrides.get("dec_in") is None:
        overrides["dec_in"] = overrides["enc_in"]
    if overrides.get("patch_len") is None:
        overrides["patch_len"] = DEFAULT_TRAIN_CONFIG["patch_len"]
    if overrides.get("stride") is None:
        overrides["stride"] = DEFAULT_TRAIN_CONFIG["stride"]
    if overrides.get("horizon") is None:
        overrides["horizon"] = DEFAULT_TRAIN_CONFIG["horizon"]

    if overrides.get("feature_ref_csv") is None:
        overrides["feature_ref_csv"] = DEFAULT_TRAIN_CONFIG["feature_ref_csv"]
    if overrides.get("extra_feature_cols") is None:
        overrides["extra_feature_cols"] = DEFAULT_TRAIN_CONFIG["extra_feature_cols"]

    args = build_args(parsed, checkpoint_path, overrides)
    args.feature_ref_csv = overrides["feature_ref_csv"]
    args.extra_feature_cols = overrides["extra_feature_cols"]
    split = overrides.get("split", "pred")
    return ForecastModel(args, checkpoint_path, split=split, state_dict=state)


if __name__ == "__main__":
    default_model = (
        "best_model/BTCM_480_1_PatchTST_custom_ftMS_sl480_ll1_pl1_dm128_nh4_el4_dl1"
        "_df512_fc1_ebtimeF_dtTrue_BTCM_240to1_mse_more_features_and_log_return_60_0"
    )

    parser = argparse.ArgumentParser(description="Run one PatchTST prediction from a CSV")
    parser.add_argument("--model_path", type=str, default=default_model, help="checkpoint.pth or its folder")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/btc_minutes_2026-06-24_02-00-00_550_predict.csv",
    )
    parser.add_argument("--target", type=str, default="future_log_return_60")
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--enc_in", type=int, default=None)
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--split", type=str, default="pred", choices=["train", "val", "test", "pred"])
    cli = parser.parse_args()

    model = load_model(
        cli.model_path,
        data_path=cli.data_path,
        target=cli.target,
        horizon=cli.horizon,
        enc_in=cli.enc_in,
        dec_in=cli.enc_in,
        patch_len=cli.patch_len,
        stride=cli.stride,
        split=cli.split,
    )
    result = model.predict()
    print(result)
