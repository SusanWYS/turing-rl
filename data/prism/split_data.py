#!/usr/bin/env python3
"""Split PRISM rows by user."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("data/prism/mixed_allu_r4_mixed_prism_alignment_history_persona_s42")
DEFAULT_OUTPUT_DIR = Path(
    "data/prism/mixed_allu_r4_mixed_prism_alignment_history_persona_s42_sft40_grpo60_test10"
)
DEFAULT_HELDOUT_USER_FRAC = 0.1
DEFAULT_GRPO_FRAC = 0.6
DEFAULT_GRPO_VAL_FRAC = 0.1
DEFAULT_SEED = 42


def _extra_info_value(row: pd.Series, key: str) -> Any:
    extra = row["extra_info"]
    if not isinstance(extra, dict):
        raise TypeError(f"extra_info must be a dict, got {type(extra)!r}")
    return extra.get(key)


def _update_extra_info(row: pd.Series, *, split: str) -> dict[str, Any]:
    extra = dict(row["extra_info"])
    extra["split"] = split
    return extra


def _extra_info_column(df: pd.DataFrame, *, split: str) -> pd.Series:
    if df.empty:
        return pd.Series(index=df.index, dtype=object)
    return df.apply(_update_extra_info, axis=1, split=split)


def _ordered_user_ids(df: pd.DataFrame) -> list[str]:
    """Return users in row order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for user_id in df["_user_id"]:
        if user_id in seen:
            continue
        seen.add(user_id)
        ordered.append(user_id)
    return ordered


def _sort_rows_for_prism_tail_split(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows for per-user tail validation."""
    if df.empty:
        return df.copy()
    sorted_df = df.copy()
    sorted_df["_post_id_sort"] = sorted_df["extra_info"].map(
        lambda extra: str(extra.get("post_id", "")) if isinstance(extra, dict) else ""
    )
    sorted_df["_target_idx_sort"] = sorted_df["extra_info"].map(
        lambda extra: int(extra.get("target_idx", 0)) if isinstance(extra, dict) else 0
    )
    sorted_df["_source_index_sort"] = sorted_df["extra_info"].map(
        lambda extra: int(extra.get("index", extra.get("prompt_idx", 0))) if isinstance(extra, dict) else 0
    )
    sorted_df["_orig_split_rank"] = sorted_df["_orig_split"].map(
        lambda value: {"train": 0, "val": 1, "test": 2}.get(str(value), 99)
    )
    return sorted_df.sort_values(
        [
            "_post_id_sort",
            "_target_idx_sort",
            "_orig_split_rank",
            "_source_index_sort",
        ],
        kind="stable",
    ).drop(columns=["_post_id_sort", "_target_idx_sort", "_source_index_sort", "_orig_split_rank"])


def _split_train_val_by_user_tail(
    df: pd.DataFrame,
    *,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be >= 0 and < 1, got {val_frac}")
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    dropped_parts: list[pd.DataFrame] = []
    for _user_id, user_df in df.groupby("_user_id", sort=True):
        user_df = _sort_rows_for_prism_tail_split(user_df)
        target_count = len(user_df)
        if target_count <= 1:
            dropped_parts.append(user_df)
            continue
        val_count = int(target_count * val_frac)
        if val_frac > 0.0 and target_count > 1:
            val_count = max(val_count, 1)
        if val_count >= target_count:
            val_count = target_count - 1
        if val_count:
            train_parts.append(user_df.iloc[:-val_count])
            val_parts.append(user_df.iloc[-val_count:])
        else:
            train_parts.append(user_df)
    columns = list(df.columns)
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=columns)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=columns)
    dropped_df = pd.concat(dropped_parts, ignore_index=True) if dropped_parts else pd.DataFrame(columns=columns)
    return train_df, val_df, dropped_df


def _write_role_dir(
    role_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    role_dir.mkdir(parents=True, exist_ok=True)
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()
    train["extra_info"] = _extra_info_column(train, split="train")
    val["extra_info"] = _extra_info_column(val, split="val")
    test["extra_info"] = _extra_info_column(test, split="test")
    train.to_parquet(role_dir / "train.parquet", index=False)
    val.to_parquet(role_dir / "val.parquet", index=False)
    test.to_parquet(role_dir / "test.parquet", index=False)


def _write_sft_dir(role_dir: Path, train_df: pd.DataFrame) -> None:
    role_dir.mkdir(parents=True, exist_ok=True)
    train = train_df.copy()
    train["extra_info"] = _extra_info_column(train, split="train")
    train.to_parquet(role_dir / "train.parquet", index=False)


def _summarize(df: pd.DataFrame) -> dict[str, Any]:
    users = {_extra_info_value(row, "user_id") for _, row in df.iterrows()}
    return {"rows": int(len(df)), "users": len(users)}


def split_data(args: argparse.Namespace) -> None:
    if not 0.0 < args.heldout_user_frac < 1.0:
        raise ValueError("--heldout-user-frac must be in (0, 1)")
    if not 0.0 < args.grpo_frac < 1.0:
        raise ValueError("--grpo-frac must be in (0, 1)")

    frames = []
    for split in ("train", "val", "test"):
        parquet_path = args.input_dir / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        df = pd.read_parquet(parquet_path).copy()
        df["_orig_split"] = split
        df["_user_id"] = df.apply(lambda row: str(_extra_info_value(row, "user_id")), axis=1)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    ordered_users = _ordered_user_ids(all_df)
    shuffled_users = ordered_users[:]
    random.Random(args.seed).shuffle(shuffled_users)
    heldout_user_count = int(len(shuffled_users) * args.heldout_user_frac)
    heldout_users = set(shuffled_users[:heldout_user_count])
    remaining_users = shuffled_users[heldout_user_count:]

    remaining_users_for_split = remaining_users[:]
    random.Random(args.seed).shuffle(remaining_users_for_split)
    grpo_user_count = int(len(remaining_users_for_split) * args.grpo_frac)
    grpo_users = set(remaining_users_for_split[:grpo_user_count])
    sft_users = set(remaining_users_for_split[grpo_user_count:])

    heldout_df = all_df.loc[all_df["_user_id"].isin(heldout_users)].copy()
    grpo_df = all_df.loc[all_df["_user_id"].isin(grpo_users)].copy()
    sft_df = all_df.loc[all_df["_user_id"].isin(sft_users)].copy()
    grpo_train_df, grpo_val_df, grpo_dropped_df = _split_train_val_by_user_tail(
        grpo_df,
        val_frac=args.grpo_val_frac,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    passthrough_cols = [column for column in all_df.columns if not column.startswith("_")]
    _write_role_dir(
        output_dir / "grpo",
        grpo_train_df[passthrough_cols],
        grpo_val_df[passthrough_cols],
        heldout_df[passthrough_cols],
    )
    _write_sft_dir(output_dir / "sft", sft_df[passthrough_cols])
    heldout_only = heldout_df[passthrough_cols].copy()
    heldout_only["extra_info"] = _extra_info_column(heldout_only, split="test")
    heldout_only.to_parquet(output_dir / "test.parquet", index=False)

    metadata = {
        "source_input_dir": str(args.input_dir),
        "seed": args.seed,
        "heldout_user_frac": args.heldout_user_frac,
        "grpo_frac": args.grpo_frac,
        "grpo_val_frac": args.grpo_val_frac,
        "counts": {
            "sft": _summarize(sft_df),
            "grpo_train": _summarize(grpo_train_df),
            "grpo_val": _summarize(grpo_val_df),
            "heldout": _summarize(heldout_df),
        },
        "user_overlap": {
            "grpo_sft": len(grpo_users & sft_users),
            "grpo_heldout": len(grpo_users & heldout_users),
            "sft_heldout": len(sft_users & heldout_users),
        },
    }
    with (output_dir / "split_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(metadata, indent=2, sort_keys=True))


def parse_split_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--heldout-user-frac", type=float, default=DEFAULT_HELDOUT_USER_FRAC)
    parser.add_argument("--grpo-frac", type=float, default=DEFAULT_GRPO_FRAC)
    parser.add_argument(
        "--grpo-val-frac",
        type=float,
        default=DEFAULT_GRPO_VAL_FRAC,
        help="Per-user GRPO validation fraction. Validation is selected from later target turns.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main() -> None:
    split_data(parse_split_args())


if __name__ == "__main__":
    main()
