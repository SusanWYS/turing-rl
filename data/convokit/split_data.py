#!/usr/bin/env python3
"""Split ConvoKit rows by user."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path("data/convokit/q35_convokit_raw_history_persona_s42")
DEFAULT_OUTPUT_DIR = Path(
    "data/convokit/q35_convokit_raw_history_persona_s42_sft40_grpo60_tifu_worldnews_test102"
)
DEFAULT_HELDOUT_SUBREDDITS = ("r/tifu", "r/worldnews")
DEFAULT_GRPO_VAL_FRAC = 0.3


def _extra_info_value(row: pd.Series, key: str) -> str:
    extra = row["extra_info"]
    if not isinstance(extra, dict):
        raise TypeError(f"extra_info must be a dict, got {type(extra)!r}")
    return str(extra.get(key))


def _update_extra_info(row: pd.Series, *, split: str) -> dict[str, Any]:
    extra = dict(row["extra_info"])
    extra["split"] = split
    return extra


def _extra_info_column(df: pd.DataFrame, *, split: str) -> pd.Series:
    if df.empty:
        return pd.Series(index=df.index, dtype=object)
    return df.apply(_update_extra_info, axis=1, split=split)


def _sort_rows_like_source_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    split_rank = {"train": 0, "val": 1, "test": 2}
    sorted_df = df.copy()
    sorted_df["_split_rank"] = sorted_df["_orig_split"].map(lambda value: split_rank.get(str(value), 99))
    sorted_df["_row_order"] = sorted_df["extra_info"].map(
        lambda extra: int(extra.get("index", extra.get("prompt_idx", 0))) if isinstance(extra, dict) else 0
    )
    return sorted_df.sort_values(
        ["_split_rank", "_row_order", "_target_subreddit", "_user_id"], kind="stable"
    ).drop(columns=["_split_rank", "_row_order"])


def _split_train_val_like_convokit_pipeline(
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
        user_df = _sort_rows_like_source_pipeline(user_df)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--heldout-subreddits", nargs="+", default=list(DEFAULT_HELDOUT_SUBREDDITS))
    parser.add_argument("--grpo-frac", type=float, default=0.6)
    parser.add_argument(
        "--grpo-val-frac",
        type=float,
        default=DEFAULT_GRPO_VAL_FRAC,
        help="Per-user GRPO validation fraction.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.grpo_frac < 1.0:
        raise ValueError("--grpo-frac must be in (0, 1)")

    frames = []
    for split in ("train", "val", "test"):
        parquet_path = args.input_dir / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        df = pd.read_parquet(parquet_path)
        df = df.copy()
        df["_orig_split"] = split
        df["_user_id"] = df.apply(lambda row: _extra_info_value(row, "user_id"), axis=1)
        df["_target_subreddit"] = df.apply(lambda row: _extra_info_value(row, "target_subreddit"), axis=1)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    heldout_subreddits = set(args.heldout_subreddits)
    original_train = all_df["_orig_split"] == "train"
    heldout_subreddit_rows = all_df["_target_subreddit"].isin(heldout_subreddits)
    heldout_users = set(all_df.loc[original_train & heldout_subreddit_rows, "_user_id"])
    heldout_df = all_df.loc[original_train & heldout_subreddit_rows & all_df["_user_id"].isin(heldout_users)].copy()

    remaining_df = all_df.loc[
        ~all_df["_user_id"].isin(heldout_users) & ~all_df["_target_subreddit"].isin(heldout_subreddits)
    ].copy()
    remaining_users = sorted(set(remaining_df["_user_id"]))
    rng = random.Random(args.seed)
    rng.shuffle(remaining_users)
    grpo_user_count = round(len(remaining_users) * args.grpo_frac)
    grpo_users = set(remaining_users[:grpo_user_count])
    sft_users = set(remaining_users[grpo_user_count:])

    grpo_df = remaining_df.loc[remaining_df["_user_id"].isin(grpo_users)].copy()
    sft_df = remaining_df.loc[remaining_df["_user_id"].isin(sft_users)].copy()
    grpo_train_df, grpo_val_df, grpo_dropped_df = _split_train_val_like_convokit_pipeline(
        grpo_df,
        val_frac=args.grpo_val_frac,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    passthrough_cols = [c for c in all_df.columns if not c.startswith("_")]
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
        "grpo_frac": args.grpo_frac,
        "grpo_val_frac": args.grpo_val_frac,
        "heldout_subreddits": sorted(heldout_subreddits),
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


if __name__ == "__main__":
    main()
