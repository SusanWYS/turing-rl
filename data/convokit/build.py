"""Build ConvoKit GRPO data."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import tempfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.prompt_utils import (
    CONDITIONING_MODE_CHOICES,
    CONDITIONING_MODE_HISTORY,
    build_grpo_prompt_payload,
    conditioning_mode_uses_persona,
)
from shared.load_personas import get_persona_for_user, load_persona_map
from shared.model_ids import DEFAULT_MODEL_ID, load_tokenizer
from data.convokit.build_history import (
    DEFAULT_MIN_CONVERSATIONS,
    DEFAULT_SOURCE_NAME,
    DEFAULT_THREAD_COMMENT_RETENTION,
    THREAD_COMMENT_RETENTION_LAST_TARGET_ANCESTOR_CHAIN,
    anonymize_user_profiles,
    normalize_subreddit_name,
    normalize_subreddit_corpus_name,
    profiles_from_convokit_corpus_zip,
    profiles_from_convokit_corpus_zips,
    profiles_from_convokit_corpora,
)
from data.utils import (
    PUBLIC_DATA_MODES,
    Thread,
    UserProfile,
    normalize_reddit_reply_artifacts,
    shuffle_parquet_rows,
)


MODEL_ID = DEFAULT_MODEL_ID
ALL_DATA_MODES = set(PUBLIC_DATA_MODES)
DEFAULT_OUTPUT = "data/convokit/subreddit_history_train.parquet"
DEFAULT_RANDOM_HISTORY_COUNT_MIN = 2
DEFAULT_RANDOM_HISTORY_COUNT_MAX = 6
DEFAULT_HISTORY_COUNT_SEED = 42
DEFAULT_VAL_FRAC = 0.1
DEFAULT_DATA_SOURCE = "reddit_user_sim_mixed"
DEFAULT_CORPUS_ZIP_DIR = "data/convokit/subreddit_corpora"
DEFAULT_CORPORA_MANIFEST = Path(__file__).with_name("corpora_urls.json")
DEFAULT_TRAIN_SUBREDDITS = [
    "AmItheAsshole",
    "AskMen",
    "AskWomen",
    "business",
    "changemyview",
    "Frugal",
    "news",
    "relationship_advice",
    "tifu",
    "worldnews",
]
DEFAULT_TEST_SUBREDDITS = [
    "Economics",
    "TrueReddit",
    "relationships",
    "MaliciousCompliance",
]


@dataclass(frozen=True)
class ProfileSelectionResult:
    profiles: list[UserProfile]
    total_profiles: int
    excluded_overlap_users: int
    user_offset: int
    max_users: int | None


@dataclass(frozen=True)
class ProfileRowBuildResult:
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    skipped_users: int


@dataclass(frozen=True)
class ProfileTestRowBuildResult:
    test_rows: list[dict[str, Any]]
    skipped_users: int


@dataclass(frozen=True)
class HistoryCountSpec:
    random_min: int = DEFAULT_RANDOM_HISTORY_COUNT_MIN
    random_max: int = DEFAULT_RANDOM_HISTORY_COUNT_MAX
    random_seed: int = DEFAULT_HISTORY_COUNT_SEED

    def validate(self) -> None:
        if self.random_min < 1:
            raise ValueError(f"random history min must be >= 1, got {self.random_min}")
        if self.random_max < self.random_min:
            raise ValueError(
                f"random history max must be >= min, got min={self.random_min} max={self.random_max}"
            )

    def resolve_for_profile(self, profile: UserProfile) -> int:
        self.validate()
        user_key = profile.raw_user_id or profile.profile_id or profile.user_id
        digest = hashlib.sha256(f"{self.random_seed}:{user_key}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return self.random_min + bucket % (self.random_max - self.random_min + 1)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "random_min": self.random_min,
            "random_max": self.random_max,
            "random_seed": self.random_seed,
        }


class _ParquetChunkWriter:
    def __init__(self, output_path: str, *, chunk_size: int = 1000):
        self.output_path = output_path
        self.chunk_size = chunk_size
        self._buffer: list[dict[str, Any]] = []
        self._writer = None
        self.count = 0

    def write(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self._buffer)
        if self._writer is None:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(self.output_path, table.schema)
        else:
            table = table.cast(self._writer.schema)
        self._writer.write_table(table)
        self.count += len(self._buffer)
        self._buffer = []

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "_ParquetChunkWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


_PARALLEL_ROW_BUILD_CONFIG: dict[str, Any] | None = None
_PARALLEL_ROW_BUILD_TOKENIZER: Any = None


def _default_num_workers() -> int:
    for env_name in ("GRPO_DATA_NUM_WORKERS", "PERSONA_GRPO_DATA_NUM_WORKERS", "SLURM_CPUS_PER_TASK"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(1, int(raw_value))
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc
    return max(1, os.cpu_count() or 1)


def _resolve_num_workers(requested: int | None, *, num_profiles: int) -> int:
    if requested is None or requested <= 0:
        requested = _default_num_workers()
    return max(1, min(int(requested), max(1, num_profiles)))


def _prefetch_profile_limit(
    *,
    user_offset: int,
    max_users: int | None,
    extra_users: int = 0,
) -> int | None:
    if max_users is None:
        return None
    return max(0, int(user_offset)) + int(max_users) + max(0, int(extra_users))


def _resolve_persona_for_profile(
    *,
    profile: UserProfile,
    conditioning_mode: str,
    persona_map: dict[str, str],
) -> str:
    if not conditioning_mode_uses_persona(conditioning_mode):
        return ""
    persona = get_persona_for_user(persona_map, profile.user_id, profile.raw_user_id)
    if persona:
        return persona
    raise ValueError(
        f"Missing persona for user_id={profile.user_id} while conditioning_mode={conditioning_mode!r}"
    )


def _configure_parallel_row_build_parent_state(*, config: dict[str, Any], tokenizer: Any) -> None:
    global _PARALLEL_ROW_BUILD_CONFIG, _PARALLEL_ROW_BUILD_TOKENIZER
    _PARALLEL_ROW_BUILD_CONFIG = dict(config)
    _PARALLEL_ROW_BUILD_TOKENIZER = tokenizer


def _init_parallel_row_build_worker(config: dict[str, Any]) -> None:
    global _PARALLEL_ROW_BUILD_CONFIG, _PARALLEL_ROW_BUILD_TOKENIZER
    _PARALLEL_ROW_BUILD_CONFIG = dict(config)
    if _PARALLEL_ROW_BUILD_TOKENIZER is None:
        _PARALLEL_ROW_BUILD_TOKENIZER = load_tokenizer(str(config["tokenizer_name"]))


def _parallel_row_build_state() -> tuple[dict[str, Any], Any]:
    if _PARALLEL_ROW_BUILD_CONFIG is None or _PARALLEL_ROW_BUILD_TOKENIZER is None:
        raise RuntimeError("Parallel GRPO row-build state was not initialized")
    return _PARALLEL_ROW_BUILD_CONFIG, _PARALLEL_ROW_BUILD_TOKENIZER


def _mp_context_for_row_build() -> Any:
    return mp.get_context("fork")


def _strip_human_suffix(context: str) -> str:
    context = context.rstrip()
    if context.endswith("[HUMAN]: "):
        return context[:-len("[HUMAN]: ")].rstrip()
    if context.endswith("[HUMAN]:"):
        return context[:-len("[HUMAN]:")].rstrip()
    return context


def _validate_data_mode(mode: str) -> None:
    if mode not in ALL_DATA_MODES:
        raise ValueError(f"Unknown mode: {mode}")


def _normalize_subreddit_list(names: Iterable[str] | None) -> list[str]:
    if names is None:
        return []
    normalized = []
    seen = set()
    for name in names:
        subreddit = normalize_subreddit_name(name)
        key = subreddit.lower()
        if not subreddit or key in seen:
            continue
        normalized.append(subreddit)
        seen.add(key)
    return normalized


def _corpus_zip_paths_for_subreddits(zip_dir: str | Path, subreddits: Iterable[str]) -> list[Path]:
    root = Path(zip_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"ConvoKit corpus zip directory not found: {root}")

    available = {path.name.lower(): path for path in root.glob("*.corpus.zip")}
    paths: list[Path] = []
    missing: list[str] = []
    for subreddit in _normalize_subreddit_list(subreddits):
        expected_name = f"{subreddit}.corpus.zip"
        exact_path = root / expected_name
        if exact_path.is_file():
            paths.append(exact_path)
            continue
        matched_path = available.get(expected_name.lower())
        if matched_path is not None:
            paths.append(matched_path)
            continue
        missing.append(str(exact_path))

    if missing:
        missing_preview = "\n  ".join(missing[:20])
        raise FileNotFoundError(
            "Missing official ConvoKit per-subreddit corpus zips:\n"
            f"  {missing_preview}"
        )
    return paths


def _slice_profiles(
    profiles: Iterable[UserProfile],
    *,
    user_offset: int = 0,
    max_users: int | None = None,
    exclude_user_ids: set[str] | None = None,
) -> ProfileSelectionResult:
    if user_offset < 0:
        raise ValueError(f"user_offset must be >= 0, got {user_offset}")
    if max_users is not None and max_users < 1:
        raise ValueError(f"max_users must be >= 1 when set, got {max_users}")

    ordered_profiles = list(profiles)
    total_profiles = len(ordered_profiles)
    excluded = exclude_user_ids or set()
    filtered_profiles = [profile for profile in ordered_profiles if profile.user_id not in excluded]
    excluded_overlap_users = total_profiles - len(filtered_profiles)

    if user_offset > len(filtered_profiles):
        raise ValueError(
            f"user_offset {user_offset} exceeds available profiles after exclusions ({len(filtered_profiles)})"
        )
    filtered_profiles = filtered_profiles[user_offset:]
    if max_users is not None:
        filtered_profiles = filtered_profiles[:max_users]
    return ProfileSelectionResult(
        profiles=filtered_profiles,
        total_profiles=total_profiles,
        excluded_overlap_users=excluded_overlap_users,
        user_offset=user_offset,
        max_users=max_users,
    )


def _history_and_target_threads(
    profile: UserProfile,
    *,
    random_history_count_min: int = DEFAULT_RANDOM_HISTORY_COUNT_MIN,
    random_history_count_max: int = DEFAULT_RANDOM_HISTORY_COUNT_MAX,
    history_count_seed: int = DEFAULT_HISTORY_COUNT_SEED,
    val_frac: float = DEFAULT_VAL_FRAC,
    max_targets_per_user: int | None = None,
) -> tuple[list[Thread], list[Thread], list[Thread]]:
    history_spec = HistoryCountSpec(
        random_min=random_history_count_min,
        random_max=random_history_count_max,
        random_seed=history_count_seed,
    )
    resolved_history_count = history_spec.resolve_for_profile(profile)
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be >= 0 and < 1, got {val_frac}")

    threads = list(profile.train_threads)
    if len(threads) <= resolved_history_count + 1:
        return [], [], []

    history_threads = threads[:resolved_history_count]
    target_threads = threads[resolved_history_count:]
    if max_targets_per_user is not None:
        if max_targets_per_user < 1:
            raise ValueError(f"max_targets_per_user must be >= 1, got {max_targets_per_user}")
        target_threads = target_threads[:max_targets_per_user]

    if not target_threads:
        return [], [], []

    val_count = int(len(target_threads) * val_frac)
    if val_frac > 0.0 and len(target_threads) > 1:
        val_count = max(val_count, 1)
    if val_count >= len(target_threads):
        val_count = len(target_threads) - 1

    if val_count:
        return history_threads, target_threads[:-val_count], target_threads[-val_count:]
    return history_threads, target_threads, []


def _iter_entries_for_threads(
    *,
    prompt_idx_start: int,
    profile: UserProfile,
    history_threads: list[Thread],
    target_threads: list[Thread],
    split_name: str,
    corpus_names: list[str],
    persona: str = "",
) -> Iterator[dict[str, Any]]:
    prompt_idx = prompt_idx_start
    user_history = profile.format_history_context(history_threads)
    history_subreddits = sorted({thread.source_label() for thread in history_threads if thread.source_label()})

    for thread in target_threads:
        if not thread.target_user_comment_indices:
            continue
        target_idx = thread.target_user_comment_indices[-1]
        ground_truth = normalize_reddit_reply_artifacts(thread.comments[target_idx].text)
        if not ground_truth:
            continue
        current_context = _strip_human_suffix(thread.format_for_eval(target_idx))
        yield {
            "prompt_idx": prompt_idx,
            "user_id": profile.user_id,
            "raw_user_id": profile.raw_user_id,
            "source_name": profile.source_name or DEFAULT_SOURCE_NAME,
            "post_id": thread.post_id,
            "target_idx": target_idx,
            "ground_truth": ground_truth,
            "user_history": user_history,
            "persona": persona,
            "current_context": current_context,
            "split": split_name,
            "history_thread_ids": [history_thread.post_id for history_thread in history_threads],
            "history_count": len(history_threads),
            "history_subreddits": history_subreddits,
            "target_subreddit": thread.source_label(),
            "convokit_corpus_names": corpus_names,
        }
        prompt_idx += 1


def build_grpo_row(
    entry: dict[str, Any],
    tokenizer: Any,
    *,
    data_source: str = DEFAULT_DATA_SOURCE,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
) -> dict[str, Any]:
    """Build one GRPO row."""
    persona = str(entry.get("persona", "") or "").strip()
    prompt_payload = build_grpo_prompt_payload(
        tokenizer,
        user_history=entry["user_history"],
        thread_context=entry["current_context"],
        prompt_mode="reasoning",
        persona=persona,
        conditioning_mode=conditioning_mode,
    )
    return {
        "data_source": data_source,
        "prompt": prompt_payload["prompt"],
        "reward_model": {
            "style": "rule",
            "ground_truth": entry["ground_truth"],
        },
        "extra_info": {
            "user_id": entry["user_id"],
            "raw_user_id": entry.get("raw_user_id", entry["user_id"]),
            "source_name": entry.get("source_name", ""),
            "user_history": entry["user_history"],
            "persona": persona,
            "post_id": entry["post_id"],
            "target_idx": entry["target_idx"],
            "context": entry["current_context"],
            "thread_context": entry["current_context"],
            "prompt_text": prompt_payload["prompt_text"],
            "prompt_mode": prompt_payload["prompt_mode"],
            "conditioning_mode": prompt_payload["conditioning_mode"],
            "split": entry.get("split", ""),
            "prompt_idx": entry["prompt_idx"],
            "index": entry["prompt_idx"],
            "raw_prompt": prompt_payload["raw_prompt"],
            "history_thread_ids": entry.get("history_thread_ids", []),
            "history_count": entry.get("history_count", len(entry.get("history_thread_ids", []))),
            "history_subreddits": entry.get("history_subreddits", []),
            "target_subreddit": entry.get("target_subreddit", ""),
            "convokit_corpus_names": entry.get("convokit_corpus_names", []),
        },
    }


def _build_profile_rows_for_target_threads(
    *,
    profile: UserProfile,
    history_threads: list[Thread],
    target_threads: list[Thread],
    split_name: str,
    corpus_names: list[str],
    persona: str,
    tokenizer: Any,
    data_source: str,
    conditioning_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in _iter_entries_for_threads(
        prompt_idx_start=0,
        profile=profile,
        history_threads=history_threads,
        target_threads=target_threads,
        split_name=split_name,
        corpus_names=corpus_names,
        persona=persona,
    ):
        rows.append(
            build_grpo_row(
                entry,
                tokenizer,
                data_source=data_source,
                conditioning_mode=conditioning_mode,
            )
        )
    return rows


def _build_profile_train_val_rows(profile: UserProfile) -> ProfileRowBuildResult:
    config, tokenizer = _parallel_row_build_state()
    history_threads, train_threads, val_threads = _history_and_target_threads(
        profile,
        random_history_count_min=int(config["random_history_count_min"]),
        random_history_count_max=int(config["random_history_count_max"]),
        history_count_seed=int(config["history_count_seed"]),
        val_frac=float(config["val_frac"]),
        max_targets_per_user=config["max_targets_per_user"],
    )
    if not history_threads or not train_threads:
        return ProfileRowBuildResult(train_rows=[], val_rows=[], skipped_users=1)

    conditioning_mode = str(config["conditioning_mode"])
    persona = _resolve_persona_for_profile(
        profile=profile,
        conditioning_mode=conditioning_mode,
        persona_map=config["persona_map"],
    )

    train_rows = _build_profile_rows_for_target_threads(
        profile=profile,
        history_threads=history_threads,
        target_threads=train_threads,
        split_name="train",
        corpus_names=config["corpus_names"],
        persona=persona,
        tokenizer=tokenizer,
        data_source=str(config["data_source"]),
        conditioning_mode=conditioning_mode,
    )
    val_rows = _build_profile_rows_for_target_threads(
        profile=profile,
        history_threads=history_threads,
        target_threads=val_threads,
        split_name="val",
        corpus_names=config["corpus_names"],
        persona=persona,
        tokenizer=tokenizer,
        data_source=str(config["data_source"]),
        conditioning_mode=conditioning_mode,
    )
    return ProfileRowBuildResult(train_rows=train_rows, val_rows=val_rows, skipped_users=0)


def _build_profile_test_rows(profile: UserProfile) -> ProfileTestRowBuildResult:
    config, tokenizer = _parallel_row_build_state()
    history_threads, test_threads, _ = _history_and_target_threads(
        profile,
        random_history_count_min=int(config["random_history_count_min"]),
        random_history_count_max=int(config["random_history_count_max"]),
        history_count_seed=int(config["history_count_seed"]),
        val_frac=0.0,
        max_targets_per_user=config["max_targets_per_user"],
    )
    if not history_threads or not test_threads:
        return ProfileTestRowBuildResult(test_rows=[], skipped_users=1)

    conditioning_mode = str(config["conditioning_mode"])
    persona = _resolve_persona_for_profile(
        profile=profile,
        conditioning_mode=conditioning_mode,
        persona_map=config["persona_map"],
    )

    test_rows = _build_profile_rows_for_target_threads(
        profile=profile,
        history_threads=history_threads,
        target_threads=test_threads,
        split_name="test",
        corpus_names=config["corpus_names"],
        persona=persona,
        tokenizer=tokenizer,
        data_source=str(config["data_source"]),
        conditioning_mode=conditioning_mode,
    )
    return ProfileTestRowBuildResult(test_rows=test_rows, skipped_users=0)


def _iter_parallel_profile_results(
    *,
    profiles: list[UserProfile],
    worker_fn: Any,
    num_workers: int,
    config: dict[str, Any],
    tokenizer: Any,
) -> Iterator[Any]:
    _configure_parallel_row_build_parent_state(config=config, tokenizer=tokenizer)
    if num_workers <= 1:
        for profile in profiles:
            yield worker_fn(profile)
        return

    chunksize = max(1, len(profiles) // (num_workers * 4))
    executor_kwargs: dict[str, Any] = {
        "max_workers": num_workers,
        "initializer": _init_parallel_row_build_worker,
        "initargs": (config,),
    }
    mp_context = _mp_context_for_row_build()
    if mp_context is not None:
        executor_kwargs["mp_context"] = mp_context
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        yield from executor.map(worker_fn, profiles, chunksize=chunksize)


def _assign_prompt_indices(rows: list[dict[str, Any]], *, start_idx: int) -> int:
    next_idx = start_idx
    for row in rows:
        extra_info = row.get("extra_info", {})
        extra_info["prompt_idx"] = next_idx
        extra_info["index"] = next_idx
        next_idx += 1
    return next_idx


def _test_row_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, int]:
    extra_info = row.get("extra_info", {})
    target_subreddit = str(extra_info.get("target_subreddit", "") or "").strip()
    source_name = str(extra_info.get("source_name", "") or "").strip()
    user_id = str(extra_info.get("user_id", "") or "")
    post_id = str(extra_info.get("post_id", "") or "")
    target_idx = int(extra_info.get("target_idx", -1))
    return (
        target_subreddit.lower(),
        target_subreddit,
        source_name.lower(),
        user_id,
        post_id,
        target_idx,
    )


def write_grpo_rows_from_profiles(
    profiles: Iterable[UserProfile],
    *,
    tokenizer: Any,
    mode: str,
    output: str,
    val_output: str,
    data_source: str = DEFAULT_DATA_SOURCE,
    random_history_count_min: int = DEFAULT_RANDOM_HISTORY_COUNT_MIN,
    random_history_count_max: int = DEFAULT_RANDOM_HISTORY_COUNT_MAX,
    history_count_seed: int = DEFAULT_HISTORY_COUNT_SEED,
    val_frac: float = DEFAULT_VAL_FRAC,
    max_targets_per_user: int | None = None,
    corpus_names: Iterable[str] | None = None,
    chunk_size: int = 1000,
    shuffle_rows: bool = False,
    shuffle_seed: int = 42,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
    persona_map: dict[str, str] | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Write train/val parquet rows."""
    ordered_profiles = list(profiles)
    _validate_data_mode(mode)
    persona_map = persona_map or {}
    normalized_corpus_names = [
        normalize_subreddit_corpus_name(name)
        if not str(name).endswith(".zip") else str(name)
        for name in (list(corpus_names) if corpus_names is not None else [])
    ]
    resolved_num_workers = _resolve_num_workers(num_workers, num_profiles=len(ordered_profiles))
    print(
        f"Building train/val GRPO rows with {resolved_num_workers} worker(s) across {len(ordered_profiles)} profile(s)",
        flush=True,
    )
    skipped_users = 0
    prompt_idx = 0
    train_users: set[str] = set()
    val_users: set[str] = set()
    worker_config = {
        "random_history_count_min": random_history_count_min,
        "random_history_count_max": random_history_count_max,
        "history_count_seed": history_count_seed,
        "val_frac": val_frac,
        "max_targets_per_user": max_targets_per_user,
        "conditioning_mode": conditioning_mode,
        "persona_map": persona_map,
        "corpus_names": normalized_corpus_names,
        "data_source": data_source,
        "tokenizer_name": getattr(tokenizer, "name_or_path", MODEL_ID),
    }

    with _ParquetChunkWriter(output, chunk_size=chunk_size) as train_writer, _ParquetChunkWriter(
        val_output, chunk_size=chunk_size
    ) as val_writer:
        for result in _iter_parallel_profile_results(
            profiles=ordered_profiles,
            worker_fn=_build_profile_train_val_rows,
            num_workers=resolved_num_workers,
            config=worker_config,
            tokenizer=tokenizer,
        ):
            skipped_users += result.skipped_users

            if result.train_rows:
                prompt_idx = _assign_prompt_indices(result.train_rows, start_idx=prompt_idx)
                train_users.add(str(result.train_rows[0]["extra_info"]["user_id"]))
                for row in result.train_rows:
                    train_writer.write(row)

            if result.val_rows:
                prompt_idx = _assign_prompt_indices(result.val_rows, start_idx=prompt_idx)
                val_users.add(str(result.val_rows[0]["extra_info"]["user_id"]))
                for row in result.val_rows:
                    val_writer.write(row)

    if shuffle_rows:
        shuffle_parquet_rows(output, seed=shuffle_seed)
        shuffle_parquet_rows(val_output, seed=shuffle_seed + 1)

    return {
        "train_rows": _count_parquet_rows(output),
        "val_rows": _count_parquet_rows(val_output),
        "train_users": len(train_users),
        "val_users": len(val_users),
        "skipped_users": skipped_users,
    }


def write_grpo_test_rows_from_profiles(
    profiles: Iterable[UserProfile],
    *,
    tokenizer: Any,
    mode: str,
    output: str,
    data_source: str = DEFAULT_DATA_SOURCE,
    random_history_count_min: int = DEFAULT_RANDOM_HISTORY_COUNT_MIN,
    random_history_count_max: int = DEFAULT_RANDOM_HISTORY_COUNT_MAX,
    history_count_seed: int = DEFAULT_HISTORY_COUNT_SEED,
    max_targets_per_user: int | None = None,
    corpus_names: Iterable[str] | None = None,
    chunk_size: int = 1000,
    shuffle_rows: bool = False,
    shuffle_seed: int = 42,
    conditioning_mode: str = CONDITIONING_MODE_HISTORY,
    persona_map: dict[str, str] | None = None,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Write heldout test rows."""
    ordered_profiles = list(profiles)
    _validate_data_mode(mode)
    persona_map = persona_map or {}
    normalized_corpus_names = [
        normalize_subreddit_corpus_name(name)
        if not str(name).endswith(".zip") else str(name)
        for name in (list(corpus_names) if corpus_names is not None else [])
    ]
    resolved_num_workers = _resolve_num_workers(num_workers, num_profiles=len(ordered_profiles))
    print(
        f"Building held-out GRPO rows with {resolved_num_workers} worker(s) across {len(ordered_profiles)} profile(s)",
        flush=True,
    )
    skipped_users = 0
    test_users: set[str] = set()
    all_test_rows: list[dict[str, Any]] = []
    worker_config = {
        "random_history_count_min": random_history_count_min,
        "random_history_count_max": random_history_count_max,
        "history_count_seed": history_count_seed,
        "max_targets_per_user": max_targets_per_user,
        "conditioning_mode": conditioning_mode,
        "persona_map": persona_map,
        "corpus_names": normalized_corpus_names,
        "data_source": data_source,
        "tokenizer_name": getattr(tokenizer, "name_or_path", MODEL_ID),
    }

    for result in _iter_parallel_profile_results(
        profiles=ordered_profiles,
        worker_fn=_build_profile_test_rows,
        num_workers=resolved_num_workers,
        config=worker_config,
        tokenizer=tokenizer,
    ):
        skipped_users += result.skipped_users
        if result.test_rows:
            test_users.add(str(result.test_rows[0]["extra_info"]["user_id"]))
            all_test_rows.extend(result.test_rows)

    ordered_test_rows = sorted(all_test_rows, key=_test_row_sort_key)
    _assign_prompt_indices(ordered_test_rows, start_idx=0)

    with _ParquetChunkWriter(output, chunk_size=chunk_size) as test_writer:
        for row in ordered_test_rows:
            test_writer.write(row)

    return {
        "test_rows": _count_parquet_rows(output),
        "test_users": len(test_users),
        "skipped_users": skipped_users,
    }


def load_convokit_corpora(corpus_names: Iterable[str]) -> list[Any]:
    """Load ConvoKit subreddit corpora."""
    from convokit import Corpus, download

    corpora = []
    for corpus_name in corpus_names:
        normalized_name = normalize_subreddit_corpus_name(corpus_name)
        print(f"Loading ConvoKit corpus: {normalized_name}")
        corpora.append(Corpus(filename=download(normalized_name)))
    return corpora


def _count_parquet_rows(output_path: str) -> int:
    path = Path(output_path)
    if not path.is_file():
        return 0
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def _default_val_output(output: str) -> str:
    output_root, output_ext = os.path.splitext(output)
    if output_root.endswith("train"):
        return f"{output_root[:-len('train')]}val{output_ext}"
    return f"{output_root}_val{output_ext}"


def _default_test_output(output: str) -> str:
    output_root, output_ext = os.path.splitext(output)
    if output_root.endswith("train"):
        return f"{output_root[:-len('train')]}test{output_ext}"
    return f"{output_root}_test{output_ext}"


def _write_metadata(summary: dict[str, Any], output: str) -> str:
    metadata_path = f"{output}.metadata.json"
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata_path


def _load_corpora_manifest(path: Path) -> dict[str, dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return manifest


def _corpus_size_matches(path: Path, expected_size: int | None) -> bool:
    if expected_size is None:
        return path.is_file()
    return path.is_file() and path.stat().st_size == expected_size


def _download_corpus_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def download_convokit_corpora(
    *,
    output_dir: str | Path,
    manifest_path: str | Path = DEFAULT_CORPORA_MANIFEST,
    subreddits: Iterable[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """Download ConvoKit corpus archives."""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    manifest = _load_corpora_manifest(manifest_path)
    selected = list(subreddits) if subreddits else list(manifest)
    missing = [name for name in selected if name not in manifest]
    if missing:
        raise SystemExit(f"Subreddit(s) missing from manifest: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.resolve() != (output_dir / manifest_path.name).resolve():
        shutil.copy2(manifest_path, output_dir / manifest_path.name)

    for subreddit in selected:
        entry = manifest[subreddit]
        url = str(entry["url"])
        expected_size = entry.get("size_bytes")
        expected_size = int(expected_size) if expected_size is not None else None
        destination = output_dir / f"{subreddit}.corpus.zip"
        if not force and _corpus_size_matches(destination, expected_size):
            print(f"exists: {destination}")
            continue
        print(f"download: {subreddit} -> {destination}")
        if dry_run:
            continue
        _download_corpus_archive(url, destination)
        if not _corpus_size_matches(destination, expected_size):
            actual = destination.stat().st_size if destination.exists() else 0
            raise SystemExit(
                f"Downloaded size mismatch for {destination}: expected {expected_size}, got {actual}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ConvoKit subreddit history-only GRPO data")
    parser.add_argument(
        "--train_subreddits",
        nargs="+",
        default=None,
        help=(
            "Subreddits to use for train/val. With --corpus_zip_dir, defaults to the curated "
            "general ConvoKit train set including AmItheAsshole."
        ),
    )
    parser.add_argument(
        "--test_subreddits",
        nargs="+",
        default=None,
        help=(
            "Subreddits to use for held-out test parquet. With --corpus_zip_dir, defaults to "
            "Economics, TrueReddit, relationships, and MaliciousCompliance."
        ),
    )
    parser.add_argument(
        "--corpus_zip",
        default=None,
        help=(
            "Path to a downloaded monolithic ConvoKit .corpus.zip archive. "
            "When set, scans that archive directly instead of downloading corpora."
        ),
    )
    parser.add_argument(
        "--corpus_zip_dir",
        default=None,
        help=(
            "Directory of official per-subreddit ConvoKit .corpus.zip archives named "
            "<subreddit>.corpus.zip. Use this to recover OP titles from conversation metadata."
        ),
    )
    parser.add_argument(
        "--op_context_zip_dir",
        default=None,
        help=(
            "Directory of official per-subreddit ConvoKit .corpus.zip archives used only to "
            "recover OP title/body/speaker for conversation ids read from --corpus_zip."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download corpus archives from the manifest into --corpus_zip_dir before building.",
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Download corpus archives from the manifest and exit without building.",
    )
    parser.add_argument(
        "--download_manifest",
        type=Path,
        default=DEFAULT_CORPORA_MANIFEST,
        help="URL manifest of per-subreddit corpus archives.",
    )
    parser.add_argument(
        "--download_subreddits",
        nargs="*",
        default=None,
        help="Subset of manifest keys to download. Defaults to all manifest entries.",
    )
    parser.add_argument(
        "--download_force",
        action="store_true",
        help="Redownload corpus archives even if the local size matches.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output train parquet path.")
    parser.add_argument("--val_output", default=None, help="Optional output val parquet path.")
    parser.add_argument("--test_output", default=None, help="Optional output test parquet path.")
    parser.add_argument(
        "--max_train_users",
        type=int,
        default=None,
        help="Optional cap on selected train users after test users are reserved.",
    )
    parser.add_argument(
        "--train_user_offset",
        type=int,
        default=0,
        help="Skip this many eligible train users after test-user exclusion before applying --max_train_users.",
    )
    parser.add_argument(
        "--max_test_users",
        type=int,
        default=None,
        help="Optional cap on selected test users. Omit to keep every eligible held-out test user.",
    )
    parser.add_argument(
        "--test_user_offset",
        type=int,
        default=0,
        help="Skip this many eligible test users before applying --max_test_users.",
    )
    parser.add_argument(
        "--allow_user_overlap_between_splits",
        action="store_true",
        help="Allow the same Reddit account to appear in both train and test. Default is disjoint users.",
    )
    parser.add_argument(
        "--min_conversations",
        type=int,
        default=DEFAULT_MIN_CONVERSATIONS,
        help="Keep users with at least this many unique conversation ids.",
    )
    parser.add_argument(
        "--random_history_count_min",
        type=int,
        default=DEFAULT_RANDOM_HISTORY_COUNT_MIN,
        help="Deterministically sample a per-user history count at least this large.",
    )
    parser.add_argument(
        "--random_history_count_max",
        type=int,
        default=DEFAULT_RANDOM_HISTORY_COUNT_MAX,
        help="Deterministically sample a per-user history count at most this large.",
    )
    parser.add_argument(
        "--history_count_seed",
        type=int,
        default=DEFAULT_HISTORY_COUNT_SEED,
        help="Seed used for deterministic per-user random history count selection.",
    )
    parser.add_argument("--val_frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--max_targets_per_user", type=int, default=None)
    parser.add_argument("--parquet_chunk_size", type=int, default=1000)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes for per-profile GRPO row construction. "
            "Defaults to GRPO_DATA_NUM_WORKERS, PERSONA_GRPO_DATA_NUM_WORKERS, "
            "SLURM_CPUS_PER_TASK, or os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--shuffle_rows",
        action="store_true",
        help="Deterministically shuffle train/val parquet rows after writing. Test rows stay ordered by subreddit.",
    )
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument(
        "--thread_comment_retention",
        choices=(THREAD_COMMENT_RETENTION_LAST_TARGET_ANCESTOR_CHAIN,),
        default=DEFAULT_THREAD_COMMENT_RETENTION,
        help=(
            "How much of each target thread to keep before GRPO row generation. "
            "Only 'last_target_ancestor_chain' is supported: OP metadata plus the ancestor "
            "chain to each thread's final target reply."
        ),
    )
    parser.add_argument("--mode", choices=sorted(PUBLIC_DATA_MODES), default="reasoning")
    parser.add_argument(
        "--conditioning_mode",
        choices=CONDITIONING_MODE_CHOICES,
        default=CONDITIONING_MODE_HISTORY,
        help="Prompt conditioning mode for generated GRPO rows.",
    )
    parser.add_argument(
        "--persona_path",
        default=None,
        help="Persona map JSON/JSONL/pickle path required for persona-backed conditioning modes.",
    )
    parser.add_argument("--tokenizer", default=MODEL_ID)
    parser.add_argument(
        "--data_source",
        default=DEFAULT_DATA_SOURCE,
        help="Top-level data_source value. Defaults to the existing GRPO data source for compatibility.",
    )
    args = parser.parse_args()

    if args.download or args.download_only:
        download_target = Path(args.corpus_zip_dir) if args.corpus_zip_dir else Path(DEFAULT_CORPUS_ZIP_DIR)
        download_convokit_corpora(
            output_dir=download_target,
            manifest_path=args.download_manifest,
            subreddits=args.download_subreddits,
            force=args.download_force,
        )
        if not args.corpus_zip_dir:
            args.corpus_zip_dir = str(download_target)
        if args.download_only:
            return

    zip_source_count = sum(bool(value) for value in (args.corpus_zip, args.corpus_zip_dir))
    if zip_source_count > 1:
        raise ValueError("Use only one of --corpus_zip or --corpus_zip_dir")
    if args.op_context_zip_dir and not args.corpus_zip:
        raise ValueError("--op_context_zip_dir is only supported with --corpus_zip")
    history_count_spec = HistoryCountSpec(
        random_min=args.random_history_count_min,
        random_max=args.random_history_count_max,
        random_seed=args.history_count_seed,
    )
    history_count_spec.validate()
    if conditioning_mode_uses_persona(args.conditioning_mode) and not args.persona_path:
        raise ValueError(f"--persona_path is required when --conditioning_mode={args.conditioning_mode}")
    persona_map = load_persona_map(args.persona_path)

    split_args_requested = args.train_subreddits is not None or args.test_subreddits is not None
    if args.corpus_zip or args.corpus_zip_dir:
        default_train_subreddits = DEFAULT_TRAIN_SUBREDDITS
        default_test_subreddits = DEFAULT_TEST_SUBREDDITS
    elif split_args_requested:
        default_train_subreddits = []
        default_test_subreddits = []
    else:
        default_train_subreddits = DEFAULT_TRAIN_SUBREDDITS
        default_test_subreddits = DEFAULT_TEST_SUBREDDITS

    train_subreddits = _normalize_subreddit_list(args.train_subreddits or default_train_subreddits)
    test_subreddits = _normalize_subreddit_list(args.test_subreddits or default_test_subreddits)
    if not train_subreddits:
        raise ValueError("No train subreddits selected")
    subreddit_overlap = {name.lower() for name in train_subreddits}.intersection(
        name.lower() for name in test_subreddits
    )
    if subreddit_overlap:
        raise ValueError(f"Train/test subreddit overlap is not allowed: {sorted(subreddit_overlap)}")

    corpora: list[Any] | None = None
    if args.corpus_zip:
        corpus_names = [Path(args.corpus_zip).name]
    elif args.corpus_zip_dir:
        download_subreddits = train_subreddits + [name for name in test_subreddits if name not in train_subreddits]
        corpus_names = [
            path.name
            for path in _corpus_zip_paths_for_subreddits(args.corpus_zip_dir, download_subreddits)
        ]
    else:
        download_subreddits = train_subreddits + [name for name in test_subreddits if name not in train_subreddits]
        corpus_names = [normalize_subreddit_corpus_name(name) for name in download_subreddits]
        corpora = load_convokit_corpora(corpus_names)

    def _load_profiles_for_subreddits(include_subreddits: list[str], *, max_users: int | None = None):
        include_filter = include_subreddits or None
        if args.corpus_zip:
            op_context_zip_paths = (
                _corpus_zip_paths_for_subreddits(args.op_context_zip_dir, include_subreddits)
                if args.op_context_zip_dir
                else None
            )
            return profiles_from_convokit_corpus_zip(
                args.corpus_zip,
                corpus_name=Path(args.corpus_zip).name,
                include_subreddits=include_filter,
                op_context_zip_paths=op_context_zip_paths,
                op_context_corpus_names=(
                    [path.name for path in op_context_zip_paths]
                    if op_context_zip_paths is not None
                    else None
                ),
                min_conversations=args.min_conversations,
                max_users=max_users,
                source_name=DEFAULT_SOURCE_NAME,
                thread_comment_retention=args.thread_comment_retention,
            )
        if args.corpus_zip_dir:
            split_zip_paths = _corpus_zip_paths_for_subreddits(args.corpus_zip_dir, include_subreddits)
            return profiles_from_convokit_corpus_zips(
                split_zip_paths,
                corpus_names=[path.name for path in split_zip_paths],
                include_subreddits=include_filter,
                min_conversations=args.min_conversations,
                max_users=max_users,
                source_name=DEFAULT_SOURCE_NAME,
                thread_comment_retention=args.thread_comment_retention,
            )
        assert corpora is not None
        return profiles_from_convokit_corpora(
            corpora,
            corpus_names=corpus_names,
            include_subreddits=include_filter,
            min_conversations=args.min_conversations,
            max_users=max_users,
            source_name=DEFAULT_SOURCE_NAME,
            thread_comment_retention=args.thread_comment_retention,
        )

    test_profiles_all: list[UserProfile] = []
    test_profiles: list[UserProfile] = []
    test_selection = ProfileSelectionResult([], 0, 0, args.test_user_offset, args.max_test_users)
    test_stats = None
    selected_test_user_ids: set[str] = set()
    if test_subreddits:
        test_profile_limit = _prefetch_profile_limit(
            user_offset=args.test_user_offset,
            max_users=args.max_test_users,
        )
        if test_profile_limit is not None:
            print(
                "Limiting heldout profile extraction to first "
                f"{test_profile_limit} eligible test users before slicing."
            )
        test_profiles_all, test_stats = _load_profiles_for_subreddits(
            test_subreddits,
            max_users=test_profile_limit,
        )
        test_selection = _slice_profiles(
            test_profiles_all,
            user_offset=args.test_user_offset,
            max_users=args.max_test_users,
        )
        test_profiles = test_selection.profiles
        selected_test_user_ids = {profile.user_id for profile in test_profiles}

    train_profile_limit = _prefetch_profile_limit(
        user_offset=args.train_user_offset,
        max_users=args.max_train_users,
        extra_users=0 if args.allow_user_overlap_between_splits else len(selected_test_user_ids),
    )
    if train_profile_limit is not None:
        print(
            "Limiting train profile extraction to first "
            f"{train_profile_limit} eligible train users before slicing/exclusion."
        )
    train_profiles_all, train_stats = _load_profiles_for_subreddits(
        train_subreddits,
        max_users=train_profile_limit,
    )
    train_selection = _slice_profiles(
        train_profiles_all,
        user_offset=args.train_user_offset,
        max_users=args.max_train_users,
        exclude_user_ids=set() if args.allow_user_overlap_between_splits else selected_test_user_ids,
    )
    train_profiles = train_selection.profiles
    combined_profiles = anonymize_user_profiles(train_profiles + test_profiles)
    train_profiles = combined_profiles[: len(train_profiles)]
    test_profiles = combined_profiles[len(train_profiles):]
    tokenizer = load_tokenizer(args.tokenizer)
    resolved_num_workers = _resolve_num_workers(args.num_workers, num_profiles=len(train_profiles))
    print(f"Resolved row-build worker count: {resolved_num_workers}")

    val_output = args.val_output or _default_val_output(args.output)
    write_result = write_grpo_rows_from_profiles(
        train_profiles,
        tokenizer=tokenizer,
        mode=args.mode,
        output=args.output,
        val_output=val_output,
        data_source=args.data_source,
        random_history_count_min=args.random_history_count_min,
        random_history_count_max=args.random_history_count_max,
        history_count_seed=args.history_count_seed,
        val_frac=args.val_frac,
        max_targets_per_user=args.max_targets_per_user,
        corpus_names=corpus_names,
        chunk_size=args.parquet_chunk_size,
        shuffle_rows=args.shuffle_rows,
        shuffle_seed=args.shuffle_seed,
        conditioning_mode=args.conditioning_mode,
        persona_map=persona_map,
        num_workers=resolved_num_workers,
    )

    if not write_result["train_rows"]:
        raise SystemExit("No train rows generated after filtering and splitting.")

    test_output = args.test_output or _default_test_output(args.output)
    test_write_result = {
        "test_rows": 0,
        "test_users": 0,
        "skipped_users": 0,
    }
    if test_profiles:
        test_write_result = write_grpo_test_rows_from_profiles(
            test_profiles,
            tokenizer=tokenizer,
            mode=args.mode,
            output=test_output,
            data_source=args.data_source,
            random_history_count_min=args.random_history_count_min,
            random_history_count_max=args.random_history_count_max,
            history_count_seed=args.history_count_seed,
            max_targets_per_user=args.max_targets_per_user,
            corpus_names=corpus_names,
            chunk_size=args.parquet_chunk_size,
            shuffle_rows=args.shuffle_rows,
            shuffle_seed=args.shuffle_seed + 2,
            conditioning_mode=args.conditioning_mode,
            persona_map=persona_map,
            num_workers=_resolve_num_workers(args.num_workers, num_profiles=len(test_profiles)),
        )

    summary = {
        "corpus_zip": args.corpus_zip,
        "corpus_zip_dir": args.corpus_zip_dir,
        "train_subreddits": train_subreddits,
        "test_subreddits": test_subreddits,
        "counts": {
            "train": {"rows": write_result["train_rows"], "users": write_result["train_users"]},
            "val": {"rows": write_result["val_rows"], "users": write_result["val_users"]},
            "test": {"rows": test_write_result["test_rows"], "users": test_write_result["test_users"]},
        },
        "train_profile_selection": {
            "eligible_profiles_before_test_exclusion": train_selection.total_profiles,
            "excluded_selected_test_users": train_selection.excluded_overlap_users,
            "user_offset": train_selection.user_offset,
            "max_users": train_selection.max_users,
            "selected_profiles": len(train_profiles),
        },
        "test_profile_selection": {
            "eligible_profiles": test_selection.total_profiles,
            "user_offset": test_selection.user_offset,
            "max_users": test_selection.max_users,
            "selected_profiles": len(test_profiles),
        },
        "history_count_spec": history_count_spec.to_metadata(),
        "val_frac": args.val_frac,
        "min_conversations": args.min_conversations,
        "thread_comment_retention": args.thread_comment_retention,
        "shuffle_seed": args.shuffle_seed if args.shuffle_rows else None,
        "conditioning_mode": args.conditioning_mode,
        "persona_path": args.persona_path if conditioning_mode_uses_persona(args.conditioning_mode) else None,
        "skipped_users_after_split": write_result["skipped_users"],
        "skipped_test_users_after_split": test_write_result["skipped_users"],
    }
    metadata_path = _write_metadata(summary, args.output)

    print(f"Corpus names: {', '.join(corpus_names)}")
    print(f"Train subreddits: {', '.join(train_subreddits)}")
    print(f"Test subreddits:  {', '.join(test_subreddits) if test_subreddits else '(none)'}")
    print(
        "Train eligible profiles: "
        f"{train_selection.total_profiles}; selected train profiles: {len(train_profiles)}; "
        f"excluded selected test users: {train_selection.excluded_overlap_users}"
    )
    if test_subreddits:
        print(f"Test eligible profiles: {test_selection.total_profiles}; selected test profiles: {len(test_profiles)}")
    print(f"Train: {write_result['train_rows']} rows ({write_result['train_users']} users) -> {args.output}")
    print(f"Val:   {write_result['val_rows']} rows ({write_result['val_users']} users) -> {val_output}")
    if test_profiles:
        print(
            f"Test:  {test_write_result['test_rows']} rows "
            f"({test_write_result['test_users']} users) -> {test_output}"
        )
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
