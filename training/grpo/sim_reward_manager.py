"""Grouped reward manager for response-judge GRPO."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections import defaultdict
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from shared.prompt_utils import (
    parse_reasoning_and_response,
)
from training.grpo.reward import build_format_reward_info, empty_format_reward_info
from shared.judge_utils import judge_response_batch

try:
    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
except ImportError:  # pragma: no cover
    DataProto = Any  # type: ignore[assignment]
    RewardManagerBase = object  # type: ignore[assignment]


def _unwrap_object(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            return _unwrap_object(value.reshape(-1)[0])
        return [_unwrap_object(v) for v in value.tolist()]
    if isinstance(value, list) and len(value) == 1:
        return _unwrap_object(value[0])
    return value


def stable_group_key_from_extra_info(extra_info: Any, global_step: int) -> str:
    extra_info = _unwrap_object(extra_info) or {}
    prompt_idx = extra_info.get("index", extra_info.get("prompt_idx"))
    prompt_text = extra_info.get("prompt_text") or extra_info.get("context") or ""
    context = extra_info.get("context") or extra_info.get("thread_context") or ""
    user_history = extra_info.get("user_history") or ""
    user_id = extra_info.get("user_id") or ""
    raw_user_id = extra_info.get("raw_user_id") or ""
    post_id = extra_info.get("post_id") or ""
    target_idx = extra_info.get("target_idx") or ""
    ground_truth = extra_info.get("ground_truth") or ""
    digest = hashlib.sha1(
        "\n".join(
            str(part)
            for part in (
                prompt_idx if prompt_idx is not None else "",
                user_id,
                raw_user_id,
                post_id,
                target_idx,
                prompt_text,
                context,
                user_history,
                ground_truth,
            )
        ).encode("utf-8")
    ).hexdigest()
    prompt_key = f"prompt_idx:{prompt_idx}:hash:{digest}" if prompt_idx is not None else f"prompt_hash:{digest}"
    return f"{global_step}:response:{prompt_key}"


class GroupedSimRewardManager(RewardManagerBase):
    """Group sibling GRPO rollouts for one prompt and score response similarity jointly."""

    def __init__(self, config, tokenizer, compute_score=None, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        reward_kwargs = {}
        reward_model = getattr(config, "reward_model", None)
        if reward_model is None:
            reward_model = getattr(config, "reward", None)
        if reward_model is not None:
            rk = getattr(reward_model, "reward_kwargs", {})
            if rk:
                reward_kwargs = OmegaConf.to_container(rk, resolve=True) or {}

        self.tokenizer = tokenizer
        self.n_rollouts = int(reward_kwargs.get("n_rollouts", getattr(config.actor_rollout_ref.rollout, "n", 1)))
        self.group_timeout_s = float(os.environ.get("SIM_GROUP_TIMEOUT_S", "180"))
        self.judge_model = str(
            reward_kwargs.get("judge_model", os.environ.get("SIM_JUDGE_MODEL", "qwen/qwen3.5-397b-a17b"))
        )
        self._pending: dict[str, list[tuple[DataProto, asyncio.Future]]] = defaultdict(list)
        self._lock: asyncio.Lock | None = None
        self._split = "train"
        self._group_count = 0

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _blank_reward_extra_info(self) -> dict[str, float]:
        result = {
            "score": 0.0,
            "total_score": 0.0,
            "raw_reward": 0.0,
            "sim_response": 0.0,
            f"{self._split}/active/response": 0.0,
            f"{self._split}/response:score": 0.0,
        }
        result.update(empty_format_reward_info())
        return result

    def _required_group_size(self, data: DataProto) -> int:
        return 1 if data.meta_info.get("validate", False) else self.n_rollouts

    def _stable_group_key(self, data_item, global_step: int) -> str:
        return stable_group_key_from_extra_info(data_item.non_tensor_batch.get("extra_info", {}), global_step)

    def _decode_response(self, data_item) -> str:
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        return self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

    def _parse_response(self, solution_str: str, prompt_mode: str | None) -> tuple[str, str]:
        _ = prompt_mode
        return parse_reasoning_and_response(solution_str)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "GroupedSimRewardManager only supports single data items"
        data_item = data[0]
        group_key = self._stable_group_key(data_item, data.meta_info.get("global_steps", 0))
        target_group_size = self._required_group_size(data)

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        async with self.lock:
            self._pending[group_key].append((data, future))
            current_count = len(self._pending[group_key])

        if current_count >= target_group_size:
            asyncio.create_task(self._flush_key(group_key))

        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=self.group_timeout_s)
        except asyncio.TimeoutError:
            print(f"[grouped sim] timeout waiting for group {group_key}; scoring available rollouts jointly", flush=True)
            await self._flush_key(group_key)
            return await future

    async def _flush_key(self, group_key: str) -> None:
        async with self.lock:
            if group_key not in self._pending or not self._pending[group_key]:
                return
            items = self._pending.pop(group_key)

        datas = [item[0] for item in items]
        futures = [item[1] for item in items]
        try:
            results = await self._score_group(datas)
        except Exception as exc:
            for future in futures:
                if not future.done():
                    future.set_exception(exc)
            return

        for future, result in zip(futures, results):
            if not future.done():
                future.set_result(result)

    async def _score_group(self, datas: list[DataProto]) -> list[dict]:
        self._split = "val" if bool(datas[0].meta_info.get("validate", False)) else "train"
        group_started = time.perf_counter()

        candidates = []
        decode_started = time.perf_counter()
        for data in datas:
            data_item = data[0]
            extra_info = _unwrap_object(data_item.non_tensor_batch.get("extra_info", {})) or {}
            reward_model = _unwrap_object(data_item.non_tensor_batch.get("reward_model", {})) or {}
            solution_str = await asyncio.to_thread(self._decode_response, data_item)
            prompt_mode = str(extra_info.get("prompt_mode", "") or "")
            _, response = self._parse_response(solution_str, prompt_mode)
            response = str(response or "").strip()
            context = str(extra_info.get("context", "") or "")
            user_history = str(extra_info.get("user_history", "") or "")
            format_reward_info = build_format_reward_info(solution_str, "sim", prompt_mode)
            candidates.append(
                {
                    "solution_str": solution_str,
                    "response": response,
                    "format_reward_info": format_reward_info,
                    "ground_truth": str(reward_model.get("ground_truth", "") or ""),
                    "context": context,
                    "user_history": user_history,
                }
            )
        decode_s = time.perf_counter() - decode_started

        ground_truth = candidates[0]["ground_truth"]
        context = candidates[0]["context"]
        user_history = candidates[0]["user_history"]
        if any(
            candidate["ground_truth"] != ground_truth
            or candidate["context"] != context
            or candidate["user_history"] != user_history
            for candidate in candidates
        ):
            raise ValueError("GroupedSimRewardManager received inconsistent prompt context within a group")

        judged_outputs = [{"score": 0.0, "node_score": 0.0, "metrics_info": ""} for _ in candidates]
        valid_indices = [idx for idx, candidate in enumerate(candidates) if candidate["response"]]
        judge_started = time.perf_counter()
        if valid_indices:
            valid_candidates = [candidates[idx]["response"] for idx in valid_indices]
            judged = await judge_response_batch(
                user_history=user_history,
                thread_context=context,
                ground_truth=ground_truth,
                candidates=valid_candidates,
                model=self.judge_model,
                label="sim response judge",
                enable_hard_flags=False,
            )
            for local_idx, candidate_idx in enumerate(valid_indices):
                judged_outputs[candidate_idx] = judged[local_idx]
        judge_s = time.perf_counter() - judge_started

        expected_group_size = self._required_group_size(datas[0])
        self._group_count += 1
        if len(candidates) != expected_group_size:
            print(
                f"[grouped sim WARNING] partial group scored: group_size={len(candidates)} expected={expected_group_size}",
                flush=True,
            )
        if self._group_count <= 3:
            score_preview = [round(float(judged_outputs[i]["score"]), 4) for i in range(len(candidates))]
            print(
                f"[grouped sim #{self._group_count}] group_size={len(candidates)} "
                f"expected={expected_group_size} scores={score_preview} "
                f"decode_s={decode_s:.2f} judge_s={judge_s:.2f} total_s={time.perf_counter() - group_started:.2f}",
                flush=True,
            )

        results = []
        for idx, candidate in enumerate(candidates):
            if not candidate["response"]:
                raw_reward = 0.0
            else:
                raw_reward = 0.9 * float(judged_outputs[idx]["score"])
            format_reward_info = candidate["format_reward_info"]
            format_score = float(format_reward_info["format_score"])
            total_score = max(0.0, raw_reward + format_score)
            reward_extra_info = self._blank_reward_extra_info()
            reward_extra_info.update(format_reward_info)
            reward_extra_info.update(
                {
                    "score": total_score,
                    "total_score": total_score,
                    "raw_reward": raw_reward,
                    "sim_response": raw_reward,
                    f"{self._split}/active/response": 1.0,
                    f"{self._split}/response:score": raw_reward,
                }
            )
            results.append(
                {
                    "reward_score": total_score,
                    "reward_extra_info": reward_extra_info,
                }
            )
        return results
