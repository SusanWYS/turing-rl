"""Load ConvoKit subreddit histories."""

from __future__ import annotations

import os
import json
import io
import multiprocessing as mp
import re
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Iterable
import zipfile

from data.utils import Comment, Thread, UserProfile


DEFAULT_MIN_CONVERSATIONS = 8
DEFAULT_SOURCE_NAME = "convokit_subreddit"
THREAD_COMMENT_RETENTION_LAST_TARGET_ANCESTOR_CHAIN = "last_target_ancestor_chain"
DEFAULT_THREAD_COMMENT_RETENTION = THREAD_COMMENT_RETENTION_LAST_TARGET_ANCESTOR_CHAIN
UNUSABLE_SPEAKER_IDS = frozenset({"", "[deleted]", "deleted", "none", "null"})
UNUSABLE_TEXTS = frozenset({"", "[deleted]", "[removed]", "deleted", "removed", "none", "null"})
AUTOMODERATOR_ID = "automoderator"
REDACTED_REDDIT_USER = "[REDDIT USER]"
REDACTED_REDDIT_THREAD = "[REDDIT THREAD]"
REDACTED_REDDIT_LINK = "[REDDIT LINK]"
REDACTED_REDDIT_ID = "[REDDIT ID]"
REDDIT_COMMENT_URL_RE = re.compile(
    r"https?://(?:(?:www|old|np)\.)?reddit\.com/r/[A-Za-z0-9_]+/comments/[A-Za-z0-9]+[^\s<>()\]]*",
    re.IGNORECASE,
)
REDDIT_SHORTLINK_RE = re.compile(
    r"https?://(?:www\.)?redd\.it/[A-Za-z0-9]+[^\s<>()\]]*",
    re.IGNORECASE,
)
REDDIT_USER_URL_RE = re.compile(
    r"https?://(?:(?:www|old|np)\.)?reddit\.com/(?:user|u)/[A-Za-z0-9_-]+/?[^\s<>()\]]*",
    re.IGNORECASE,
)
REDDIT_HOSTED_URL_RE = re.compile(
    r"https?://[^\s<>()\]]*(?:reddit\.com|redd\.it)[^\s<>()\]]*",
    re.IGNORECASE,
)
REDDIT_USER_MENTION_RE = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_-]{3,20}\b", re.IGNORECASE)
REDDIT_THING_ID_RE = re.compile(r"\bt[1-6]_[a-z0-9]{5,}\b", re.IGNORECASE)


@dataclass
class ConvokitSubredditStats:
    """ConvoKit adaptation counters."""

    corpus_names: list[str] = field(default_factory=list)
    total_utterances: int = 0
    kept_utterances: int = 0
    skipped_missing_conversation: int = 0
    skipped_unusable_speaker: int = 0
    skipped_unusable_text: int = 0
    skipped_unusable_root: int = 0
    total_conversations: int = 0
    usable_conversations: int = 0
    eligible_users: int = 0
    selected_users: int = 0
    missing_op_text_conversations: int = 0
    subreddit_count: int = 0
    all_user_conversation_counts: dict[str, int] = field(default_factory=dict)
    selected_user_ids: list[str] = field(default_factory=list)


@dataclass
class _RawComment:
    user_id: str
    text: str
    timestamp: float
    comment_id: str
    parent_id: str
    context_only: bool = False
    display_label: str = ""


@dataclass
class _ThreadRecord:
    post_id: str
    source_name: str
    subreddit: str = ""
    title: str = ""
    op_text: str = ""
    root_id: str = ""
    root_speaker_id: str = ""
    unusable_root: bool = False
    comments: list[_RawComment] = field(default_factory=list)


@dataclass
class _OpContext:
    title: str = ""
    op_text: str = ""
    root_id: str = ""
    root_speaker_id: str = ""
    subreddit: str = ""


@dataclass
class _ZipFirstPassResult:
    conversation_meta_by_id: dict[str, dict[str, Any]]
    conversation_subreddits: dict[str, str]
    user_thread_ts: dict[str, dict[str, float]]
    unusable_root_ids: set[str]
    subreddits: set[str]
    total_utterances: int = 0
    kept_utterances: int = 0
    skipped_missing_conversation: int = 0
    skipped_unusable_speaker: int = 0
    skipped_unusable_text: int = 0


@dataclass
class _ZipSecondPassResult:
    records: dict[str, _ThreadRecord]


def normalize_subreddit_corpus_name(name: str) -> str:
    """Return the ConvoKit corpus name."""
    stripped = str(name or "").strip()
    if not stripped:
        raise ValueError("subreddit corpus name must be non-empty")
    if stripped.startswith("subreddit-"):
        return stripped
    return f"subreddit-{stripped}"


def subreddit_from_corpus_name(name: str) -> str:
    """Infer the subreddit label."""
    stripped = Path(str(name or "").strip()).name
    if stripped.endswith(".corpus.zip"):
        stripped = stripped[: -len(".corpus.zip")]
    elif stripped.endswith(".zip"):
        stripped = stripped[: -len(".zip")]
    if stripped.startswith("subreddit-"):
        return stripped[len("subreddit-"):]
    return stripped


def normalize_subreddit_name(name: str) -> str:
    """Normalize a subreddit label."""
    stripped = str(name or "").strip()
    if stripped.lower().startswith("r/"):
        stripped = stripped[2:]
    if stripped.lower().startswith("subreddit-"):
        stripped = stripped[len("subreddit-"):]
    return stripped.strip()


def _subreddit_filter_set(names: Iterable[str] | None) -> set[str] | None:
    if names is None:
        return None
    return {
        normalize_subreddit_name(name).lower()
        for name in names
        if normalize_subreddit_name(name)
    }


def _subreddit_is_selected(
    subreddit: str,
    *,
    include_subreddits: set[str] | None,
    exclude_subreddits: set[str] | None,
) -> bool:
    key = normalize_subreddit_name(subreddit).lower()
    if include_subreddits is not None and key not in include_subreddits:
        return False
    if exclude_subreddits is not None and key in exclude_subreddits:
        return False
    return True


def is_unusable_speaker_id(speaker_id: str) -> bool:
    """Return whether a speaker should be skipped."""
    normalized = str(speaker_id or "").strip().lower()
    if normalized in UNUSABLE_SPEAKER_IDS:
        return True
    if normalized == AUTOMODERATOR_ID:
        return True
    return normalized.endswith("bot")


def is_unusable_text(text: str) -> bool:
    """Return whether text is unusable."""
    normalized = str(text or "").strip()
    return normalized.lower() in UNUSABLE_TEXTS


def _as_dict(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, Mapping):
        return dict(meta)
    return {}


def _get_value(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if callable(value) and name.startswith("get_"):
            try:
                return value()
            except TypeError:
                continue
        return value
    return default


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def scrub_reddit_identifiers(text: str) -> str:
    """Redact Reddit identifiers."""
    cleaned = _string_or_empty(text)
    if not cleaned:
        return ""
    cleaned = REDDIT_COMMENT_URL_RE.sub(REDACTED_REDDIT_THREAD, cleaned)
    cleaned = REDDIT_SHORTLINK_RE.sub(REDACTED_REDDIT_THREAD, cleaned)
    cleaned = REDDIT_USER_URL_RE.sub(REDACTED_REDDIT_USER, cleaned)
    cleaned = REDDIT_HOSTED_URL_RE.sub(REDACTED_REDDIT_LINK, cleaned)
    cleaned = REDDIT_USER_MENTION_RE.sub(REDACTED_REDDIT_USER, cleaned)
    cleaned = REDDIT_THING_ID_RE.sub(REDACTED_REDDIT_ID, cleaned)
    return cleaned


def _context_only_comment_text(text: str) -> str:
    """Return placeholder text for skipped comments."""
    normalized = _string_or_empty(text)
    lowered = normalized.lower()
    if lowered in {"[deleted]", "deleted"}:
        return "[deleted]"
    if lowered in {"[removed]", "removed"}:
        return "[removed]"
    if lowered in {"", "none", "null"}:
        return "[unavailable comment]"
    return scrub_reddit_identifiers(normalized)


def _context_only_display_label(speaker_id: str, text: str) -> str:
    """Return a label for skipped context comments."""
    normalized_speaker = str(speaker_id or "").strip().lower()
    normalized_text = str(text or "").strip().lower()

    if normalized_text in {"[deleted]", "deleted"}:
        return "OTHER - DELETED"
    if normalized_text in {"[removed]", "removed"}:
        return "OTHER - REMOVED"
    if normalized_speaker == AUTOMODERATOR_ID:
        return "OTHER - AUTOMOD"
    if normalized_speaker.endswith("bot"):
        return "OTHER - BOT"
    if normalized_speaker in UNUSABLE_SPEAKER_IDS:
        return "OTHER - DELETED USER"
    return "OTHER - UNAVAILABLE USER"


def _make_raw_comment(
    *,
    speaker_id: str,
    text: str,
    timestamp: float,
    comment_id: str,
    parent_id: str,
    context_only: bool = False,
) -> _RawComment:
    return _RawComment(
        user_id="" if context_only else speaker_id,
        text=_context_only_comment_text(text) if context_only else scrub_reddit_identifiers(text),
        timestamp=timestamp,
        comment_id=comment_id,
        parent_id=parent_id,
        context_only=context_only,
        display_label=_context_only_display_label(speaker_id, text) if context_only else "",
    )


def _coerce_timestamp(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _speaker_id(utterance: Any) -> str:
    speaker = _get_value(utterance, "speaker", default=None)
    if speaker is None:
        return _string_or_empty(_get_value(utterance, "speaker_id", default=""))
    return _string_or_empty(_get_value(speaker, "id", "name", default=speaker))


def _utterance_id(utterance: Any) -> str:
    return _string_or_empty(_get_value(utterance, "id", "utterance_id", default=""))


def _conversation_id(utterance: Any) -> str:
    return _string_or_empty(_get_value(utterance, "conversation_id", default=""))


def _reply_to(utterance: Any) -> str:
    return _string_or_empty(_get_value(utterance, "reply_to", default=""))


def _utterance_text(utterance: Any) -> str:
    return _string_or_empty(_get_value(utterance, "text", default=""))


def _utterance_meta(utterance: Any) -> dict[str, Any]:
    return _as_dict(_get_value(utterance, "meta", default={}))


def _utterance_dict_meta(utterance: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(utterance.get("meta", {}))


def _conversation_object_id(conversation: Any) -> str:
    return _string_or_empty(_get_value(conversation, "id", "conversation_id", default=""))


def _conversation_meta(conversation: Any) -> dict[str, Any]:
    return _as_dict(_get_value(conversation, "meta", default={}))


def _conversation_dict_meta(conversation: Any) -> dict[str, Any]:
    data = _as_dict(conversation)
    meta = _as_dict(data.get("meta"))
    if meta:
        return meta
    return data


def _iter_conversations(corpus: Any) -> Iterable[Any]:
    if hasattr(corpus, "iter_conversations"):
        return corpus.iter_conversations()
    conversations = _get_value(corpus, "conversations", default=())
    if isinstance(conversations, dict):
        return conversations.values()
    return conversations or ()


def _iter_utterances(corpus: Any) -> Iterable[Any]:
    if hasattr(corpus, "iter_utterances"):
        return corpus.iter_utterances()
    utterances = _get_value(corpus, "utterances", default=())
    if isinstance(utterances, dict):
        return utterances.values()
    return utterances or ()


def _compose_op_text(title: str, text: str) -> str:
    title = scrub_reddit_identifiers(title)
    text = scrub_reddit_identifiers(text)
    if title and text and title != text:
        return f"{title}\n\n{text}"
    return text or title


def _compose_root_op_text(title: str, text: str) -> str:
    """Compose displayable OP text."""
    body = "" if is_unusable_text(text) else text
    return _compose_op_text(title, body)


def _fallback_op_text(post_id: str) -> str:
    return "[Original post text unavailable in ConvoKit archive]"


def _thread_sort_timestamp(record: _ThreadRecord, target_user_id: str) -> float:
    timestamps = [
        comment.timestamp
        for comment in record.comments
        if not comment.context_only and comment.user_id == target_user_id
    ]
    if timestamps:
        return min(timestamps)
    return min(
        (comment.timestamp for comment in record.comments if not comment.context_only),
        default=0.0,
    )


def _build_thread(record: _ThreadRecord, target_user_id: str) -> Thread:
    sorted_comments = sorted(record.comments, key=lambda c: (c.timestamp, c.comment_id))
    comments: list[Comment] = []
    target_indices: list[int] = []
    comment_id_to_idx: dict[str, int] = {}

    for index, raw in enumerate(sorted_comments):
        is_target = not raw.context_only and raw.user_id == target_user_id
        display_label = raw.display_label
        if not is_target and not raw.context_only and raw.user_id == record.root_speaker_id:
            display_label = "OTHER - OP"
        comments.append(
            Comment(
                text=raw.text,
                user_id="" if raw.context_only else raw.user_id,
                timestamp=raw.timestamp,
                turn_id=index,
                is_target_user=is_target,
                comment_id=raw.comment_id,
                parent_id=raw.parent_id,
                display_label=display_label,
            )
        )
        if raw.comment_id:
            comment_id_to_idx[raw.comment_id] = index
        if is_target:
            target_indices.append(index)

    return Thread(
        post_id=record.post_id,
        op_text=record.op_text,
        comments=comments,
        target_user_comment_indices=target_indices,
        comment_id_to_idx=comment_id_to_idx,
        source_name=record.source_name,
        subreddit=record.subreddit,
    )


def _trim_threads_to_last_target_reply_chain(threads: Iterable[Thread]) -> list[Thread]:
    """Keep OP metadata plus the final target chain."""
    trimmed_threads: list[Thread] = []
    for thread in threads:
        if not thread.target_user_comment_indices:
            continue
        trimmed = thread.trim_to_target_reply_chain()
        if trimmed.target_user_comment_indices:
            trimmed_threads.append(trimmed)
    return trimmed_threads


def _normalize_thread_comment_retention(thread_comment_retention: str) -> str:
    normalized = str(thread_comment_retention or "").strip().lower().replace("-", "_")
    if not normalized or normalized == THREAD_COMMENT_RETENTION_LAST_TARGET_ANCESTOR_CHAIN:
        return THREAD_COMMENT_RETENTION_LAST_TARGET_ANCESTOR_CHAIN
    raise ValueError(
        "thread_comment_retention only supports "
        f"{THREAD_COMMENT_RETENTION_LAST_TARGET_ANCESTOR_CHAIN!r}; got {thread_comment_retention!r}"
    )


def _apply_thread_comment_retention(
    threads: Iterable[Thread],
    *,
    thread_comment_retention: str,
) -> list[Thread]:
    _normalize_thread_comment_retention(thread_comment_retention)
    return _trim_threads_to_last_target_reply_chain(threads)


def _build_sequential_id_map(values: Iterable[str], prefix: str) -> dict[str, str]:
    ordered = sorted({_string_or_empty(value) for value in values if _string_or_empty(value)})
    return {value: f"{prefix}_{index:06d}" for index, value in enumerate(ordered)}


def anonymize_user_profiles(
    profiles: Iterable[UserProfile],
    *,
    user_prefix: str = "convokit_user",
    thread_prefix: str = "convokit_thread",
    comment_prefix: str = "convokit_comment",
) -> list[UserProfile]:
    """Return anonymized profiles."""
    profile_list = list(profiles)
    user_ids: set[str] = set()
    thread_ids: set[str] = set()
    comment_ids: set[str] = set()

    for profile in profile_list:
        user_ids.update(
            user_id for user_id in (profile.user_id, profile.raw_user_id, profile.profile_id) if user_id
        )
        for thread in profile.train_threads:
            if thread.post_id:
                thread_ids.add(thread.post_id)
            for comment in thread.comments:
                if comment.user_id:
                    user_ids.add(comment.user_id)
                if comment.comment_id:
                    comment_ids.add(comment.comment_id)
                if comment.parent_id:
                    comment_ids.add(comment.parent_id)
            comment_ids.update(str(comment_id) for comment_id in thread.comment_id_to_idx)

    user_id_map = _build_sequential_id_map(user_ids, user_prefix)
    thread_id_map = _build_sequential_id_map(thread_ids, thread_prefix)
    comment_id_map = _build_sequential_id_map(comment_ids, comment_prefix)

    anonymized_profiles: list[UserProfile] = []
    for profile in profile_list:
        profile_user_id = user_id_map.get(profile.user_id, profile.user_id)
        anonymized_threads: list[Thread] = []
        for thread in profile.train_threads:
            anonymized_comments: list[Comment] = []
            comment_id_to_idx: dict[str, int] = {}
            for index, comment in enumerate(thread.comments):
                comment_id = comment_id_map.get(comment.comment_id, "")
                parent_id = comment_id_map.get(comment.parent_id, "")
                anonymized_comments.append(
                    Comment(
                        text=scrub_reddit_identifiers(comment.text),
                        user_id=(
                            ""
                            if comment.display_label == "OTHER - OP"
                            else user_id_map.get(comment.user_id, comment.user_id)
                        ),
                        timestamp=comment.timestamp,
                        turn_id=comment.turn_id,
                        is_target_user=comment.is_target_user,
                        comment_id=comment_id,
                        parent_id=parent_id,
                        display_label=comment.display_label,
                    )
                )
                if comment_id:
                    comment_id_to_idx[comment_id] = index

            anonymized_threads.append(
                Thread(
                    post_id=thread_id_map.get(thread.post_id, thread.post_id),
                    op_text=scrub_reddit_identifiers(thread.op_text),
                    comments=anonymized_comments,
                    target_user_comment_indices=list(thread.target_user_comment_indices),
                    comment_id_to_idx=comment_id_to_idx,
                    source_name=thread.source_name,
                    subreddit=thread.subreddit,
                )
            )

        anonymized_profiles.append(
            UserProfile(
                user_id=profile_user_id,
                raw_user_id=profile_user_id,
                profile_id=profile_user_id,
                train_threads=anonymized_threads,
                source_name=profile.source_name,
            )
        )
    return anonymized_profiles


def _collect_conversation_meta(corpus: Any, default_subreddit: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for conversation in _iter_conversations(corpus):
        conversation_id = _conversation_object_id(conversation)
        if not conversation_id:
            continue
        meta = _conversation_meta(conversation)
        if default_subreddit and not meta.get("subreddit"):
            meta["subreddit"] = default_subreddit
        by_id[conversation_id] = meta
    return by_id


def _speaker_id_from_utterance_dict(utterance: dict[str, Any]) -> str:
    speaker = utterance.get("speaker")
    if isinstance(speaker, dict):
        return _string_or_empty(speaker.get("id") or speaker.get("name"))
    if speaker is not None:
        return _string_or_empty(speaker)
    return _string_or_empty(utterance.get("user"))


def _conversation_id_from_utterance_dict(utterance: dict[str, Any]) -> str:
    return _string_or_empty(
        utterance.get("conversation_id")
        or utterance.get("root")
        or utterance.get("conversation")
    )


def _reply_to_from_utterance_dict(utterance: dict[str, Any]) -> str:
    return _string_or_empty(utterance.get("reply_to") or utterance.get("reply-to"))


def _utterance_id_from_dict(utterance: dict[str, Any]) -> str:
    return _string_or_empty(utterance.get("id") or utterance.get("utterance_id"))


def _find_zip_member(zip_file: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in zip_file.namelist() if name.endswith(f"/{basename}") or name == basename]
    if not matches:
        raise FileNotFoundError(f"{basename} not found in {zip_file.filename}")
    return matches[0]


def _load_zip_conversation_meta(zip_file: zipfile.ZipFile) -> dict[str, dict[str, Any]]:
    member = _find_zip_member(zip_file, "conversations.json")
    raw = json.loads(zip_file.read(member))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid conversations.json in {zip_file.filename}")
    return {
        _string_or_empty(conversation_id): _conversation_dict_meta(value)
        for conversation_id, value in raw.items()
        if _string_or_empty(conversation_id)
    }


def _read_more_json_chunk(stream: io.TextIOBase, buffer: str, *, chunk_size: int = 1 << 20) -> tuple[str, bool]:
    chunk = stream.read(chunk_size)
    if not chunk:
        return buffer, True
    return buffer + chunk, False


def _load_zip_conversation_meta_subset(
    zip_file: zipfile.ZipFile,
    conversation_ids: set[str],
    *,
    default_subreddit: str = "",
) -> dict[str, dict[str, Any]]:
    """Load selected conversation metadata."""
    if not conversation_ids:
        return {}
    member = _find_zip_member(zip_file, "conversations.json")

    wanted = set(conversation_ids)
    found: dict[str, dict[str, Any]] = {}
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    eof = False
    started = False

    with zip_file.open(member) as raw_handle:
        stream = io.TextIOWrapper(raw_handle, encoding="utf-8")
        while wanted:
            if pos > (1 << 20):
                buffer = buffer[pos:]
                pos = 0

            while True:
                if pos >= len(buffer) and not eof:
                    buffer, eof = _read_more_json_chunk(stream, buffer)
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if not started:
                    if pos >= len(buffer):
                        if eof:
                            return found
                        continue
                    if buffer[pos] != "{":
                        raise ValueError(f"Invalid conversations.json in {zip_file.filename}")
                    pos += 1
                    started = True
                    continue
                if pos >= len(buffer):
                    if eof:
                        return found
                    continue
                if buffer[pos] == "}":
                    return found
                if buffer[pos] == ",":
                    pos += 1
                    continue
                break

            while True:
                try:
                    conversation_id, end = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    buffer, eof = _read_more_json_chunk(stream, buffer)
            conversation_id = _string_or_empty(conversation_id)
            pos = end

            while True:
                if pos >= len(buffer) and not eof:
                    buffer, eof = _read_more_json_chunk(stream, buffer)
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer):
                    break
                if eof:
                    return found
            if buffer[pos] != ":":
                raise ValueError(f"Invalid conversations.json in {zip_file.filename}")
            pos += 1

            while True:
                if pos >= len(buffer) and not eof:
                    buffer, eof = _read_more_json_chunk(stream, buffer)
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                try:
                    value, end = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    buffer, eof = _read_more_json_chunk(stream, buffer)
            pos = end

            if conversation_id in wanted:
                meta = _conversation_dict_meta(value)
                if default_subreddit and not meta.get("subreddit"):
                    meta["subreddit"] = default_subreddit
                found[conversation_id] = meta
                wanted.remove(conversation_id)

    return found


def _iter_zip_utterance_dicts(zip_file: zipfile.ZipFile) -> Iterable[dict[str, Any]]:
    member = _find_zip_member(zip_file, "utterances.jsonl")
    with zip_file.open(member) as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def _convokit_zip_max_workers(num_paths: int) -> int:
    for env_name in ("CONVOKIT_ZIP_NUM_WORKERS", "PERSONA_CONVOKIT_ZIP_NUM_WORKERS", "SLURM_CPUS_PER_TASK"):
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        try:
            return max(1, min(num_paths, int(raw_value)))
        except ValueError as exc:
            raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc
    return max(1, min(num_paths, os.cpu_count() or 1))


def _convokit_zip_mp_context() -> mp.context.BaseContext:
    available_methods = set(mp.get_all_start_methods())
    if "fork" in available_methods:
        return mp.get_context("fork")
    return mp.get_context()


def _scan_convokit_zip_first_pass_mp(
    path: str | Path,
    corpus_name: str,
    *,
    include_subreddits: set[str] | None,
    exclude_subreddits: set[str] | None,
) -> "_ZipFirstPassResult":
    return _scan_convokit_zip_first_pass(
        path=path,
        corpus_name=corpus_name,
        include_subreddits=include_subreddits,
        exclude_subreddits=exclude_subreddits,
    )


def _scan_convokit_zip_second_pass_mp(
    path: str | Path,
    corpus_name: str,
    *,
    selected_conversation_ids: set[str],
    conversation_meta_by_id: dict[str, dict[str, Any]],
    conversation_subreddits: dict[str, str],
    source_name: str,
) -> "_ZipSecondPassResult":
    return _scan_convokit_zip_second_pass(
        path=path,
        corpus_name=corpus_name,
        selected_conversation_ids=selected_conversation_ids,
        conversation_meta_by_id=conversation_meta_by_id,
        conversation_subreddits=conversation_subreddits,
        source_name=source_name,
    )


def _merge_user_thread_ts(
    destination: dict[str, dict[str, float]],
    source: dict[str, dict[str, float]],
) -> None:
    for user_id, ts_map in source.items():
        dest_map = destination.setdefault(user_id, {})
        for conversation_id, timestamp in ts_map.items():
            previous = dest_map.get(conversation_id)
            if previous is None or timestamp < previous:
                dest_map[conversation_id] = timestamp


def _merge_thread_record(destination: _ThreadRecord, source: _ThreadRecord) -> None:
    if source.subreddit and not destination.subreddit:
        destination.subreddit = source.subreddit
    if source.title and not destination.title:
        destination.title = source.title
    if source.op_text and not destination.op_text:
        destination.op_text = source.op_text
    if source.root_id and not destination.root_id:
        destination.root_id = source.root_id
    if source.root_speaker_id and not destination.root_speaker_id:
        destination.root_speaker_id = source.root_speaker_id
    destination.unusable_root = destination.unusable_root or source.unusable_root
    destination.comments.extend(source.comments)


def _scan_convokit_zip_first_pass(
    *,
    path: str | Path,
    corpus_name: str,
    include_subreddits: set[str] | None,
    exclude_subreddits: set[str] | None,
) -> _ZipFirstPassResult:
    zip_path = Path(path)
    default_subreddit = subreddit_from_corpus_name(corpus_name)
    conversation_meta_by_id: dict[str, dict[str, Any]] = {}
    conversation_subreddits: dict[str, str] = {}
    user_thread_ts: dict[str, dict[str, float]] = {}
    unusable_root_ids: set[str] = set()
    subreddits: set[str] = set()
    result = _ZipFirstPassResult(
        conversation_meta_by_id=conversation_meta_by_id,
        conversation_subreddits=conversation_subreddits,
        user_thread_ts=user_thread_ts,
        unusable_root_ids=unusable_root_ids,
        subreddits=subreddits,
    )

    with zipfile.ZipFile(zip_path) as zip_file:
        zip_conversation_meta_by_id = _load_zip_conversation_meta(zip_file)
        for conversation_id, raw_meta in zip_conversation_meta_by_id.items():
            meta = dict(raw_meta)
            if default_subreddit and not meta.get("subreddit"):
                meta["subreddit"] = default_subreddit
            conversation_meta_by_id[conversation_id] = meta

            subreddit = _string_or_empty(meta.get("subreddit"))
            if subreddit and _subreddit_is_selected(
                subreddit,
                include_subreddits=include_subreddits,
                exclude_subreddits=exclude_subreddits,
            ):
                conversation_subreddits[conversation_id] = subreddit
                subreddits.add(subreddit)

        for utterance in _iter_zip_utterance_dicts(zip_file):
            result.total_utterances += 1
            utterance_id = _utterance_id_from_dict(utterance)
            conversation_id = _conversation_id_from_utterance_dict(utterance)
            if not conversation_id:
                result.skipped_missing_conversation += 1
                continue

            reply_to = _reply_to_from_utterance_dict(utterance)
            speaker_id = _speaker_id_from_utterance_dict(utterance)
            text = _string_or_empty(utterance.get("text"))
            timestamp = _coerce_timestamp(utterance.get("timestamp"))
            meta = _utterance_dict_meta(utterance)
            conversation_meta = conversation_meta_by_id.get(conversation_id, {})
            subreddit = _string_or_empty(
                meta.get("subreddit")
                or conversation_meta.get("subreddit")
                or default_subreddit
            )
            if not _subreddit_is_selected(
                subreddit,
                include_subreddits=include_subreddits,
                exclude_subreddits=exclude_subreddits,
            ):
                continue
            if subreddit:
                conversation_subreddits.setdefault(conversation_id, subreddit)
                subreddits.add(subreddit)

            is_root = not reply_to or utterance_id == conversation_id
            root_title = _string_or_empty(conversation_meta.get("title") or meta.get("title"))
            root_op_text = _compose_root_op_text(root_title, text) if is_root else ""
            unusable_speaker = is_unusable_speaker_id(speaker_id)
            unusable_text = is_unusable_text(text)
            if is_root and root_op_text:
                unusable_speaker = False
                unusable_text = False
            if unusable_speaker:
                result.skipped_unusable_speaker += 1
            if unusable_text:
                result.skipped_unusable_text += 1
            if unusable_speaker or unusable_text:
                if is_root:
                    unusable_root_ids.add(conversation_id)
                continue

            result.kept_utterances += 1
            if is_root:
                continue

            previous = user_thread_ts.setdefault(speaker_id, {}).get(conversation_id)
            if previous is None or timestamp < previous:
                user_thread_ts[speaker_id][conversation_id] = timestamp

    return result


def _scan_convokit_zip_second_pass(
    *,
    path: str | Path,
    corpus_name: str,
    selected_conversation_ids: set[str],
    conversation_meta_by_id: dict[str, dict[str, Any]],
    conversation_subreddits: dict[str, str],
    source_name: str,
) -> _ZipSecondPassResult:
    zip_path = Path(path)
    default_subreddit = subreddit_from_corpus_name(corpus_name)
    records: dict[str, _ThreadRecord] = {}

    with zipfile.ZipFile(zip_path) as zip_file:
        for utterance in _iter_zip_utterance_dicts(zip_file):
            conversation_id = _conversation_id_from_utterance_dict(utterance)
            if conversation_id not in selected_conversation_ids:
                continue

            utterance_id = _utterance_id_from_dict(utterance)
            reply_to = _reply_to_from_utterance_dict(utterance)
            speaker_id = _speaker_id_from_utterance_dict(utterance)
            text = _string_or_empty(utterance.get("text"))
            meta = _utterance_dict_meta(utterance)
            conversation_meta = conversation_meta_by_id.get(conversation_id, {})
            subreddit = _string_or_empty(
                meta.get("subreddit")
                or conversation_meta.get("subreddit")
                or conversation_subreddits.get(conversation_id)
                or default_subreddit
            )
            title = _string_or_empty(conversation_meta.get("title") or meta.get("title"))
            record = records.get(conversation_id)
            if record is None:
                record = _ThreadRecord(
                    post_id=conversation_id,
                    source_name=source_name,
                    subreddit=subreddit,
                    title=title,
                )
                records[conversation_id] = record
            else:
                if subreddit and not record.subreddit:
                    record.subreddit = subreddit
                if title and not record.title:
                    record.title = title

            is_root = not reply_to or utterance_id == conversation_id
            root_op_text = _compose_root_op_text(title, text) if is_root else ""
            unusable_speaker = is_unusable_speaker_id(speaker_id)
            unusable_text = is_unusable_text(text)
            if is_root and root_op_text:
                unusable_speaker = False
                unusable_text = False
            if unusable_speaker or unusable_text:
                if is_root:
                    record.unusable_root = True
                else:
                    parent_id = "" if not reply_to or reply_to == conversation_id else reply_to
                    record.comments.append(
                        _make_raw_comment(
                            speaker_id=speaker_id,
                            text=text,
                            timestamp=_coerce_timestamp(utterance.get("timestamp")),
                            comment_id=utterance_id,
                            parent_id=parent_id,
                            context_only=True,
                        )
                    )
                continue

            if is_root:
                record.root_id = utterance_id or conversation_id
                record.root_speaker_id = speaker_id
                record.op_text = root_op_text
                continue

            parent_id = "" if not reply_to or reply_to == conversation_id else reply_to
            record.comments.append(
                _make_raw_comment(
                    speaker_id=speaker_id,
                    text=text,
                    timestamp=_coerce_timestamp(utterance.get("timestamp")),
                    comment_id=utterance_id,
                    parent_id=parent_id,
                )
            )

    return _ZipSecondPassResult(records=records)


def _ensure_thread_record(
    records: dict[str, _ThreadRecord],
    conversation_id: str,
    *,
    source_name: str,
    conversation_meta: dict[str, Any],
    utterance_meta: dict[str, Any],
    default_subreddit: str,
) -> _ThreadRecord:
    if conversation_id not in records:
        title = _string_or_empty(conversation_meta.get("title") or utterance_meta.get("title"))
        subreddit = _string_or_empty(
            conversation_meta.get("subreddit")
            or utterance_meta.get("subreddit")
            or default_subreddit
        )
        records[conversation_id] = _ThreadRecord(
            post_id=conversation_id,
            source_name=source_name,
            subreddit=subreddit,
            title=title,
        )
    else:
        record = records[conversation_id]
        if not record.title:
            record.title = _string_or_empty(conversation_meta.get("title") or utterance_meta.get("title"))
        if not record.subreddit:
            record.subreddit = _string_or_empty(
                conversation_meta.get("subreddit")
                or utterance_meta.get("subreddit")
                or default_subreddit
            )
    return records[conversation_id]


def build_thread_records_from_convokit_corpora(
    corpora: Iterable[Any],
    *,
    corpus_names: Iterable[str] | None = None,
    include_subreddits: Iterable[str] | None = None,
    exclude_subreddits: Iterable[str] | None = None,
    source_name: str = DEFAULT_SOURCE_NAME,
) -> tuple[dict[str, _ThreadRecord], ConvokitSubredditStats]:
    """Aggregate subreddit corpora into thread records."""
    corpus_list = list(corpora)
    normalized_names = [
        normalize_subreddit_corpus_name(name)
        for name in (list(corpus_names) if corpus_names is not None else [])
    ]
    if normalized_names and len(normalized_names) != len(corpus_list):
        raise ValueError("corpus_names length must match corpora length")

    include_filter = _subreddit_filter_set(include_subreddits)
    exclude_filter = _subreddit_filter_set(exclude_subreddits)
    stats = ConvokitSubredditStats(corpus_names=normalized_names)
    records: dict[str, _ThreadRecord] = {}

    for corpus_index, corpus in enumerate(corpus_list):
        corpus_name = normalized_names[corpus_index] if normalized_names else ""
        default_subreddit = subreddit_from_corpus_name(corpus_name)
        conversation_meta_by_id = _collect_conversation_meta(corpus, default_subreddit)

        for utterance in _iter_utterances(corpus):
            stats.total_utterances += 1
            conversation_id = _conversation_id(utterance)
            if not conversation_id:
                stats.skipped_missing_conversation += 1
                continue

            utterance_meta = _utterance_meta(utterance)
            conversation_meta = conversation_meta_by_id.get(conversation_id, {})
            subreddit = _string_or_empty(
                conversation_meta.get("subreddit")
                or utterance_meta.get("subreddit")
                or default_subreddit
            )
            if not _subreddit_is_selected(
                subreddit,
                include_subreddits=include_filter,
                exclude_subreddits=exclude_filter,
            ):
                continue
            record = _ensure_thread_record(
                records,
                conversation_id,
                source_name=source_name,
                conversation_meta=conversation_meta,
                utterance_meta=utterance_meta,
                default_subreddit=default_subreddit,
            )

            utterance_id = _utterance_id(utterance)
            speaker_id = _speaker_id(utterance)
            reply_to = _reply_to(utterance)
            text = _utterance_text(utterance)
            is_root = not reply_to or utterance_id == conversation_id
            root_title = _string_or_empty(conversation_meta.get("title") or utterance_meta.get("title"))
            root_op_text = _compose_root_op_text(root_title, text) if is_root else ""

            if is_root:
                record.root_id = utterance_id or conversation_id
                record.root_speaker_id = speaker_id

            unusable_speaker = is_unusable_speaker_id(speaker_id)
            unusable_text = is_unusable_text(text)
            if is_root and root_op_text:
                unusable_speaker = False
                unusable_text = False
            if unusable_speaker:
                stats.skipped_unusable_speaker += 1
            if unusable_text:
                stats.skipped_unusable_text += 1
            if unusable_speaker or unusable_text:
                if is_root:
                    record.unusable_root = True
                else:
                    parent_id = "" if not reply_to or reply_to == conversation_id else reply_to
                    record.comments.append(
                        _make_raw_comment(
                            speaker_id=speaker_id,
                            text=text,
                            timestamp=_coerce_timestamp(_get_value(utterance, "timestamp", default=0.0)),
                            comment_id=utterance_id,
                            parent_id=parent_id,
                            context_only=True,
                        )
                    )
                continue

            stats.kept_utterances += 1
            if is_root:
                record.op_text = root_op_text
                continue

            parent_id = "" if not reply_to or reply_to == conversation_id else reply_to
            record.comments.append(
                _make_raw_comment(
                    speaker_id=speaker_id,
                    text=text,
                    timestamp=_coerce_timestamp(_get_value(utterance, "timestamp", default=0.0)),
                    comment_id=utterance_id,
                    parent_id=parent_id,
                )
            )

    stats.total_conversations = len(records)
    return records, stats


def extract_user_profiles_from_thread_records(
    records: dict[str, _ThreadRecord],
    *,
    stats: ConvokitSubredditStats | None = None,
    min_conversations: int = DEFAULT_MIN_CONVERSATIONS,
    max_users: int | None = None,
    source_name: str = DEFAULT_SOURCE_NAME,
    thread_comment_retention: str = DEFAULT_THREAD_COMMENT_RETENTION,
) -> list[UserProfile]:
    """Build eligible user profiles."""
    run_stats = stats or ConvokitSubredditStats()
    usable_records = {
        post_id: record
        for post_id, record in records.items()
        if not record.unusable_root and record.op_text.strip() and record.comments
    }
    run_stats.skipped_unusable_root = sum(1 for record in records.values() if record.unusable_root)
    run_stats.usable_conversations = len(usable_records)

    user_thread_ts: dict[str, dict[str, float]] = {}
    for post_id, record in usable_records.items():
        for comment in record.comments:
            if comment.context_only:
                continue
            if comment.user_id == record.root_speaker_id:
                continue
            previous = user_thread_ts.setdefault(comment.user_id, {}).get(post_id)
            if previous is None or comment.timestamp < previous:
                user_thread_ts[comment.user_id][post_id] = comment.timestamp

    run_stats.all_user_conversation_counts = {
        user_id: len(ts_map)
        for user_id, ts_map in sorted(user_thread_ts.items())
    }

    eligible_user_ids = [
        user_id
        for user_id, ts_map in sorted(user_thread_ts.items())
        if len(ts_map) >= min_conversations
    ]
    run_stats.eligible_users = len(eligible_user_ids)
    if max_users is not None:
        eligible_user_ids = eligible_user_ids[:max_users]
    run_stats.selected_users = len(eligible_user_ids)
    run_stats.selected_user_ids = eligible_user_ids

    profiles: list[UserProfile] = []
    for user_id in eligible_user_ids:
        thread_ids = [
            post_id
            for post_id, _ in sorted(
                user_thread_ts[user_id].items(),
                key=lambda item: (item[1], item[0]),
            )
        ]
        threads = [
            _build_thread(usable_records[post_id], user_id)
            for post_id in thread_ids
        ]
        threads = _apply_thread_comment_retention(
            threads,
            thread_comment_retention=thread_comment_retention,
        )
        threads.sort(
            key=lambda thread: (
                _thread_sort_timestamp(usable_records[thread.post_id], user_id),
                thread.post_id,
            )
        )
        if len(threads) < min_conversations:
            continue
        profiles.append(
            UserProfile(
                user_id=user_id,
                raw_user_id=user_id,
                profile_id=user_id,
                train_threads=threads,
                source_name=source_name,
            )
        )

    run_stats.selected_users = len(profiles)
    run_stats.selected_user_ids = [profile.user_id for profile in profiles]
    return profiles


def profiles_from_convokit_corpora(
    corpora: Iterable[Any],
    *,
    corpus_names: Iterable[str] | None = None,
    include_subreddits: Iterable[str] | None = None,
    exclude_subreddits: Iterable[str] | None = None,
    min_conversations: int = DEFAULT_MIN_CONVERSATIONS,
    max_users: int | None = None,
    source_name: str = DEFAULT_SOURCE_NAME,
    thread_comment_retention: str = DEFAULT_THREAD_COMMENT_RETENTION,
) -> tuple[list[UserProfile], ConvokitSubredditStats]:
    """Convert subreddit corpora into profiles."""
    records, stats = build_thread_records_from_convokit_corpora(
        corpora,
        corpus_names=corpus_names,
        include_subreddits=include_subreddits,
        exclude_subreddits=exclude_subreddits,
        source_name=source_name,
    )
    profiles = extract_user_profiles_from_thread_records(
        records,
        stats=stats,
        min_conversations=min_conversations,
        max_users=max_users,
        source_name=source_name,
        thread_comment_retention=thread_comment_retention,
    )
    return profiles, stats


def profiles_from_convokit_corpus_zips(
    zip_paths: Iterable[str | Path],
    *,
    corpus_names: Iterable[str] | None = None,
    include_subreddits: Iterable[str] | None = None,
    exclude_subreddits: Iterable[str] | None = None,
    min_conversations: int = DEFAULT_MIN_CONVERSATIONS,
    max_users: int | None = None,
    source_name: str = DEFAULT_SOURCE_NAME,
    thread_comment_retention: str = DEFAULT_THREAD_COMMENT_RETENTION,
) -> tuple[list[UserProfile], ConvokitSubredditStats]:
    """Convert ConvoKit archives into profiles."""
    paths = [Path(path) for path in zip_paths]
    if not paths:
        raise ValueError("at least one ConvoKit corpus zip path is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"ConvoKit corpus zip not found: {path}")

    corpus_name_list = list(corpus_names) if corpus_names is not None else [path.name for path in paths]
    if len(corpus_name_list) != len(paths):
        raise ValueError("corpus_names length must match zip_paths length")

    stats = ConvokitSubredditStats(corpus_names=corpus_name_list)
    include_filter = _subreddit_filter_set(include_subreddits)
    exclude_filter = _subreddit_filter_set(exclude_subreddits)
    user_thread_ts: dict[str, dict[str, float]] = {}
    conversation_subreddits: dict[str, str] = {}
    conversation_meta_by_id: dict[str, dict[str, Any]] = {}
    unusable_root_ids: set[str] = set()
    subreddits: set[str] = set()
    zip_pairs = list(zip(paths, corpus_name_list))
    zip_workers = _convokit_zip_max_workers(len(zip_pairs))
    print(f"ConvoKit zip scan phase 1: {len(zip_pairs)} zip(s) across {zip_workers} worker(s)")
    with ProcessPoolExecutor(max_workers=zip_workers, mp_context=_convokit_zip_mp_context()) as executor:
        first_pass_results = list(
            executor.map(
                partial(
                    _scan_convokit_zip_first_pass_mp,
                    include_subreddits=include_filter,
                    exclude_subreddits=exclude_filter,
                ),
                (pair[0] for pair in zip_pairs),
                (pair[1] for pair in zip_pairs),
            )
        )

    for result in first_pass_results:
        stats.total_utterances += result.total_utterances
        stats.kept_utterances += result.kept_utterances
        stats.skipped_missing_conversation += result.skipped_missing_conversation
        stats.skipped_unusable_speaker += result.skipped_unusable_speaker
        stats.skipped_unusable_text += result.skipped_unusable_text
        unusable_root_ids.update(result.unusable_root_ids)
        subreddits.update(result.subreddits)
        _merge_user_thread_ts(user_thread_ts, result.user_thread_ts)
        for conversation_id, meta in result.conversation_meta_by_id.items():
            existing = conversation_meta_by_id.get(conversation_id, {})
            merged = dict(existing)
            for key, value in meta.items():
                if value or key not in merged:
                    merged[key] = value
            conversation_meta_by_id[conversation_id] = merged
        for conversation_id, subreddit in result.conversation_subreddits.items():
            conversation_subreddits.setdefault(conversation_id, subreddit)

    stats.all_user_conversation_counts = {
        user_id: len(ts_map)
        for user_id, ts_map in sorted(user_thread_ts.items())
    }
    eligible_user_ids = [
        user_id
        for user_id, ts_map in sorted(user_thread_ts.items())
        if len(ts_map) >= min_conversations
    ]
    stats.eligible_users = len(eligible_user_ids)
    if max_users is not None:
        eligible_user_ids = eligible_user_ids[:max_users]
    selected_user_ids = set(eligible_user_ids)
    selected_conversation_ids = {
        conversation_id
        for user_id in selected_user_ids
        for conversation_id in user_thread_ts[user_id]
    }

    records: dict[str, _ThreadRecord] = {}
    print(f"ConvoKit zip scan phase 2: {len(zip_pairs)} zip(s) across {zip_workers} worker(s)")
    with ProcessPoolExecutor(max_workers=zip_workers, mp_context=_convokit_zip_mp_context()) as executor:
        second_pass_results = list(
            executor.map(
                partial(
                    _scan_convokit_zip_second_pass_mp,
                    selected_conversation_ids=selected_conversation_ids,
                    conversation_meta_by_id=conversation_meta_by_id,
                    conversation_subreddits=conversation_subreddits,
                    source_name=source_name,
                ),
                (pair[0] for pair in zip_pairs),
                (pair[1] for pair in zip_pairs),
            )
        )

    for result in second_pass_results:
        for conversation_id, partial_record in result.records.items():
            existing = records.get(conversation_id)
            if existing is None:
                records[conversation_id] = partial_record
            else:
                _merge_thread_record(existing, partial_record)

    for conversation_id, record in records.items():
        if not record.op_text.strip() and conversation_id not in unusable_root_ids:
            record.op_text = _fallback_op_text(conversation_id)
    stats.total_conversations = len(conversation_meta_by_id) or len(conversation_subreddits) or len(records)
    stats.usable_conversations = sum(
        1 for record in records.values()
        if not record.unusable_root and record.op_text.strip() and record.comments
    )
    stats.skipped_unusable_root = sum(1 for record in records.values() if record.unusable_root)
    stats.missing_op_text_conversations = sum(
        1 for record in records.values()
        if record.op_text.startswith("[Original post text unavailable")
    )
    stats.subreddit_count = len(subreddits)

    profiles: list[UserProfile] = []
    usable_records = {
        post_id: record
        for post_id, record in records.items()
        if not record.unusable_root and record.op_text.strip() and record.comments
    }
    for user_id in eligible_user_ids:
        thread_ids = [
            post_id
            for post_id, _ in sorted(
                user_thread_ts[user_id].items(),
                key=lambda item: (item[1], item[0]),
            )
            if post_id in usable_records
        ]
        threads = [_build_thread(usable_records[post_id], user_id) for post_id in thread_ids]
        threads = _apply_thread_comment_retention(
            threads,
            thread_comment_retention=thread_comment_retention,
        )
        if len(threads) < min_conversations:
            continue
        profiles.append(
            UserProfile(
                user_id=user_id,
                raw_user_id=user_id,
                profile_id=user_id,
                train_threads=threads,
                source_name=source_name,
            )
        )

    stats.selected_users = len(profiles)
    stats.selected_user_ids = [profile.user_id for profile in profiles]
    return profiles, stats


def load_op_contexts_from_convokit_corpus_zips(
    zip_paths: Iterable[str | Path],
    *,
    conversation_ids: Iterable[str],
    corpus_names: Iterable[str] | None = None,
    include_subreddits: Iterable[str] | None = None,
    exclude_subreddits: Iterable[str] | None = None,
) -> dict[str, _OpContext]:
    """Load root OP metadata from official zips."""
    target_ids = {_string_or_empty(conversation_id) for conversation_id in conversation_ids}
    target_ids.discard("")
    if not target_ids:
        return {}

    paths = [Path(path) for path in zip_paths]
    corpus_name_list = list(corpus_names) if corpus_names is not None else [path.name for path in paths]
    if len(corpus_name_list) != len(paths):
        raise ValueError("corpus_names length must match zip_paths length")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"ConvoKit corpus zip not found: {path}")

    include_filter = _subreddit_filter_set(include_subreddits)
    exclude_filter = _subreddit_filter_set(exclude_subreddits)
    contexts: dict[str, _OpContext] = {}
    remaining = set(target_ids)

    for path, corpus_name in zip(paths, corpus_name_list):
        if not remaining:
            break
        default_subreddit = subreddit_from_corpus_name(corpus_name)
        with zipfile.ZipFile(path) as zip_file:
            conversation_meta_by_id = _load_zip_conversation_meta_subset(
                zip_file,
                remaining,
                default_subreddit=default_subreddit,
            )
            for conversation_id in list(remaining):
                meta = conversation_meta_by_id.get(conversation_id)
                if not meta:
                    continue
                subreddit = _string_or_empty(meta.get("subreddit") or default_subreddit)
                if not _subreddit_is_selected(
                    subreddit,
                    include_subreddits=include_filter,
                    exclude_subreddits=exclude_filter,
                ):
                    continue
                title = _string_or_empty(meta.get("title"))
                if title:
                    contexts[conversation_id] = _OpContext(
                        title=title,
                        op_text=_compose_root_op_text(title, ""),
                        subreddit=subreddit,
                    )

            for utterance in _iter_zip_utterance_dicts(zip_file):
                conversation_id = _conversation_id_from_utterance_dict(utterance)
                if conversation_id not in remaining:
                    continue
                utterance_id = _utterance_id_from_dict(utterance)
                reply_to = _reply_to_from_utterance_dict(utterance)
                is_root = not reply_to or utterance_id == conversation_id
                if not is_root:
                    continue

                meta = _utterance_dict_meta(utterance)
                conversation_meta = conversation_meta_by_id.get(conversation_id, {})
                subreddit = _string_or_empty(
                    meta.get("subreddit")
                    or conversation_meta.get("subreddit")
                    or default_subreddit
                )
                if not _subreddit_is_selected(
                    subreddit,
                    include_subreddits=include_filter,
                    exclude_subreddits=exclude_filter,
                ):
                    continue

                title = _string_or_empty(conversation_meta.get("title") or meta.get("title"))
                text = _string_or_empty(utterance.get("text"))
                op_text = _compose_root_op_text(title, text)
                if not op_text:
                    continue

                contexts[conversation_id] = _OpContext(
                    title=title,
                    op_text=op_text,
                    root_id=utterance_id or conversation_id,
                    root_speaker_id=_speaker_id_from_utterance_dict(utterance),
                    subreddit=subreddit,
                )
                remaining.discard(conversation_id)

    return contexts


def profiles_from_convokit_corpus_zip(
    zip_path: str | Path,
    *,
    corpus_name: str | None = None,
    include_subreddits: Iterable[str] | None = None,
    exclude_subreddits: Iterable[str] | None = None,
    op_context_zip_paths: Iterable[str | Path] | None = None,
    op_context_corpus_names: Iterable[str] | None = None,
    min_conversations: int = DEFAULT_MIN_CONVERSATIONS,
    max_users: int | None = None,
    source_name: str = DEFAULT_SOURCE_NAME,
    thread_comment_retention: str = DEFAULT_THREAD_COMMENT_RETENTION,
) -> tuple[list[UserProfile], ConvokitSubredditStats]:
    """Convert one ConvoKit archive into profiles."""
    path = Path(zip_path)
    if not path.is_file():
        raise FileNotFoundError(f"ConvoKit corpus zip not found: {path}")

    stats = ConvokitSubredditStats(corpus_names=[corpus_name or path.name])
    include_filter = _subreddit_filter_set(include_subreddits)
    exclude_filter = _subreddit_filter_set(exclude_subreddits)
    user_thread_ts: dict[str, dict[str, float]] = {}
    conversation_subreddits: dict[str, str] = {}
    conversation_meta_by_id: dict[str, dict[str, Any]] = {}
    unusable_root_ids: set[str] = set()
    subreddits: set[str] = set()

    with zipfile.ZipFile(path) as zip_file:
        conversation_meta_by_id = _load_zip_conversation_meta(zip_file)
        for conversation_id, meta in conversation_meta_by_id.items():
            subreddit = _string_or_empty(meta.get("subreddit"))
            if subreddit and _subreddit_is_selected(
                subreddit,
                include_subreddits=include_filter,
                exclude_subreddits=exclude_filter,
            ):
                conversation_subreddits[conversation_id] = subreddit
                subreddits.add(subreddit)

        for utterance in _iter_zip_utterance_dicts(zip_file):
            stats.total_utterances += 1
            utterance_id = _utterance_id_from_dict(utterance)
            conversation_id = _conversation_id_from_utterance_dict(utterance)
            reply_to = _reply_to_from_utterance_dict(utterance)
            speaker_id = _speaker_id_from_utterance_dict(utterance)
            text = _string_or_empty(utterance.get("text"))
            timestamp = _coerce_timestamp(utterance.get("timestamp"))
            meta = _utterance_dict_meta(utterance)
            conversation_meta = conversation_meta_by_id.get(conversation_id, {})
            subreddit = _string_or_empty(
                meta.get("subreddit")
                or conversation_meta.get("subreddit")
            )
            if not _subreddit_is_selected(
                subreddit,
                include_subreddits=include_filter,
                exclude_subreddits=exclude_filter,
            ):
                continue
            if subreddit:
                conversation_subreddits.setdefault(conversation_id, subreddit)
                subreddits.add(subreddit)

            if not conversation_id:
                stats.skipped_missing_conversation += 1
                continue

            is_root = not reply_to or utterance_id == conversation_id
            root_title = _string_or_empty(conversation_meta.get("title") or meta.get("title"))
            root_op_text = _compose_root_op_text(root_title, text) if is_root else ""
            unusable_speaker = is_unusable_speaker_id(speaker_id)
            unusable_text = is_unusable_text(text)
            if is_root and root_op_text:
                unusable_speaker = False
                unusable_text = False
            if unusable_speaker:
                stats.skipped_unusable_speaker += 1
            if unusable_text:
                stats.skipped_unusable_text += 1
            if unusable_speaker or unusable_text:
                if is_root:
                    unusable_root_ids.add(conversation_id)
                continue

            stats.kept_utterances += 1
            if is_root:
                continue

            previous = user_thread_ts.setdefault(speaker_id, {}).get(conversation_id)
            if previous is None or timestamp < previous:
                user_thread_ts[speaker_id][conversation_id] = timestamp

    stats.all_user_conversation_counts = {
        user_id: len(ts_map)
        for user_id, ts_map in sorted(user_thread_ts.items())
    }
    eligible_user_ids = [
        user_id
        for user_id, ts_map in sorted(user_thread_ts.items())
        if len(ts_map) >= min_conversations
    ]
    stats.eligible_users = len(eligible_user_ids)
    if max_users is not None:
        eligible_user_ids = eligible_user_ids[:max_users]
    selected_user_ids = set(eligible_user_ids)
    selected_conversation_ids = {
        conversation_id
        for user_id in selected_user_ids
        for conversation_id in user_thread_ts[user_id]
    }
    op_context_by_id: dict[str, _OpContext] = {}
    if op_context_zip_paths is not None:
        op_context_by_id = load_op_contexts_from_convokit_corpus_zips(
            op_context_zip_paths,
            conversation_ids=selected_conversation_ids,
            corpus_names=op_context_corpus_names,
            include_subreddits=include_subreddits,
            exclude_subreddits=exclude_subreddits,
        )

    records: dict[str, _ThreadRecord] = {}
    with zipfile.ZipFile(path) as zip_file:
        for utterance in _iter_zip_utterance_dicts(zip_file):
            conversation_id = _conversation_id_from_utterance_dict(utterance)
            if conversation_id not in selected_conversation_ids:
                continue

            utterance_id = _utterance_id_from_dict(utterance)
            reply_to = _reply_to_from_utterance_dict(utterance)
            speaker_id = _speaker_id_from_utterance_dict(utterance)
            text = _string_or_empty(utterance.get("text"))
            meta = _utterance_dict_meta(utterance)
            conversation_meta = conversation_meta_by_id.get(conversation_id, {})
            subreddit = _string_or_empty(
                meta.get("subreddit")
                or conversation_meta.get("subreddit")
                or conversation_subreddits.get(conversation_id)
            )
            op_context = op_context_by_id.get(conversation_id)
            title = _string_or_empty(
                conversation_meta.get("title")
                or meta.get("title")
                or (op_context.title if op_context else "")
            )
            if op_context and op_context.subreddit and not subreddit:
                subreddit = op_context.subreddit
            record = records.get(conversation_id)
            if record is None:
                record = _ThreadRecord(
                    post_id=conversation_id,
                    source_name=source_name,
                    subreddit=subreddit,
                    title=title,
                )
                records[conversation_id] = record
            else:
                if subreddit and not record.subreddit:
                    record.subreddit = subreddit
                if title and not record.title:
                    record.title = title
            if op_context:
                if not record.op_text:
                    record.op_text = op_context.op_text
                if op_context.root_id and not record.root_id:
                    record.root_id = op_context.root_id
                if op_context.root_speaker_id and not record.root_speaker_id:
                    record.root_speaker_id = op_context.root_speaker_id

            is_root = not reply_to or utterance_id == conversation_id
            root_op_text = _compose_root_op_text(title, text) if is_root else ""
            unusable_speaker = is_unusable_speaker_id(speaker_id)
            unusable_text = is_unusable_text(text)
            if is_root and root_op_text:
                unusable_speaker = False
                unusable_text = False
            if unusable_speaker or unusable_text:
                if is_root:
                    record.unusable_root = True
                else:
                    parent_id = "" if not reply_to or reply_to == conversation_id else reply_to
                    record.comments.append(
                        _make_raw_comment(
                            speaker_id=speaker_id,
                            text=text,
                            timestamp=_coerce_timestamp(utterance.get("timestamp")),
                            comment_id=utterance_id,
                            parent_id=parent_id,
                            context_only=True,
                        )
                    )
                continue

            if is_root:
                record.root_id = utterance_id or conversation_id
                record.root_speaker_id = speaker_id
                record.op_text = root_op_text
                continue

            parent_id = "" if not reply_to or reply_to == conversation_id else reply_to
            record.comments.append(
                _make_raw_comment(
                    speaker_id=speaker_id,
                    text=text,
                    timestamp=_coerce_timestamp(utterance.get("timestamp")),
                    comment_id=utterance_id,
                    parent_id=parent_id,
                )
            )

    for conversation_id, record in records.items():
        if not record.op_text.strip() and conversation_id not in unusable_root_ids:
            record.op_text = _fallback_op_text(conversation_id)
    stats.total_conversations = len(conversation_meta_by_id) or len(conversation_subreddits) or len(records)
    stats.usable_conversations = sum(
        1 for record in records.values()
        if not record.unusable_root and record.op_text.strip() and record.comments
    )
    stats.skipped_unusable_root = sum(1 for record in records.values() if record.unusable_root)
    stats.missing_op_text_conversations = sum(
        1 for record in records.values()
        if record.op_text.startswith("[Original post text unavailable")
    )
    stats.subreddit_count = len(subreddits)

    profiles: list[UserProfile] = []
    usable_records = {
        post_id: record
        for post_id, record in records.items()
        if not record.unusable_root and record.op_text.strip() and record.comments
    }
    for user_id in eligible_user_ids:
        thread_ids = [
            post_id
            for post_id, _ in sorted(
                user_thread_ts[user_id].items(),
                key=lambda item: (item[1], item[0]),
            )
            if post_id in usable_records
        ]
        threads = [_build_thread(usable_records[post_id], user_id) for post_id in thread_ids]
        threads = _apply_thread_comment_retention(
            threads,
            thread_comment_retention=thread_comment_retention,
        )
        if len(threads) < min_conversations:
            continue
        profiles.append(
            UserProfile(
                user_id=user_id,
                raw_user_id=user_id,
                profile_id=user_id,
                train_threads=threads,
                source_name=source_name,
            )
        )

    stats.selected_users = len(profiles)
    stats.selected_user_ids = [profile.user_id for profile in profiles]
    return profiles, stats
