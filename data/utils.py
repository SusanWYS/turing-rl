"""Shared Reddit thread data structures."""

import copy
import html
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


PUBLIC_DATA_MODES = ["reasoning"]


def shuffle_parquet_rows(output_path: str, *, seed: int) -> None:
    """Rewrite a parquet file with a deterministic row shuffle."""
    path = Path(output_path)
    if not path.is_file():
        return

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if table.num_rows <= 1:
        return

    indices = list(range(table.num_rows))
    random.Random(seed).shuffle(indices)
    shuffled = table.take(pa.array(indices, type=pa.int64()))
    pq.write_table(shuffled, path)


DEFAULT_MAX_WORDS = 2048
DEFAULT_PERSONA_CONTEXT_INTRO = (
    "Below are examples of [HUMAN]'s responses to Reddit threads. "
    "These examples may come from multiple subreddits."
)
_INDENTED_SPEAKER_RE = re.compile(r"^[ \t]+(\[(?:HUMAN|OTHER[^\]]*)\]:)")
_QUOTE_PREFIX_RE = r"(?:(?:&gt;|>)\s*)+"
_SPEAKER_QUOTE_RE = re.compile(rf"^(\[(?:HUMAN|OTHER[^\]]*)\]:)\s*{_QUOTE_PREFIX_RE}(.*)$")
_LINE_QUOTE_RE = re.compile(rf"^{_QUOTE_PREFIX_RE}(.*)$")
_ESCAPED_HTML_ENTITY_RE = re.compile(r"\\(?=&(?:gt|lt|amp|quot|apos);)")


def _coerce_text(value: Any) -> str:
    """Extract text from chat-shaped content."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_coerce_text(item).strip() for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content"):
            if key in value:
                return _coerce_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def unescape_reddit_html_entities(value: Any) -> str:
    """Decode Reddit/HTML entities."""
    text = _coerce_text(value)
    if not text:
        return ""
    text = _ESCAPED_HTML_ENTITY_RE.sub("", text)
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return text


def normalize_reddit_reply_artifacts(value: Any) -> str:
    """Remove Reddit quote and indentation artifacts."""
    text = unescape_reddit_html_entities(value)
    if not text:
        return ""

    normalized_lines: list[str] = []
    for raw_line in text.splitlines():
        line = _INDENTED_SPEAKER_RE.sub(r"\1", raw_line)
        speaker_match = _SPEAKER_QUOTE_RE.match(line)
        if speaker_match is not None:
            speaker, remainder = speaker_match.groups()
            line = speaker if not remainder else f"{speaker} {remainder}"
        else:
            line_quote_match = _LINE_QUOTE_RE.match(line)
            if line_quote_match is not None:
                line = line_quote_match.group(1)
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


@dataclass
class Comment:
    """One Reddit comment."""
    text: str
    user_id: str
    timestamp: float
    turn_id: int
    is_target_user: bool = False
    comment_id: str = ""
    parent_id: str = ""  # empty = top-level (reply to OP)
    display_label: str = ""


@dataclass
class Thread:
    """One Reddit thread."""
    post_id: str
    op_text: str
    comments: list[Comment] = field(default_factory=list)
    target_user_comment_indices: list[int] = field(default_factory=list)
    comment_id_to_idx: dict = field(default_factory=dict)
    source_name: str = ""
    subreddit: str = ""

    def source_label(self) -> str:
        """Return a display source label."""
        label = (self.subreddit or "").strip()
        if label:
            if label.lower().startswith("r/"):
                return label
            return f"r/{label}"
        return (self.source_name or "").strip()

    def _format_source_header(self) -> list[str]:
        return []

    def get_reply_chain(self, target_idx: int) -> list[int]:
        """Return ancestor indices from root to target."""
        chain = [target_idx]
        current = self.comments[target_idx]
        while current.parent_id and current.parent_id in self.comment_id_to_idx:
            parent_idx = self.comment_id_to_idx[current.parent_id]
            chain.append(parent_idx)
            current = self.comments[parent_idx]
        chain.reverse()
        return chain

    def trim_to_target_reply_chain(self, target_idx: Optional[int] = None) -> "Thread":
        """Keep only the target reply chain."""
        if not self.target_user_comment_indices:
            return copy.deepcopy(self)

        if target_idx is None:
            target_idx = self.target_user_comment_indices[-1]
        if target_idx < 0 or target_idx >= len(self.comments):
            raise IndexError(f"target_idx {target_idx} out of range for {len(self.comments)} comments")

        keep_old_indices = set(self.get_reply_chain(target_idx))
        ordered_old_indices = [idx for idx in range(len(self.comments)) if idx in keep_old_indices]
        comments: list[Comment] = []
        comment_id_to_idx: dict[str, int] = {}
        target_indices: list[int] = []
        for new_idx, old_idx in enumerate(ordered_old_indices):
            comment = self.comments[old_idx]
            comments.append(
                Comment(
                    text=comment.text,
                    user_id=comment.user_id,
                    timestamp=comment.timestamp,
                    turn_id=new_idx,
                    is_target_user=comment.is_target_user,
                    comment_id=comment.comment_id,
                    parent_id=comment.parent_id,
                    display_label=comment.display_label,
                )
            )
            if comment.comment_id:
                comment_id_to_idx[comment.comment_id] = new_idx
            if old_idx == target_idx and comment.is_target_user:
                target_indices.append(new_idx)

        return Thread(
            post_id=self.post_id,
            op_text=self.op_text,
            comments=comments,
            target_user_comment_indices=target_indices,
            comment_id_to_idx=comment_id_to_idx,
            source_name=self.source_name,
            subreddit=self.subreddit,
        )

    def _format_comment(self, idx: int, depth: int = 0,
                        max_words: Optional[int] = None,
                        bold_human: bool = False) -> str:
        """Format one labeled comment."""
        comment = self.comments[idx]
        text = normalize_reddit_reply_artifacts(comment.text)
        if max_words:
            words = text.split()
            if len(words) > max_words:
                text = " ".join(words[:max_words]) + "..."
        if comment.is_target_user:
            return f"[HUMAN]: {text}"
        label = comment.display_label or "OTHER"
        return f"[{label}]: {text}"

    def _format_reply_chains(
        self, human_indices: list[int],
        target_parent_idx: Optional[int] = None,
        max_words: Optional[int] = None,
        bold_human: bool = False,
    ) -> str:
        """Format deduplicated reply chains."""
        shown = set()
        blocks = []

        all_chains = [self.get_reply_chain(idx) for idx in human_indices]

        if target_parent_idx is not None:
            target_chain = self.get_reply_chain(target_parent_idx)
            if len(target_chain) > 1:
                all_chains.append(target_chain[:-1])
            elif target_chain:
                comment = self.comments[target_parent_idx]
                if comment.parent_id and comment.parent_id not in self.comment_id_to_idx:
                    blocks.append("[... earlier comments by other users ...]")

        for chain in all_chains:
            chain_lines = []
            root_comment = self.comments[chain[0]]
            if root_comment.parent_id and root_comment.parent_id not in self.comment_id_to_idx:
                if chain[0] not in shown:
                    chain_lines.append("[... earlier comments by other users ...]")

            for depth, c_idx in enumerate(chain):
                if c_idx in shown:
                    continue
                shown.add(c_idx)
                chain_lines.append(self._format_comment(c_idx, depth, max_words, bold_human))

            if chain_lines:
                blocks.append("\n".join(chain_lines))

        return "\n\n".join(blocks)

    def _format_op_and_chains(self, target_idx: int,
                              max_words: Optional[int] = None,
                              bold_human: bool = False) -> str:
        """Format OP plus relevant reply chains."""
        lines = self._format_source_header()
        lines.extend(["[OTHER - OP]", normalize_reddit_reply_artifacts(self.op_text)])

        earlier_indices = [i for i in self.target_user_comment_indices if i < target_idx]
        chains = self._format_reply_chains(
            earlier_indices, target_parent_idx=target_idx, max_words=max_words,
            bold_human=bold_human,
        )
        if chains:
            lines.append("")
            lines.append(chains)

        return "\n".join(lines)

    def format_context_for_persona(
        self, target_idx: int, max_words: Optional[int] = None,
    ) -> tuple[str, str]:
        """Return persona context and response."""
        context = self._format_op_and_chains(target_idx, max_words=max_words)
        response = normalize_reddit_reply_artifacts(self.comments[target_idx].text)
        if max_words is not None:
            words = response.split()
            if len(words) > max_words:
                response = " ".join(words[:max_words])
        return context, response

    def format_for_eval(self, target_idx: int) -> str:
        """Format eval context."""
        return self._format_op_and_chains(target_idx) + "\n\n[HUMAN]: "


@dataclass
class UserProfile:
    """One user's train/dev/test split."""
    user_id: str
    train_threads: list[Thread] = field(default_factory=list)
    dev_threads: list[Thread] = field(default_factory=list)
    test_threads: list[Thread] = field(default_factory=list)
    source_name: str = ""
    raw_user_id: str = ""
    profile_id: str = ""

    def __post_init__(self) -> None:
        if not self.profile_id:
            self.profile_id = self.user_id
        if not self.raw_user_id:
            self.raw_user_id = self.user_id

    def format_train_context_for_persona(self, max_words: Optional[int] = 1024) -> str:
        """Format persona-induction history."""
        lines = [
            DEFAULT_PERSONA_CONTEXT_INTRO,
            "[HUMAN]'s messages are enclosed in ** ** for emphasis.",
        ]

        example_num = 0
        for thread in self.train_threads:
            if not thread.target_user_comment_indices:
                continue
            last_idx = thread.target_user_comment_indices[-1]
            context, response = thread.format_context_for_persona(last_idx, max_words)
            example_num += 1
            lines.append(f"\n<|Start of Example {example_num}|>")
            lines.append(context)
            lines.append("")
            lines.append(f"**[HUMAN]'s Response: {response}**")
            lines.append(f"<|End of Example {example_num}|>")

        return "\n".join(lines)

    def format_history_context(
        self, threads: list["Thread"], max_words: Optional[int] = DEFAULT_MAX_WORDS,
    ) -> str:
        """Format historical context."""
        lines = [
            "[USER HISTORY]",
            "Below are [HUMAN]'s previous messages.",
        ]

        for i, thread in enumerate(threads, 1):
            if not thread.target_user_comment_indices:
                continue

            lines.append(f"\n<Post {i}>")
            lines.append(f"[OTHER - OP]\n{_coerce_text(thread.op_text)}")

            chains = thread._format_reply_chains(
                thread.target_user_comment_indices, max_words=max_words,
            )
            if chains:
                lines.append("")
                lines.append(chains)

            lines.append(f"</Post {i}>")

        return "\n".join(lines)
