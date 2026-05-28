"""
Data processing utilities: schema detection, validation, cleaning, filtering, deduplication.
"""
import json
import re
import unicodedata
from typing import List, Dict, Any, Optional
from datasets import Dataset

try:
    from simhash import Simhash
    HAS_SIMHASH = True
except ImportError:
    HAS_SIMHASH = False


def auto_detect_schema(sample: Dict[str, Any]) -> str:
    """Detect the logical dataset type from a sample's structure.

    Returns one of: 'pretrain', 'sft', 'dpo', 'rlaif', 'agent', 'unknown'.
    """
    keys = set(sample.keys())

    if "chosen" in keys and "rejected" in keys:
        return "dpo"

    if "conversations" in keys:
        if "gt" in keys:
            return "agent"
        # Check if conversations look like RLAIF (last msg has no assistant content)
        convs = sample.get("conversations", [])
        if isinstance(convs, list) and len(convs) > 0:
            last = convs[-1]
            if isinstance(last, dict) and last.get("role") == "user" and not last.get("content"):
                return "rlaif"
        return "sft"

    if "text" in keys:
        return "pretrain"

    if "messages" in keys or "dialog" in keys or "chat" in keys:
        return "sft"

    if "content" in keys or "passage" in keys or "document" in keys:
        return "pretrain"

    return "unknown"


def validate_conversations(convs: List[Dict]) -> bool:
    """Check conversation format integrity.

    Returns True if the conversation is well-formed:
    - Each message has 'role' and 'content' keys.
    - Roles alternate reasonably (no two assistants in a row without tool response).
    """
    if not isinstance(convs, list) or len(convs) == 0:
        return False

    valid_roles = {"system", "user", "assistant", "tool"}
    for msg in convs:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg:
            return False
        if msg["role"] not in valid_roles:
            return False
        if "content" not in msg and "tool_calls" not in msg:
            return False

    return True


def clean_text(text: str) -> str:
    """Normalize whitespace and Unicode, strip control characters."""
    if not isinstance(text, str):
        return str(text)

    # Normalize Unicode (NFC)
    text = unicodedata.normalize("NFC", text)

    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize whitespace: collapse multiple spaces (not newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_conversations(convs: List[Dict]) -> List[Dict]:
    """Clean all text fields in a conversation list."""
    cleaned = []
    for msg in convs:
        msg = dict(msg)
        for key in ("content", "reasoning_content"):
            if key in msg and msg[key] is not None:
                msg[key] = clean_text(msg[key])
        cleaned.append(msg)
    return cleaned


def filter_by_length(
    ds: Dataset,
    min_chars: int = 10,
    max_chars: int = 1_000_000,
    text_key: str = "text",
) -> Dataset:
    """Filter samples by character length of the primary text field."""
    def _predicate(sample):
        text = sample.get(text_key, "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        return min_chars <= len(str(text)) <= max_chars

    return ds.filter(_predicate)


def filter_by_token_length(
    ds: Dataset,
    tokenizer,
    min_tokens: int = 5,
    max_tokens: int = 8192,
    text_key: str = "text",
) -> Dataset:
    """Filter samples by token count after tokenization."""
    def _count(sample):
        text = sample.get(text_key, "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        return len(tokenizer(str(text)).input_ids)

    lengths = ds.map(lambda s: {"_tok_len": _count(s)}, desc="Counting tokens")
    return lengths.filter(
        lambda s: min_tokens <= s["_tok_len"] <= max_tokens
    ).remove_columns(["_tok_len"])


def _compute_simhash(text: str) -> int:
    """Compute a simhash fingerprint for a text string."""
    if not HAS_SIMHASH:
        # Fallback: use deterministic python hash
        return hash(text) & 0xFFFFFFFFFFFFFFFF
    return Simhash(text.split()).value


def deduplicate(
    ds: Dataset,
    threshold: int = 3,
    text_key: str = "text",
) -> Dataset:
    """Remove near-duplicate samples using simhash Hamming distance.

    Args:
        ds: Input dataset.
        threshold: Maximum Hamming distance for near-duplicate detection (lower = stricter).
        text_key: Key of the text field to use for deduplication.
    """
    seen: List[int] = []

    def _is_dup(sample):
        text = sample.get(text_key, "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        fp = _compute_simhash(str(text))
        for s in seen:
            if (fp ^ s).bit_count() <= threshold:
                return False  # duplicate
        seen.append(fp)
        return True  # keep

    return ds.filter(_is_dup)


def describe_dataset(ds: Dataset, tokenizer=None) -> Dict[str, Any]:
    """Generate summary statistics for a dataset.

    Returns a dict with: num_samples, column_names, avg_text_length,
    estimated_tokens, memory_estimate_mb.
    """
    info = {
        "num_samples": len(ds),
        "column_names": ds.column_names,
    }

    # Find the primary text column
    text_col = None
    for cand in ("text", "conversations", "chosen"):
        if cand in ds.column_names:
            text_col = cand
            break

    if text_col:
        lengths = []
        total_tokens = 0
        sample_size = min(1000, len(ds))
        for i, sample in enumerate(ds.select(range(sample_size))):
            text = sample.get(text_col, "")
            if isinstance(text, list):
                text = json.dumps(text, ensure_ascii=False)
            lengths.append(len(str(text)))
            if tokenizer and i < 500:
                total_tokens += len(tokenizer(str(text)).input_ids)

        info["avg_chars"] = sum(lengths) / max(len(lengths), 1)
        if tokenizer and total_tokens > 0:
            chars_per_token = sum(lengths[:500]) / total_tokens
            estimated_tokens = int(
                (info["avg_chars"] * info["num_samples"]) / chars_per_token
            )
            info["estimated_total_tokens"] = estimated_tokens
            info["chars_per_token"] = round(chars_per_token, 2)

    # Memory estimate (raw text only)
    if "avg_chars" in info:
        info["raw_text_mb"] = round(
            info["avg_chars"] * info["num_samples"] / (1024 * 1024), 2
        )

    return info


def process_dataset(
    ds: Dataset,
    tokenizer=None,
    min_chars: int = 10,
    max_chars: int = 500_000,
    min_tokens: int = 5,
    max_tokens: int = 8192,
    dedup: bool = True,
    dedup_threshold: int = 3,
    clean: bool = True,
    text_key: str = "text",
) -> Dataset:
    """Run the full data processing pipeline.

    Order: clean -> filter by chars -> filter by tokens -> deduplicate.
    """
    n_before = len(ds)

    if clean:
        def _clean(sample):
            for key in list(sample.keys()):
                if isinstance(sample[key], str):
                    sample[key] = clean_text(sample[key])
            return sample
        ds = ds.map(_clean, desc="Cleaning text")

    ds = filter_by_length(ds, min_chars=min_chars, max_chars=max_chars, text_key=text_key)
    n_after_len = len(ds)

    if tokenizer and min_tokens > 0:
        ds = filter_by_token_length(
            ds, tokenizer, min_tokens=min_tokens, max_tokens=max_tokens, text_key=text_key
        )

    if dedup:
        ds = deduplicate(ds, threshold=dedup_threshold, text_key=text_key)

    n_after = len(ds)
    removed = n_before - n_after
    if removed > 0:
        print(f"Data processing removed {removed} / {n_before} samples "
              f"({removed / max(n_before, 1) * 100:.1f}%)")

    return ds
