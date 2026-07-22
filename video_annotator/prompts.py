"""Prompt normalization for category-union and referring-expression modes."""
from __future__ import annotations

import re
from dataclasses import dataclass


_ARTICLES = {"a", "an", "the"}
_NON_CATEGORY_WORDS = {
    "above", "after", "ahead", "back", "behind", "below", "between", "front",
    "left", "moving", "near", "next", "right", "running", "stationary", "touching",
    "under", "walking", "wearing", "with",
}


@dataclass(frozen=True)
class PromptSpec:
    raw: str
    mode: str
    targets: tuple[str, ...]
    detector_prompt: str
    motion_text: str | None = None


def _clean_target(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" \t\r\n.,;:!?"))
    words = [word for word in value.split() if word.lower() not in _ARTICLES and word.lower() != "only"]
    return " ".join(words)


def _is_simple_category(value: str) -> bool:
    words = value.lower().split()
    return bool(words) and len(words) <= 3 and not any(word in _NON_CATEGORY_WORDS for word in words)


def _category_targets(raw: str) -> tuple[str, ...]:
    text = re.sub(r"\s+", " ", raw.strip())
    # Periods and commas are explicit class-list separators in Grounding DINO.
    parts = [part for part in re.split(r"[,.，、;]+", text) if part.strip()]
    if len(parts) == 1 and re.search(r"\s+and\s+", parts[0], flags=re.IGNORECASE):
        left, right = re.split(r"\s+and\s+", parts[0], maxsplit=1, flags=re.IGNORECASE)
        if _is_simple_category(left) and _is_simple_category(right):
            parts = [left, right]
    targets: list[str] = []
    for part in parts:
        target = _clean_target(part)
        if target and target.lower() not in {item.lower() for item in targets}:
            targets.append(target)
    return tuple(targets)


def parse_prompt(raw: str, mode: str = "category", explicit_targets: list[str] | tuple[str, ...] | None = None) -> PromptSpec:
    """Create a normalized prompt specification without over-splitting prose."""
    raw = raw.strip()
    if not raw and not explicit_targets:
        raise ValueError("Prompt cannot be empty")
    if mode not in {"category", "referring"}:
        raise ValueError("mode must be 'category' or 'referring'")
    if explicit_targets:
        targets = tuple(dict.fromkeys(_clean_target(item) for item in explicit_targets if _clean_target(item)))
    elif mode == "category":
        targets = _category_targets(raw)
    else:
        targets = ()
    if mode == "category" and not targets:
        raise ValueError("No category targets found; use --target at least once or provide a category prompt")
    detector_prompt = " . ".join(targets) + " ." if targets else ""
    return PromptSpec(raw=raw, mode="category_union" if mode == "category" else "referring", targets=targets, detector_prompt=detector_prompt, motion_text=raw if mode == "referring" else None)
