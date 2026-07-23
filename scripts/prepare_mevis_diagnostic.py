"""Create a deterministic, fixed MeViS diagnostic manifest.

The selection is based only on manifest order and expression text. It never
looks at model output, so the diagnostic set cannot be tuned after evaluation.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_BUCKETS = {
    "multi_target": re.compile(r"\b(and|both|two|three|each|all)\b", re.I),
    "motion": re.compile(r"\b(run(?:ning)?|walk(?:ing)?|move|fly(?:ing)?|jump|turn|fall|swim|drive|left|right|ahead|behind)\b", re.I),
    "relation": re.compile(r"\b(near|next to|behind|in front|beside|touch|holding|carry|follow)\b", re.I),
    "temporal": re.compile(r"\b(then|after|before|first|last|start|later|comes|enter|leave)\b", re.I),
}


def _bucket(prompt: str) -> str:
    for name, pattern in _BUCKETS.items():
        if pattern.search(prompt):
            return name
    return "single_target"


def prepare(input_manifest: Path, output: Path, max_items: int = 10) -> dict:
    payload = json.loads(input_manifest.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    grouped: dict[str, list[dict]] = {}
    for index, case in enumerate(cases):
        prompt = str(case.get("prompt", "")).strip()
        if not prompt:
            continue
        item = dict(case)
        item["source_index"] = index
        item["diagnostic_bucket"] = _bucket(prompt)
        grouped.setdefault(item["diagnostic_bucket"], []).append(item)
    selected: list[dict] = []
    # First cover every available behavior bucket, then fill deterministically.
    for name in ("multi_target", "motion", "relation", "temporal", "single_target"):
        if grouped.get(name):
            selected.append(grouped[name].pop(0))
    remaining = [item for items in grouped.values() for item in items]
    remaining.sort(key=lambda item: int(item["source_index"]))
    selected.extend(remaining)
    selected = selected[:max_items]
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": payload.get("dataset", "mevis-valid-u"),
        "source_manifest": str(input_manifest),
        "case_count": len(selected),
        "selection_policy": "fixed source order with one case per available expression bucket before deterministic fill",
        "buckets": {name: sum(item["diagnostic_bucket"] == name for item in selected) for name in _BUCKETS} | {"single_target": sum(item["diagnostic_bucket"] == "single_target" for item in selected)},
        "candidate_recall_definition": "A ground-truth annotation is candidate_present when at least one cached predicted track overlaps it at IoU >= 0.10 on any annotated frame; otherwise candidate_missing. Selector misses are candidate_present_but_not_selected.",
        "cases": selected,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"cases": len(selected), "output": str(output), "buckets": report["buckets"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output, args.max_items), indent=2))


if __name__ == "__main__":
    main()
