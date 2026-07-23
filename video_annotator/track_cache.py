"""Versioned, prompt-scope-aware candidate track cache.

The cache deliberately stores metadata and references to mask files rather than
embedding full per-frame mask tensors in one JSON document. It lets several
expressions over the same compatible candidate scope share expensive tracking.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CACHE_SCHEMA_VERSION = 2


def scope_fingerprint(scope: list[str] | tuple[str, ...], proposal_prompts: list[str] | tuple[str, ...] = ()) -> str:
    payload = {
        "targets": sorted({" ".join(str(item).lower().split()) for item in scope if str(item).strip()}),
        "proposal_prompts": sorted({str(item).strip() for item in proposal_prompts if str(item).strip()}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TrackFrame:
    frame_index: int
    visible: bool
    box_xyxy: tuple[float, float, float, float] | None = None
    mask_path: str | None = None
    detector_score: float | None = None
    propagation_score: float | None = None
    center_xy: tuple[float, float] | None = None
    area: float | None = None


@dataclass(frozen=True)
class AnchorToken:
    frame_index: int
    timestamp: float
    position_xy: tuple[float, float]
    size_wh: tuple[float, float]
    velocity_xy: tuple[float, float]
    acceleration_xy: tuple[float, float]
    visibility: float
    confidence: float
    provenance: str
    relation_features: dict[str, float] = field(default_factory=dict)
    image_embedding: list[float] = field(default_factory=list)


@dataclass
class CachedTrack:
    track_id: int
    label: str
    detector_score: float
    frames: list[TrackFrame] = field(default_factory=list)
    appearance_features: list[list[float]] = field(default_factory=list)
    temporal_features: dict[str, float | list[float] | str] = field(default_factory=dict)
    relation_features: dict[str, Any] = field(default_factory=dict)
    anchor_tokens: list[AnchorToken] = field(default_factory=list)


@dataclass
class TrackCache:
    source: dict[str, Any]
    config: dict[str, Any]
    proposal_scope: dict[str, Any]
    tracks: list[CachedTrack] = field(default_factory=list)
    schema_version: int = CACHE_SCHEMA_VERSION

    @property
    def scope_fingerprint(self) -> str:
        return str(self.proposal_scope.get("fingerprint", ""))


def _frame_from_dict(item: dict[str, Any]) -> TrackFrame:
    box = item.get("box_xyxy")
    center = item.get("center_xy")
    return TrackFrame(
        frame_index=int(item["frame_index"]),
        visible=bool(item["visible"]),
        box_xyxy=tuple(float(value) for value in box) if box is not None else None,
        mask_path=str(item["mask_path"]) if item.get("mask_path") is not None else None,
        detector_score=float(item["detector_score"]) if item.get("detector_score") is not None else None,
        propagation_score=float(item["propagation_score"]) if item.get("propagation_score") is not None else None,
        center_xy=tuple(float(value) for value in center) if center is not None else None,
        area=float(item["area"]) if item.get("area") is not None else None,
    )


def _track_from_dict(item: dict[str, Any]) -> CachedTrack:
    return CachedTrack(
        track_id=int(item["track_id"]),
        label=str(item["label"]),
        detector_score=float(item.get("detector_score", 0.0)),
        frames=[_frame_from_dict(frame) for frame in item.get("frames", [])],
        appearance_features=[[float(value) for value in row] for row in item.get("appearance_features", [])],
        temporal_features=dict(item.get("temporal_features", {})),
        relation_features=dict(item.get("relation_features", {})),
        anchor_tokens=[AnchorToken(frame_index=int(token["frame_index"]), timestamp=float(token["timestamp"]), position_xy=tuple(float(value) for value in token["position_xy"]), size_wh=tuple(float(value) for value in token["size_wh"]), velocity_xy=tuple(float(value) for value in token["velocity_xy"]), acceleration_xy=tuple(float(value) for value in token["acceleration_xy"]), visibility=float(token["visibility"]), confidence=float(token["confidence"]), provenance=str(token["provenance"]), relation_features={str(key): float(value) for key, value in token.get("relation_features", {}).items()}, image_embedding=[float(value) for value in token.get("image_embedding", [])]) for token in item.get("anchor_tokens", [])],
    )


def _payload(cache: TrackCache) -> dict[str, Any]:
    return {"schema_version": cache.schema_version, "source": cache.source, "config": cache.config, "proposal_scope": cache.proposal_scope, "tracks": [asdict(track) for track in cache.tracks]}


def validate_cache(cache: TrackCache) -> None:
    if cache.schema_version != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported track-cache schema {cache.schema_version}; expected {CACHE_SCHEMA_VERSION}")
    if not cache.source.get("source_id"):
        raise ValueError("Track cache is missing source.source_id")
    if not cache.source.get("frame_count") or int(cache.source["frame_count"]) < 1:
        raise ValueError("Track cache must contain a positive source.frame_count")
    if not cache.scope_fingerprint:
        raise ValueError("Track cache is missing proposal_scope.fingerprint")
    ids = [track.track_id for track in cache.tracks]
    if len(ids) != len(set(ids)):
        raise ValueError("Track cache contains duplicate global track IDs")
    for track in cache.tracks:
        if track.track_id < 1 or not track.label.strip():
            raise ValueError("Every cached track needs a positive ID and non-empty label")
        for frame in track.frames:
            if frame.frame_index < 0 or frame.frame_index >= int(cache.source["frame_count"]):
                raise ValueError(f"Track {track.track_id} contains an out-of-range frame index")
            if frame.box_xyxy is not None and len(frame.box_xyxy) != 4:
                raise ValueError(f"Track {track.track_id} contains an invalid box")
        embedding_dims = {len(token.image_embedding) for token in track.anchor_tokens if token.image_embedding}
        if len(embedding_dims) > 1:
            raise ValueError(f"Track {track.track_id} contains inconsistent semantic embedding dimensions")
        expected_dim = cache.config.get("semantic_embedding_dim")
        if expected_dim is not None and any(len(token.image_embedding) not in {0, int(expected_dim)} for token in track.anchor_tokens):
            raise ValueError(f"Track {track.track_id} has a semantic embedding dimension different from cache metadata")


def save_cache(path: Path, cache: TrackCache) -> None:
    """Validate and atomically write a cache JSON file."""
    validate_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_payload(cache), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def load_cache(path: Path, *, expected_source_id: str | None = None, expected_scope_fingerprint: str | None = None) -> TrackCache:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read track cache {path}: {exc}") from exc
    cache = TrackCache(
        source=dict(payload.get("source", {})),
        config=dict(payload.get("config", {})),
        proposal_scope=dict(payload.get("proposal_scope", {})),
        tracks=[_track_from_dict(item) for item in payload.get("tracks", [])],
        schema_version=int(payload.get("schema_version", 0)),
    )
    validate_cache(cache)
    if expected_source_id is not None and cache.source.get("source_id") != expected_source_id:
        raise ValueError("Track cache source fingerprint does not match the requested video")
    if expected_scope_fingerprint is not None and cache.scope_fingerprint != expected_scope_fingerprint:
        raise ValueError("Track cache proposal scope does not cover the requested expression")
    return cache


def make_proposal_scope(targets: list[str] | tuple[str, ...], proposal_prompts: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    targets = tuple(sorted({" ".join(str(item).lower().split()) for item in targets if str(item).strip()}))
    prompts = tuple(sorted({str(item).strip() for item in proposal_prompts if str(item).strip()}))
    return {"targets": list(targets), "proposal_prompts": list(prompts), "fingerprint": scope_fingerprint(list(targets), list(prompts))}
