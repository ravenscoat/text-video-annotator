from dataclasses import dataclass
from pathlib import Path


@dataclass
class AnnotationConfig:
    input_path: Path
    prompt: str
    output_path: Path
    export_json: Path | None = None
    export_masks: Path | None = None
    prompt_mode: str = "category"
    long_side: int = 768
    chunk_frames: int = 60
    redetect_every: int = 60
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    max_objects: int = 10
    max_objects_per_target: int = 5
    device: str = "cuda"
    keep_audio: bool = False
    targets: tuple[str, ...] | None = None
    track_cache_path: Path | None = None
    reuse_track_cache: bool = False
    semantic_model_id: str | None = None
    semantic_device: str = "cpu"
    selector_checkpoint: Path | None = None
    selector_threshold: float = 0.1

    def validate(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input does not exist: {self.input_path}")
        if not self.prompt.strip() and not self.targets:
            raise ValueError("Prompt cannot be empty")
        if self.prompt_mode not in {"category", "referring"}:
            raise ValueError("prompt_mode must be 'category' or 'referring'")
        if self.long_side < 256:
            raise ValueError("long_side must be at least 256")
        if self.chunk_frames < 1 or self.redetect_every < 1:
            raise ValueError("chunk_frames and redetect_every must be positive")
        if self.max_objects < 1:
            raise ValueError("max_objects must be positive")
        if self.max_objects_per_target < 1:
            raise ValueError("max_objects_per_target must be positive")
        if self.input_path.resolve() == self.output_path.resolve():
            raise ValueError("Output path must differ from input path")
