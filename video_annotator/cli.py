from pathlib import Path
import typer

from .config import AnnotationConfig
from .pipeline import annotate_media

app = typer.Typer(help="Offline text-prompted image/video annotation")


@app.command()
def annotate(
    input: Path = typer.Option(..., "--input", exists=True, readable=True),
    prompt: str = typer.Option("", "--prompt"),
    target: list[str] | None = typer.Option(None, "--target", help="Explicit category target; repeat for multiple classes"),
    output: Path = typer.Option(..., "--output"),
    export_json: Path | None = typer.Option(None, "--export-json"),
    export_masks: Path | None = typer.Option(None, "--export-masks"),
    prompt_mode: str = typer.Option("category", "--prompt-mode"),
    long_side: int = typer.Option(768, "--long-side"),
    chunk_frames: int = typer.Option(60, "--chunk-frames"),
    redetect_every: int = typer.Option(60, "--redetect-every"),
    box_threshold: float = typer.Option(0.30, "--box-threshold"),
    text_threshold: float = typer.Option(0.25, "--text-threshold"),
    max_objects: int = typer.Option(10, "--max-objects"),
    max_objects_per_target: int = typer.Option(5, "--max-objects-per-target"),
    device: str = typer.Option("cuda", "--device"),
    track_cache: Path | None = typer.Option(None, "--track-cache", help="Write/reuse a candidate-track cache JSON"),
    reuse_track_cache: bool = typer.Option(False, "--reuse-track-cache", help="Render from an existing compatible cache without model inference"),
    semantic_model: str | None = typer.Option(None, "--semantic-model", help="Explicit cached Hugging Face image/text model for ordered embeddings"),
    semantic_device: str = typer.Option("cpu", "--semantic-device", help="Device for optional semantic encoder; CPU is safest with 8GB VRAM"),
    selector_checkpoint: Path | None = typer.Option(None, "--selector-checkpoint", help="Optional learned selector checkpoint for cache rendering"),
    selector_threshold: float = typer.Option(0.1, "--selector-threshold"),
):
    result = annotate_media(AnnotationConfig(input, prompt, output, export_json, export_masks, prompt_mode, long_side, chunk_frames, redetect_every, box_threshold, text_threshold, max_objects, max_objects_per_target=max_objects_per_target, device=device, targets=tuple(target) if target else None, track_cache_path=track_cache, reuse_track_cache=reuse_track_cache, semantic_model_id=semantic_model, semantic_device=semantic_device, selector_checkpoint=selector_checkpoint, selector_threshold=selector_threshold))
    typer.echo(f"Wrote {result.output_path} ({result.objects_found} objects detected)")


if __name__ == "__main__":
    app()
