"""Text-prompted image and video annotation."""

from .config import AnnotationConfig


def annotate_media(*args, **kwargs):
    """Lazy entry point so metadata/helpers work before OpenCV is installed."""
    from .pipeline import annotate_media as _annotate_media
    return _annotate_media(*args, **kwargs)

__all__ = ["AnnotationConfig", "annotate_media"]
