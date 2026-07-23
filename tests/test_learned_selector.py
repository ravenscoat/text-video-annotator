from pathlib import Path

import torch

from video_annotator.learned_selector import TemporalTrackSelector, load_selector_checkpoint


def test_temporal_selector_forward_shapes():
    model = TemporalTrackSelector(track_dim=8, token_dim=12, text_dim=16, hidden=24)
    output = model(torch.zeros(3, 8), torch.zeros(3, 5, 12), torch.zeros(16))
    assert tuple(output.shape) == (3,)


def test_selector_checkpoint_round_trip(tmp_path: Path):
    model = TemporalTrackSelector(track_dim=8, token_dim=12, text_dim=16, hidden=24)
    checkpoint = tmp_path / "selector.pt"
    torch.save({"model": model.state_dict(), "track_mean": [0.0] * 8, "track_std": [1.0] * 8, "token_mean": [0.0] * 12, "token_std": [1.0] * 12}, checkpoint)
    restored, payload = load_selector_checkpoint(checkpoint)
    assert restored.token.weight_ih_l0.shape[1] == 12
    assert restored.head[0].weight.shape[1] == 8 + 24 + 16
    assert len(payload["track_mean"]) == 8
