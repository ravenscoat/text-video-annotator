"""Small temporal track selector trained on cached proposal features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


class TemporalTrackSelector(nn.Module):
    def __init__(self, track_dim: int, token_dim: int, text_dim: int, hidden: int = 96):
        super().__init__()
        self.token = nn.GRU(token_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(track_dim + hidden + text_dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, tracks: torch.Tensor, tokens: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        _, state = self.token(tokens)
        text = text.expand(tracks.shape[0], -1)
        return self.head(torch.cat((tracks, state[-1], text), dim=-1)).squeeze(-1)


def load_selector_checkpoint(path: Path, device: str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["model"]
    token_dim = int(state["token.weight_ih_l0"].shape[1])
    track_dim = len(payload["track_mean"])
    hidden = int(state["token.weight_hh_l0"].shape[1])
    text_dim = int(state["head.0.weight"].shape[1] - track_dim - hidden)
    model = TemporalTrackSelector(track_dim, token_dim, text_dim, hidden=hidden).to(device)
    model.load_state_dict(state); model.eval()
    return model, payload


def score_cached_tracks(cache, prompt: str, checkpoint: Path, semantic_encoder, device: str = "cpu") -> dict[int, float]:
    """Score every cached track for a prompt using a trained selector."""
    from .selector_data import _track_vector, _token_vector
    model, payload = load_selector_checkpoint(checkpoint, device)
    track_mean=torch.as_tensor(payload["track_mean"],dtype=torch.float32,device=device); track_std=torch.as_tensor(payload["track_std"],dtype=torch.float32,device=device)
    token_mean=torch.as_tensor(payload["token_mean"],dtype=torch.float32,device=device); token_std=torch.as_tensor(payload["token_std"],dtype=torch.float32,device=device)
    text=torch.as_tensor(semantic_encoder.encode_text([prompt])[0],dtype=torch.float32,device=device)
    token_dim=int(model.token.weight_ih_l0.shape[1]); rows=[]; sequences=[]
    for track in cache.tracks:
        vector,_=_track_vector(track); rows.append(vector); sequences.append([_token_vector(token) for token in track.anchor_tokens])
    if not rows: return {}
    tracks=torch.as_tensor(rows,dtype=torch.float32,device=device); tracks=(tracks-track_mean)/track_std
    max_len=max(1,max((len(seq) for seq in sequences),default=1)); padded=torch.zeros((len(rows),max_len,token_dim),device=device)
    for i,seq in enumerate(sequences):
        for j,token in enumerate(seq[:max_len]):
            values=torch.as_tensor(token[:token_dim],dtype=torch.float32,device=device); padded[i,j,:values.numel()]=(values-token_mean[:values.numel()])/token_std[:values.numel()]
    with torch.no_grad(): scores=torch.sigmoid(model(tracks,padded,text)).cpu().tolist()
    return {track.track_id: float(score) for track,score in zip(cache.tracks,scores)}


def load_selector_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
