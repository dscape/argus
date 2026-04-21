from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from study.templates.inference.embedder import embed


@dataclass(frozen=True)
class TemplateMatchResult:
    piece_type: str
    confidence: float
    margin: float
    piece_similarities: dict[str, float]


@dataclass(frozen=True)
class TemplateInstanceMatch:
    piece_type: str
    similarity: float
    metadata: dict[str, Any]


def load_template_bank(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid template bank payload: {path}")
    embeddings_by_piece_type = payload.get("embeddings_by_piece_type")
    if not isinstance(embeddings_by_piece_type, dict):
        raise ValueError(f"Template bank is missing embeddings_by_piece_type: {path}")
    return payload


def classify_crop(
    crop: Any,
    template_bank: dict[str, Any],
) -> TemplateMatchResult:
    embedding = _embed_for_template_bank(crop, template_bank)
    return classify_embedding(embedding, template_bank)


def classify_embedding(
    embedding: torch.Tensor,
    template_bank: dict[str, Any],
) -> TemplateMatchResult:
    if embedding.ndim != 1:
        raise ValueError(f"Expected 1D embedding, got shape {tuple(embedding.shape)}")

    normalized_embedding = F.normalize(embedding.to(torch.float32).unsqueeze(0), dim=-1)
    piece_similarities: dict[str, float] = {}
    for piece_type, embeddings in _embeddings_by_piece_type(template_bank).items():
        if embeddings.numel() == 0:
            continue
        normalized_templates = F.normalize(embeddings.to(torch.float32), dim=-1)
        similarities = normalized_embedding @ normalized_templates.T
        piece_similarities[str(piece_type)] = float(similarities.max().item())

    if not piece_similarities:
        raise ValueError("Template bank has no embeddings")

    ranked = sorted(piece_similarities.items(), key=lambda item: item[1], reverse=True)
    top_piece_type, top_similarity = ranked[0]
    second_similarity = ranked[1][1] if len(ranked) > 1 else -1.0
    return TemplateMatchResult(
        piece_type=top_piece_type,
        confidence=top_similarity,
        margin=top_similarity - second_similarity,
        piece_similarities=piece_similarities,
    )


def top_template_matches_for_crop(
    crop: Any,
    template_bank: dict[str, Any],
    *,
    top_k: int = 3,
) -> list[TemplateInstanceMatch]:
    embedding = _embed_for_template_bank(crop, template_bank)
    return top_template_matches_for_embedding(embedding, template_bank, top_k=top_k)


def top_template_matches_for_embedding(
    embedding: torch.Tensor,
    template_bank: dict[str, Any],
    *,
    top_k: int = 3,
) -> list[TemplateInstanceMatch]:
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if embedding.ndim != 1:
        raise ValueError(f"Expected 1D embedding, got shape {tuple(embedding.shape)}")

    normalized_embedding = F.normalize(embedding.to(torch.float32).unsqueeze(0), dim=-1)
    metadata_by_piece_type = _template_metadata_by_piece_type(template_bank)
    matches: list[TemplateInstanceMatch] = []
    for piece_type, embeddings in _embeddings_by_piece_type(template_bank).items():
        if embeddings.numel() == 0:
            continue
        normalized_templates = F.normalize(embeddings.to(torch.float32), dim=-1)
        similarities = (normalized_embedding @ normalized_templates.T).squeeze(0)
        metadata_entries = metadata_by_piece_type.get(piece_type, [])
        for index, similarity in enumerate(similarities.tolist()):
            metadata = metadata_entries[index] if index < len(metadata_entries) else {}
            matches.append(
                TemplateInstanceMatch(
                    piece_type=piece_type,
                    similarity=float(similarity),
                    metadata=dict(metadata),
                )
            )
    matches.sort(key=lambda match: match.similarity, reverse=True)
    return matches[:top_k]


def _embed_for_template_bank(crop: Any, template_bank: dict[str, Any]) -> torch.Tensor:
    encoder_config = dict(template_bank.get("encoder_config", {}))
    return embed(
        crop,
        encoder_type=str(encoder_config.get("encoder_type", "dinov3")),
        model_name=(
            None
            if encoder_config.get("model_name") is None
            else str(encoder_config.get("model_name"))
        ),
        input_size=int(encoder_config.get("input_size", 224)),
        device=str(encoder_config.get("device", "cpu")),
    )


def _embeddings_by_piece_type(template_bank: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw = template_bank.get("embeddings_by_piece_type")
    if not isinstance(raw, dict):
        raise ValueError("Template bank is missing embeddings_by_piece_type")
    result: dict[str, torch.Tensor] = {}
    for piece_type, embeddings in raw.items():
        if not isinstance(piece_type, str):
            raise ValueError("Template bank piece types must be strings")
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError(f"Template bank embeddings for {piece_type!r} must be tensors")
        if embeddings.ndim != 2:
            raise ValueError(
                "Template bank embeddings for"
                f" {piece_type!r} must be 2D, got {tuple(embeddings.shape)}"
            )
        result[piece_type] = embeddings
    return result


def _template_metadata_by_piece_type(
    template_bank: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    raw = template_bank.get("template_metadata_by_piece_type", {})
    if not isinstance(raw, dict):
        raise ValueError("Template bank template_metadata_by_piece_type must be a dict")
    result: dict[str, list[dict[str, Any]]] = {}
    for piece_type, entries in raw.items():
        if not isinstance(piece_type, str):
            raise ValueError("Template bank piece types must be strings")
        if not isinstance(entries, list):
            raise ValueError(f"Template metadata for {piece_type!r} must be a list")
        result[piece_type] = [dict(entry) for entry in entries if isinstance(entry, dict)]
    return result


__all__ = [
    "TemplateInstanceMatch",
    "TemplateMatchResult",
    "classify_crop",
    "classify_embedding",
    "load_template_bank",
    "top_template_matches_for_crop",
    "top_template_matches_for_embedding",
]
