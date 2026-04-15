"""Vision encoders for per-frame board feature extraction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Dinov2Model, Siglip2VisionModel, SiglipModel, SiglipVisionModel

_WEIGHTS_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "weights"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_SIGLIP_MEAN = (0.5, 0.5, 0.5)
_SIGLIP_STD = (0.5, 0.5, 0.5)

DEFAULT_DINOV2_MODEL = "facebook/dinov2-base"
DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-224"
DEFAULT_SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_YOLO_MODEL = "weights/yolo_base/yolo11n.pt"


def default_model_name_for_encoder_type(encoder_type: str) -> str:
    normalized = encoder_type.lower()
    if normalized in {"dino", "dinov2"}:
        return DEFAULT_DINOV2_MODEL
    if normalized == "siglip":
        return DEFAULT_SIGLIP_MODEL
    if normalized == "siglip2":
        return DEFAULT_SIGLIP2_MODEL
    if normalized == "yolo":
        return DEFAULT_YOLO_MODEL
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")


def _has_transformers_weights(directory: Path) -> bool:
    weight_filenames = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return any((directory / filename).exists() for filename in weight_filenames)


def _resolve_dino_model_path(model_name: str) -> str:
    """Return a local DINO weights path if available, otherwise the HF repo id."""
    short = model_name.split("/")[-1]
    local = _WEIGHTS_ROOT / short
    if (local / "config.json").exists() and _has_transformers_weights(local):
        return str(local)
    return model_name


def _resolve_siglip_model_path(model_name: str) -> str:
    """Return a local SigLIP weights path if available, otherwise the HF repo id."""
    short = model_name.split("/")[-1]
    local = _WEIGHTS_ROOT / short
    if (local / "config.json").exists() and _has_transformers_weights(local):
        return str(local)
    cache_path = _resolve_hf_cache_model_path(model_name)
    if cache_path is not None:
        return cache_path
    return model_name


def _resolve_hf_cache_model_path(model_name: str) -> str | None:
    if "/" not in model_name:
        return None
    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_root = hub_root / f"models--{model_name.replace('/', '--')}"
    if not model_root.exists():
        return None
    refs_main = model_root / "refs" / "main"
    if refs_main.exists():
        snapshot = model_root / "snapshots" / refs_main.read_text().strip()
        if snapshot.exists() and _has_transformers_weights(snapshot):
            return str(snapshot)
    snapshots_dir = model_root / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(
        path
        for path in snapshots_dir.iterdir()
        if path.is_dir() and _has_transformers_weights(path)
    )
    if not snapshots:
        return None
    return str(snapshots[-1])


def _resolve_yolo_model_path(model_name: str) -> str:
    """Return a local YOLO checkpoint path if available."""
    candidate = Path(model_name)
    if candidate.exists():
        return str(candidate)
    local = _WEIGHTS_ROOT / model_name
    if local.exists():
        return str(local)
    return model_name


def _load_transformers_config(model_name: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return AutoConfig.from_pretrained(model_name)


def _load_siglip_vision_model(model_name: str) -> nn.Module:
    config = _load_transformers_config(model_name)
    model_type = getattr(config, "model_type", None)
    try:
        if model_type == "siglip":
            return cast(
                nn.Module,
                SiglipModel.from_pretrained(model_name, local_files_only=True).vision_model,
            )
        if model_type == "siglip_vision_model":
            return SiglipVisionModel.from_pretrained(model_name, local_files_only=True)
    except OSError:
        if model_type == "siglip":
            return cast(nn.Module, SiglipModel.from_pretrained(model_name).vision_model)
        if model_type == "siglip_vision_model":
            return SiglipVisionModel.from_pretrained(model_name)
    raise ValueError(
        f"Requested a SigLIP encoder for {model_name!r}, but config model_type is {model_type!r}"
    )


def _load_siglip2_vision_model(model_name: str) -> nn.Module:
    config = _load_transformers_config(model_name)
    model_type = getattr(config, "model_type", None)
    if model_type not in {"siglip2", "siglip2_vision_model"}:
        raise ValueError(
            "Requested encoder_type='siglip2' for"
            f" {model_name!r}, but config model_type is {model_type!r}."
            " Use encoder_type='siglip' for SigLIP checkpoints or pass a real SigLIP2"
            " checkpoint."
        )
    try:
        return Siglip2VisionModel.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return Siglip2VisionModel.from_pretrained(model_name)


def _forward_siglip_patch_tokens(
    vision_model: Any,
    pixel_values: torch.Tensor,
    *,
    feature_layer_indices: Sequence[int],
) -> torch.Tensor:
    hidden_states = vision_model.embeddings(
        pixel_values,
        interpolate_pos_encoding=True,
    )
    if not feature_layer_indices:
        for layer in vision_model.encoder.layers:
            hidden_states = layer(hidden_states, None)
        return cast(torch.Tensor, vision_model.post_layernorm(hidden_states))

    selected_states: list[torch.Tensor] = []
    requested_indices = {int(index) for index in feature_layer_indices}
    for layer_index, layer in enumerate(vision_model.encoder.layers):
        hidden_states = layer(hidden_states, None)
        if layer_index in requested_indices:
            selected_states.append(vision_model.post_layernorm(hidden_states))
    if not selected_states:
        raise RuntimeError("SigLIP encoder did not produce the requested hidden states")
    return torch.stack(selected_states, dim=0).mean(dim=0)


class Dinov2Backbone(nn.Module):
    """DINOv2 ViT wrapper exposing patch tokens."""

    def __init__(
        self,
        model_name: str = DEFAULT_DINOV2_MODEL,
        frozen: bool = True,
        embed_dim: int | None = None,
        feature_layer_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        resolved = _resolve_dino_model_path(model_name)
        try:
            self.model = Dinov2Model.from_pretrained(resolved, local_files_only=True)
        except OSError:
            self.model = Dinov2Model.from_pretrained(model_name)

        hidden_size = int(self.model.config.hidden_size)
        raw_feature_layers = feature_layer_indices or ()
        self.feature_layer_indices = tuple(int(idx) for idx in raw_feature_layers)
        num_hidden_layers = int(self.model.config.num_hidden_layers)
        for layer_index in self.feature_layer_indices:
            if layer_index < 0 or layer_index >= num_hidden_layers:
                raise ValueError(
                    "Invalid DINO feature layer index"
                    f" {layer_index}; expected 0 <= idx < {num_hidden_layers}"
                )
        if embed_dim is not None and embed_dim != hidden_size:
            raise ValueError(
                f"Requested DINO embed_dim={embed_dim}, but model outputs {hidden_size}"
            )
        self.embed_dim = hidden_size
        if frozen:
            self.freeze()

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int) -> None:
        self.freeze()
        if n <= 0:
            return
        for layer in self.model.encoder.layer[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        if not self.feature_layer_indices:
            outputs = self.model(pixel_values=x)
            result: torch.Tensor = outputs.last_hidden_state
            return result

        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("DINO encoder did not return hidden states")
        selected_states = [
            hidden_states[layer_index + 1] for layer_index in self.feature_layer_indices
        ]
        result = torch.stack(selected_states, dim=0).mean(dim=0)
        return result

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.forward_patches(x)
        return patches[:, 1:, :].mean(dim=1)

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        if return_patches:
            return self.forward_patches(x)
        return self.forward_pooled(x)


class _SiglipNormalizationMixin:
    input_mean: torch.Tensor
    input_std: torch.Tensor
    siglip_mean: torch.Tensor
    siglip_std: torch.Tensor

    def register_buffer(
        self,
        name: str,
        tensor: torch.Tensor,
        persistent: bool = False,
    ) -> None: ...

    def _register_input_buffers(self) -> None:
        self.register_buffer(
            "input_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "siglip_mean",
            torch.tensor(_SIGLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "siglip_std",
            torch.tensor(_SIGLIP_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        imagenet_mean = self.input_mean.to(device=x.device, dtype=x.dtype)
        imagenet_std = self.input_std.to(device=x.device, dtype=x.dtype)
        siglip_mean = self.siglip_mean.to(device=x.device, dtype=x.dtype)
        siglip_std = self.siglip_std.to(device=x.device, dtype=x.dtype)
        rgb = (x * imagenet_std + imagenet_mean).clamp(0.0, 1.0)
        return (rgb - siglip_mean) / siglip_std


class SiglipBackbone(nn.Module, _SiglipNormalizationMixin):
    """SigLIP vision wrapper exposing dense patch tokens."""

    def __init__(
        self,
        model_name: str = DEFAULT_SIGLIP_MODEL,
        frozen: bool = True,
        embed_dim: int | None = None,
        feature_layer_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        resolved = _resolve_siglip_model_path(model_name)
        loaded_model = _load_siglip_vision_model(resolved)
        self.model = loaded_model
        self._vision_model = cast(
            Any,
            loaded_model.vision_model if hasattr(loaded_model, "vision_model") else loaded_model,
        )

        hidden_size = int(self._vision_model.config.hidden_size)
        raw_feature_layers = feature_layer_indices or ()
        self.feature_layer_indices = tuple(int(idx) for idx in raw_feature_layers)
        num_hidden_layers = int(self._vision_model.config.num_hidden_layers)
        for layer_index in self.feature_layer_indices:
            if layer_index < 0 or layer_index >= num_hidden_layers:
                raise ValueError(
                    "Invalid SigLIP feature layer index"
                    f" {layer_index}; expected 0 <= idx < {num_hidden_layers}"
                )
        if embed_dim is not None and embed_dim != hidden_size:
            raise ValueError(
                f"Requested SigLIP embed_dim={embed_dim}, but model outputs {hidden_size}"
            )
        self.embed_dim = hidden_size
        self._register_input_buffers()
        if frozen:
            self.freeze()

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int) -> None:
        self.freeze()
        if n <= 0:
            return
        for layer in self._vision_model.encoder.layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        prepared = self._prepare_inputs(x)
        return _forward_siglip_patch_tokens(
            self._vision_model,
            prepared,
            feature_layer_indices=self.feature_layer_indices,
        )

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.forward_patches(x)
        return patches.mean(dim=1)

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        if return_patches:
            return self.forward_patches(x)
        return self.forward_pooled(x)


class Siglip2Backbone(nn.Module, _SiglipNormalizationMixin):
    """SigLIP2 vision wrapper exposing dense patch tokens."""

    def __init__(
        self,
        model_name: str = DEFAULT_SIGLIP2_MODEL,
        frozen: bool = True,
        embed_dim: int | None = None,
        feature_layer_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        resolved = _resolve_siglip_model_path(model_name)
        loaded_model = _load_siglip2_vision_model(resolved)
        self.model = loaded_model
        self._vision_model = cast(
            Any,
            loaded_model.vision_model if hasattr(loaded_model, "vision_model") else loaded_model,
        )

        hidden_size = int(self._vision_model.config.hidden_size)
        raw_feature_layers = feature_layer_indices or ()
        self.feature_layer_indices = tuple(int(idx) for idx in raw_feature_layers)
        num_hidden_layers = int(self._vision_model.config.num_hidden_layers)
        for layer_index in self.feature_layer_indices:
            if layer_index < 0 or layer_index >= num_hidden_layers:
                raise ValueError(
                    "Invalid SigLIP2 feature layer index"
                    f" {layer_index}; expected 0 <= idx < {num_hidden_layers}"
                )
        if embed_dim is not None and embed_dim != hidden_size:
            raise ValueError(
                f"Requested SigLIP2 embed_dim={embed_dim}, but model outputs {hidden_size}"
            )
        self.embed_dim = hidden_size
        self.patch_size = int(self._vision_model.config.patch_size)
        self._register_input_buffers()
        if frozen:
            self.freeze()

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int) -> None:
        self.freeze()
        if n <= 0:
            return
        for layer in self._vision_model.encoder.layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def _resize_to_patch_grid(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        target_height = max(self.patch_size, round(height / self.patch_size) * self.patch_size)
        target_width = max(self.patch_size, round(width / self.patch_size) * self.patch_size)
        if target_height == height and target_width == width:
            return x
        return F.interpolate(
            x,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

    def _patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        resized = self._resize_to_patch_grid(x)
        batch_size, channels, height, width = resized.shape
        patches_h = height // self.patch_size
        patches_w = width // self.patch_size
        patches = resized.unfold(2, self.patch_size, self.patch_size).unfold(
            3,
            self.patch_size,
            self.patch_size,
        )
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            batch_size,
            patches_h * patches_w,
            channels * self.patch_size * self.patch_size,
        )
        attention_mask = torch.ones(
            batch_size,
            patches_h * patches_w,
            dtype=torch.bool,
            device=resized.device,
        )
        spatial_shapes = torch.tensor(
            [[patches_h, patches_w]] * batch_size,
            dtype=torch.long,
            device=resized.device,
        )
        return patches, attention_mask, spatial_shapes

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        prepared = self._prepare_inputs(x)
        patch_values, pixel_attention_mask, spatial_shapes = self._patchify(prepared)
        if not self.feature_layer_indices:
            outputs = self.model(
                pixel_values=patch_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
            result: torch.Tensor = outputs.last_hidden_state
            return result

        outputs = self.model(
            pixel_values=patch_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("SigLIP2 encoder did not return hidden states")
        selected_states = [
            hidden_states[layer_index + 1] for layer_index in self.feature_layer_indices
        ]
        result = torch.stack(selected_states, dim=0).mean(dim=0)
        return result

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.forward_patches(x)
        return patches.mean(dim=1)

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        if return_patches:
            return self.forward_patches(x)
        return self.forward_pooled(x)


class YoloBackbone(nn.Module):
    """YOLO-derived multi-scale visual frontend.

    The existing Argus data path normalizes inputs for DINO. To keep that path
    fixed for the frontend comparison, this backbone first undoes ImageNet
    normalization, then extracts multi-scale YOLO feature maps and fuses them
    into a square token grid.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_YOLO_MODEL,
        frozen: bool = True,
        feature_layer_indices: Sequence[int] | None = None,
        output_grid_size: int = 14,
    ) -> None:
        super().__init__()
        resolved = _resolve_yolo_model_path(model_name)
        from ultralytics import YOLO  # type: ignore[attr-defined]

        self.model: Any = YOLO(resolved).model
        raw_feature_layers = feature_layer_indices or (16, 19, 22)
        self.feature_layer_indices = tuple(int(idx) for idx in raw_feature_layers)
        if len(self.feature_layer_indices) == 0:
            raise ValueError("feature_layer_indices must not be empty")
        self.output_grid_size = output_grid_size
        self._stop_layer = max(self.feature_layer_indices)
        self.register_buffer(
            "input_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.embed_dim = self._infer_embed_dim()
        if frozen:
            self.freeze()

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n: int) -> None:
        self.freeze()
        if n <= 0:
            return
        layers = list(self.model.model[:-1])
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

    def _infer_embed_dim(self) -> int:
        training = self.model.training
        self.model.eval()
        with torch.no_grad():
            feature_maps = self._forward_feature_maps(torch.zeros(1, 3, 224, 224))
        if training:
            self.model.train()
        return int(sum(feature_map.shape[1] for feature_map in feature_maps))

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.input_mean).to(device=x.device, dtype=x.dtype)
        std = cast(torch.Tensor, self.input_std).to(device=x.device, dtype=x.dtype)
        result = x * std + mean
        return result.clamp(0.0, 1.0)

    def _forward_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self._prepare_inputs(x)
        cached: list[torch.Tensor | None] = []
        outputs: dict[int, torch.Tensor] = {}
        for module in self.model.model:
            if module.f != -1:
                module_input = (
                    cached[module.f]
                    if isinstance(module.f, int)
                    else [x if idx == -1 else cached[idx] for idx in module.f]
                )
            else:
                module_input = x
            x = module(module_input)
            cached.append(x if module.i in self.model.save else None)
            if module.i in self.feature_layer_indices:
                outputs[module.i] = x
            if module.i >= self._stop_layer:
                break
        return [outputs[idx] for idx in self.feature_layer_indices]

    def _resize_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feature_map.shape
        target = self.output_grid_size
        if height == target and width == target:
            return feature_map
        return F.interpolate(
            feature_map,
            size=(target, target),
            mode="bilinear",
            align_corners=False,
        )

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = self._forward_feature_maps(x)
        resized = [self._resize_feature_map(feature_map) for feature_map in feature_maps]
        fused = torch.cat(resized, dim=1)
        tokens = fused.flatten(2).transpose(1, 2)
        result: torch.Tensor = F.layer_norm(tokens, (tokens.shape[-1],))
        return result

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.forward_patches(x)
        return patches.mean(dim=1)

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        if return_patches:
            return self.forward_patches(x)
        return self.forward_pooled(x)


class VisionBackboneProtocol(Protocol):
    embed_dim: int

    def freeze(self) -> None: ...
    def unfreeze_last_n_layers(self, n: int) -> None: ...
    def unfreeze(self) -> None: ...
    def forward_patches(self, x: torch.Tensor) -> torch.Tensor: ...
    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor: ...
    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor: ...


class VisionEncoder(nn.Module):
    """Unified frontend wrapper for DINOv2, SigLIP, SigLIP2, and YOLO features."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        frozen: bool = True,
        embed_dim: int | None = None,
        encoder_type: str = "dinov2",
        feature_layer_indices: Sequence[int] | None = None,
        output_grid_size: int = 14,
    ) -> None:
        super().__init__()
        normalized_type = encoder_type.lower()
        self.encoder_type = normalized_type

        self.backend: VisionBackboneProtocol
        if normalized_type in {"dino", "dinov2"}:
            self.backend = Dinov2Backbone(
                model_name=model_name,
                frozen=frozen,
                embed_dim=embed_dim,
                feature_layer_indices=feature_layer_indices,
            )
        elif normalized_type == "siglip":
            self.backend = SiglipBackbone(
                model_name=model_name,
                frozen=frozen,
                embed_dim=embed_dim,
                feature_layer_indices=feature_layer_indices,
            )
        elif normalized_type == "siglip2":
            self.backend = Siglip2Backbone(
                model_name=model_name,
                frozen=frozen,
                embed_dim=embed_dim,
                feature_layer_indices=feature_layer_indices,
            )
        elif normalized_type == "yolo":
            self.backend = YoloBackbone(
                model_name=model_name,
                frozen=frozen,
                feature_layer_indices=feature_layer_indices,
                output_grid_size=output_grid_size,
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.embed_dim = self.backend.embed_dim

    def freeze(self) -> None:
        self.backend.freeze()

    def unfreeze_last_n_layers(self, n: int) -> None:
        self.backend.unfreeze_last_n_layers(n)

    def unfreeze(self) -> None:
        self.backend.unfreeze()

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.forward_patches(x)

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.forward_pooled(x)

    def forward(self, x: torch.Tensor, return_patches: bool = False) -> torch.Tensor:
        return self.backend.forward(x, return_patches=return_patches)
