"""Forward-only geometry interventions for multimodal models."""

from .hooks import GeometryScaleConfig, VLMGeometryScaler
from .presets import MODEL_PRESETS, PresetSpec, resolve_preset

__all__ = [
    "GeometryScaleConfig",
    "MODEL_PRESETS",
    "PresetSpec",
    "VLMGeometryScaler",
    "resolve_preset",
]
