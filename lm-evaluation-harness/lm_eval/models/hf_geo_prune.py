from __future__ import annotations

import logging
import fcntl
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import torch

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


eval_logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPRESSION_CODE = _REPO_ROOT / "compression" / "code"
if str(_COMPRESSION_CODE) not in sys.path:
    sys.path.insert(0, str(_COMPRESSION_CODE))

from geometry_aware_pruning import GeometryScoreCollector, apply_geometry_pruning  # noqa: E402


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value_norm = value.strip().lower()
        if value_norm in {"1", "true", "yes", "y", "on"}:
            return True
        if value_norm in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def _split_prompts(value: str) -> list[str]:
    prompts = []
    for item in value.split("||"):
        item = item.strip()
        if item:
            prompts.append(item)
    return prompts


def _default_prompts() -> list[str]:
    return [
        "Transformer compression should preserve behaviorally important representation directions.",
        "Residual-space geometry separates rescaling-like changes from direction-changing updates.",
    ]


@register_model("hf-geo-prune")
class HFGeoPruneLM(HFLM):
    """Hugging Face backend with in-memory geometry-aware WANDA pruning."""

    _GEO_KWARG_KEYS = {
        "geo_prune_enabled",
        "geo_prune_method",
        "geo_prune_strategy",
        "geo_prune_targets",
        "geo_geometry_mode",
        "geo_geometry_alpha",
        "geo_geometry_targets",
        "geo_sparsity_ratio",
        "geo_sparsity_type",
        "geo_threshold_scope",
        "geo_token_scope",
        "geo_calib_prompts",
        "geo_calib_file",
        "geo_max_prompts",
        "geo_max_length",
        "geo_score_cache_path",
    }

    def __init__(
        self,
        *args,
        geo_prune_enabled: bool = True,
        geo_prune_method: Literal["wanda", "magnitude"] = "wanda",
        geo_prune_strategy: str = "none",
        geo_prune_targets: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        geo_geometry_mode: str = "residual_error",
        geo_geometry_alpha: float = 1.0,
        geo_geometry_targets: str = "o_proj,down_proj,v_proj",
        geo_sparsity_ratio: float = 0.5,
        geo_sparsity_type: Literal["unstructured", "2:4", "4:8"] = "2:4",
        geo_threshold_scope: Literal["row", "global"] = "global",
        geo_token_scope: Literal["last", "all"] = "last",
        geo_calib_prompts: str = "",
        geo_calib_file: str = "",
        geo_max_prompts: int = 2,
        geo_max_length: int = 256,
        geo_score_cache_path: str = "",
        **kwargs,
    ) -> None:
        geo_prune_enabled = kwargs.pop("geo_prune_enabled", geo_prune_enabled)
        geo_prune_method = kwargs.pop("geo_prune_method", geo_prune_method)
        geo_prune_strategy = kwargs.pop("geo_prune_strategy", geo_prune_strategy)
        geo_prune_targets = kwargs.pop("geo_prune_targets", geo_prune_targets)
        geo_geometry_mode = kwargs.pop("geo_geometry_mode", geo_geometry_mode)
        geo_geometry_alpha = kwargs.pop("geo_geometry_alpha", geo_geometry_alpha)
        geo_geometry_targets = kwargs.pop("geo_geometry_targets", geo_geometry_targets)
        geo_sparsity_ratio = kwargs.pop("geo_sparsity_ratio", geo_sparsity_ratio)
        geo_sparsity_type = kwargs.pop("geo_sparsity_type", geo_sparsity_type)
        geo_threshold_scope = kwargs.pop("geo_threshold_scope", geo_threshold_scope)
        geo_token_scope = kwargs.pop("geo_token_scope", geo_token_scope)
        geo_calib_prompts = kwargs.pop("geo_calib_prompts", geo_calib_prompts)
        geo_calib_file = kwargs.pop("geo_calib_file", geo_calib_file)
        geo_max_prompts = kwargs.pop("geo_max_prompts", geo_max_prompts)
        geo_max_length = kwargs.pop("geo_max_length", geo_max_length)
        geo_score_cache_path = kwargs.pop("geo_score_cache_path", geo_score_cache_path)

        self.geo_prune_enabled = _as_bool(geo_prune_enabled)
        self.geo_prune_method = str(geo_prune_method)
        self.geo_prune_strategy = str(geo_prune_strategy)
        self.geo_prune_targets = str(geo_prune_targets)
        self.geo_geometry_mode = str(geo_geometry_mode)
        self.geo_geometry_alpha = float(geo_geometry_alpha)
        self.geo_geometry_targets = str(geo_geometry_targets)
        self.geo_sparsity_ratio = float(geo_sparsity_ratio)
        self.geo_sparsity_type = str(geo_sparsity_type)
        self.geo_threshold_scope = str(geo_threshold_scope)
        self.geo_token_scope = str(geo_token_scope)
        self.geo_calib_prompts = str(geo_calib_prompts)
        self.geo_calib_file = str(geo_calib_file)
        self.geo_max_prompts = int(geo_max_prompts)
        self.geo_max_length = int(geo_max_length)
        self.geo_score_cache_path = str(geo_score_cache_path)
        super().__init__(*args, **kwargs)
        self._apply_geo_prune_if_needed()

    def _create_model(self, *args, **kwargs) -> None:
        for key in self._GEO_KWARG_KEYS:
            kwargs.pop(key, None)
        super()._create_model(*args, **kwargs)

    def _load_calibration_prompts(self) -> list[str]:
        if self.geo_calib_prompts:
            prompts = _split_prompts(self.geo_calib_prompts)
        elif self.geo_calib_file:
            path = Path(self.geo_calib_file)
            prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            prompts = _default_prompts()
        return prompts[: self.geo_max_prompts]

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _collect_geo_scores(self) -> dict[str, Any]:
        cache_path = Path(self.geo_score_cache_path) if self.geo_score_cache_path else None
        lock_handle = None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
            lock_handle = lock_path.open("w")
            eval_logger.info("Waiting for geometry score cache lock: %s", lock_path)
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            if cache_path.is_file():
                eval_logger.info("Loading cached geometry scores from %s", cache_path)
                try:
                    return torch.load(cache_path, map_location="cpu")
                finally:
                    fcntl.flock(lock_handle, fcntl.LOCK_UN)
                    lock_handle.close()

        prompts = self._load_calibration_prompts()
        eval_logger.info(
            "Collecting geometry scores: prompts=%s token_scope=%s max_length=%s",
            len(prompts),
            self.geo_token_scope,
            self.geo_max_length,
        )
        collector = GeometryScoreCollector(self.model, token_scope=self.geo_token_scope)
        old_use_cache = getattr(self.model.config, "use_cache", None)
        if old_use_cache is not None:
            self.model.config.use_cache = False
        device = self._model_device()
        collector.install()
        try:
            for prompt in prompts:
                enc = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.geo_max_length,
                )
                enc = {key: value.to(device) for key, value in enc.items()}
                with torch.no_grad():
                    self.model(**enc, use_cache=False)
        finally:
            collector.remove()
            if old_use_cache is not None:
                self.model.config.use_cache = old_use_cache
        scores = collector.to_tensors()
        if cache_path is not None:
            torch.save(scores, cache_path)
            eval_logger.info("Saved geometry scores to %s", cache_path)
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()
        return scores

    def _apply_geo_prune_if_needed(self) -> None:
        if not self.geo_prune_enabled or self.geo_sparsity_ratio <= 0.0:
            eval_logger.info("Geometry pruning disabled for this model instance.")
            return
        scores = self._collect_geo_scores()
        prune_args = SimpleNamespace(
            prune_method=self.geo_prune_method,
            geometry_strategy=self.geo_prune_strategy,
            prune_targets=self.geo_prune_targets,
            geometry_mode=self.geo_geometry_mode,
            geometry_alpha=self.geo_geometry_alpha,
            geometry_targets=self.geo_geometry_targets,
            sparsity_ratio=self.geo_sparsity_ratio,
            sparsity_type=self.geo_sparsity_type,
            threshold_scope=self.geo_threshold_scope,
        )
        records = apply_geometry_pruning(self.model, scores, prune_args)
        mean_sparsity = sum(item["sparsity"] for item in records) / max(len(records), 1)
        eval_logger.info(
            "Applied geometry pruning: method=%s strategy=%s mode=%s alpha=%s prune_targets=%s geometry_targets=%s sparsity_type=%s target_sparsity=%s modules=%s mean_module_sparsity=%.6f",
            self.geo_prune_method,
            self.geo_prune_strategy,
            self.geo_geometry_mode,
            self.geo_geometry_alpha,
            self.geo_prune_targets,
            self.geo_geometry_targets,
            self.geo_sparsity_type,
            self.geo_sparsity_ratio,
            len(records),
            mean_sparsity,
        )
