from __future__ import annotations

from typing import Any

from .residual_scale_fx_patch import log_scale_on_fx_stats_to_wandb


class ResidualScaleWandbCallback:
    """Lightweight HF Trainer callback for residual alpha stats.

    Usage:
        trainer.add_callback(ResidualScaleWandbCallback(prefix="residual"))
    """

    def __init__(self, prefix: str = "residual", commit: bool = False, reset: bool = True):
        self.prefix = prefix
        self.commit = commit
        self.reset = reset

    # Keep signature compatible with transformers.TrainerCallback.on_log
    def on_log(self, args: Any, state: Any, control: Any, model=None, **kwargs: Any):
        if model is None:
            return control
        is_world_process_zero = bool(getattr(state, "is_world_process_zero", True))
        if not is_world_process_zero:
            return control

        step = getattr(state, "global_step", None)
        log_scale_on_fx_stats_to_wandb(
            model=model,
            step=step,
            reset=self.reset,
            prefix=self.prefix,
            commit=self.commit,
        )
        return control
