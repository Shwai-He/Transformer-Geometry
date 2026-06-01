from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PresetSpec:
    """Module-path hints for one multimodal model family.

    The hook code remains model-agnostic.  Presets only describe where the
    understanding and generation transformers usually live, plus loose child
    module names for attention and MLP branches.  Callers can override all of
    these from the CLI when a local checkout uses different names.
    """

    name: str
    aliases: tuple[str, ...]
    und_layer_paths: tuple[str, ...] = field(default_factory=tuple)
    gen_layer_paths: tuple[str, ...] = field(default_factory=tuple)
    attn_name_regex: str = r"(self_attn|attn|attention|to_q|q_proj)"
    mlp_name_regex: str = r"(mlp|ffn|feed_forward|feedforward|ff|block\.ff|wi_0)"
    value_name_regex: str = r"(v_proj|to_v|add_v_proj|v_proj_moe_gen)"


MODEL_PRESETS: dict[str, PresetSpec] = {
    "qwen-image": PresetSpec(
        name="qwen-image",
        aliases=("qwenimage", "qwen_image", "qwen-image", "qwen-image-edit"),
        und_layer_paths=(
            # Text encoders used by Qwen-Image pipelines vary across local
            # checkouts.  These are hints; use --layer-paths for exact paths.
            "model.language_model.layers",
            "language_model.model.layers",
            "language_model.layers",
            "model.model.layers",
            "model.layers",
            "text_encoder.model.language_model.layers",
            "text_encoder.language_model.model.layers",
            "text_encoder.encoder.layers",
            "text_encoder.model.layers",
            "text_encoder.transformer.layers",
        ),
        gen_layer_paths=(
            "transformer.transformer_blocks",
            "transformer.blocks",
            "dit.transformer_blocks",
            "model.transformer_blocks",
            "model.diffusion_model.transformer_blocks",
        ),
        attn_name_regex=r"(attn|attention|self_attn|attn1|attn2|to_q|q_proj)",
        mlp_name_regex=r"(ff|ffn|mlp|feed_forward|feedforward)",
        value_name_regex=r"(to_v|add_v_proj|v_proj|v_proj_moe_gen)",
    ),
    "bagel": PresetSpec(
        name="bagel",
        aliases=("bagel", "bagel-vlm"),
        und_layer_paths=(
            "model.model.model.layers",
            "model.language_model.model.layers",
            "language_model.model.layers",
            "language_model.layers",
            "llm.model.layers",
            "model.language_model.model.layers",
            "model.llm.model.layers",
        ),
        gen_layer_paths=(
            "language_model.model.layers",
            "language_model.layers",
            "model.language_model.model.layers",
            "model.llm.model.layers",
            "image_decoder.layers",
            "image_decoder.blocks",
            "gen_model.layers",
            "gen_model.blocks",
            "generator.layers",
            "generator.blocks",
            "model.image_decoder.layers",
            "model.image_decoder.blocks",
        ),
        attn_name_regex=r"(self_attn|attn|attention|to_q|q_proj)",
        mlp_name_regex=r"(mlp|ffn|feed_forward|feedforward|ff)",
        value_name_regex=r"(v_proj_moe_gen|v_proj|to_v|add_v_proj)",
    ),
    "ming": PresetSpec(
        name="ming",
        aliases=("ming", "ming-lite", "ming-vl", "ming-vlm"),
        und_layer_paths=(
            "model.model.layers",
            "model.layers",
            "model.model.model.layers",
            "model.language_model.model.layers",
            "language_model.model.layers",
            "language_model.layers",
            "llm.model.layers",
            "model.llm.model.layers",
        ),
        gen_layer_paths=(
            "diffusion_loss.train_model.transformer.transformer_blocks",
            "diffusion_loss.pipelines.transformer.transformer_blocks",
            "diffusion_model.transformer_blocks",
            "image_model.transformer_blocks",
            "image_decoder.transformer_blocks",
            "generator.transformer_blocks",
            "model.diffusion_model.transformer_blocks",
            "model.image_decoder.transformer_blocks",
        ),
        attn_name_regex=r"(self_attn|attn|attention|attn1|attn2|to_q|q_proj)",
        mlp_name_regex=r"(mlp|ffn|feed_forward|feedforward|ff)",
        value_name_regex=r"(v_proj_moe_gen|v_proj|to_v|add_v_proj)",
    ),
    "janus": PresetSpec(
        name="janus",
        aliases=("janus", "janus-pro", "janus-pro-1b", "janus-pro-7b"),
        und_layer_paths=(
            "language_model.model.layers",
            "model.language_model.model.layers",
            "language_model.layers",
            "model.layers",
        ),
        gen_layer_paths=(
            "language_model.model.layers",
            "model.language_model.model.layers",
            "language_model.layers",
            "model.layers",
        ),
        attn_name_regex=r"(self_attn|attn|attention|q_proj)",
        mlp_name_regex=r"(mlp|ffn|feed_forward|feedforward)",
        value_name_regex=r"(v_proj|to_v|add_v_proj)",
    ),
}


def resolve_preset(name: str) -> PresetSpec:
    key = (name or "").strip().lower()
    for spec in MODEL_PRESETS.values():
        if key == spec.name or key in spec.aliases:
            return spec
    known = ", ".join(sorted(MODEL_PRESETS))
    raise KeyError(f"Unknown model preset {name!r}. Known presets: {known}")
