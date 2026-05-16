import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def _to_last_token(x):
    if x is None:
        return None
    if x.dim() >= 3:
        return x[:, -1:, :].detach()
    return x.detach()


class SublayerTraceCollector:
    """Runtime hook collector for attn/mlp sublayer transitions."""

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.current_step = None
        self.steps = []

    def start_step(self):
        self.current_step = {"attn": {}, "mlp": {}}

    def end_step(self):
        if self.current_step is not None:
            self.steps.append(self.current_step)
            self.current_step = None

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def attach(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            self._attach_layer_hooks(layer, layer_idx)

    def _attach_layer_hooks(self, layer, layer_idx):
        def layer_pre_hook(_module, inputs, kwargs):
            if self.current_step is None:
                return
            hidden = _to_last_token(inputs[0]) if inputs else None
            if hidden is not None:
                slot = self.current_step["attn"].setdefault(layer_idx, {})
                slot["residual"] = hidden

        self.handles.append(layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))

        if hasattr(layer, "self_attn") and layer.self_attn is not None:
            def attn_hook(_module, _inputs, output):
                if self.current_step is None:
                    return
                attn_out = output[0] if isinstance(output, tuple) else output
                attn_out = _to_last_token(attn_out)
                slot = self.current_step["attn"].setdefault(layer_idx, {})
                residual = slot.get("residual")
                if residual is not None and attn_out is not None:
                    slot["hidden_states"] = residual + attn_out

            self.handles.append(layer.self_attn.register_forward_hook(attn_hook))

        if hasattr(layer, "post_attention_layernorm") and layer.post_attention_layernorm is not None:
            def mlp_pre_hook(_module, inputs):
                if self.current_step is None:
                    return
                hidden = _to_last_token(inputs[0]) if inputs else None
                if hidden is not None:
                    slot = self.current_step["mlp"].setdefault(layer_idx, {})
                    slot["residual"] = hidden

            self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(mlp_pre_hook))

        if hasattr(layer, "mlp") and layer.mlp is not None:
            def mlp_hook(_module, _inputs, output):
                if self.current_step is None:
                    return
                mlp_out = output[0] if isinstance(output, tuple) else output
                mlp_out = _to_last_token(mlp_out)
                slot = self.current_step["mlp"].setdefault(layer_idx, {})
                residual = slot.get("residual")
                if residual is not None and mlp_out is not None:
                    slot["hidden_states"] = residual + mlp_out

            self.handles.append(layer.mlp.register_forward_hook(mlp_hook))


def apply_drop_masks(model, target_layer, drop_attn_list=None, drop_mlp_list=None, drop_n=None):
    drop_attn_set = set(drop_attn_list or [])
    drop_mlp_set = set(drop_mlp_list or [])
    for idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "drop_attn"):
            layer.drop_attn = idx in drop_attn_set if "attn" in target_layer else False
        if hasattr(layer, "drop_mlp"):
            layer.drop_mlp = idx in drop_mlp_set if "mlp" in target_layer else False
        if drop_n is not None and hasattr(layer, "drop_n"):
            layer.drop_n = drop_n


def _build_logits_warpers(temperature=0.0, top_k=0, top_p=1.0):
    warpers = LogitsProcessorList()
    if temperature is not None and temperature > 0 and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k > 0:
        warpers.append(TopKLogitsWarper(top_k))
    if top_p is not None and 0 < top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p))
    return warpers


def _sample_next_token(last_logits, input_ids, warpers, temperature=0.0):
    if temperature == 0.0:
        return torch.argmax(last_logits, dim=-1, keepdim=True)

    logits = warpers(input_ids, last_logits) if len(warpers) > 0 else last_logits
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _analysis_temperature(temperature):
    # Keep probability-space analysis numerically stable when generation is greedy.
    return temperature if (temperature is not None and temperature > 0) else 1.0


@torch.no_grad()
def generate_with_custom_forward(
    model,
    tokenizer,
    device,
    prompts=None,
    input_ids=None,
    max_length=512,
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    use_cache=True,
    collect_sublayer=False,
):
    if input_ids is None:
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    cur_ids = input_ids
    generated_logits = []
    generated_probs = []
    generated_hidden = []
    sublayer_steps = []

    max_new_tokens = max(0, max_length - cur_ids.shape[1])
    past_key_values = None
    collector = None
    if collect_sublayer:
        collector = SublayerTraceCollector(model)
        collector.attach()
    warpers = _build_logits_warpers(temperature=temperature, top_k=top_k, top_p=top_p)
    prob_temp = _analysis_temperature(temperature)

    try:
        for _ in range(max_new_tokens):
            model_inputs = {
                "input_ids": cur_ids if past_key_values is None else cur_ids[:, -1:],
                "use_cache": use_cache,
                "return_dict": True,
                "output_hidden_states": True,
            }
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values

            if collector is not None:
                collector.start_step()
            outputs = model(**model_inputs)
            if collector is not None:
                collector.end_step()

            last_logits = outputs.logits[:, -1, :]
            next_token = _sample_next_token(
                last_logits=last_logits,
                input_ids=cur_ids,
                warpers=warpers,
                temperature=temperature,
            )

            generated_logits.append(last_logits)
            generated_probs.append(F.softmax(last_logits / prob_temp, dim=-1))
            generated_hidden.append(outputs.hidden_states[-1][:, -1, :])

            cur_ids = torch.cat([cur_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
    finally:
        if collector is not None:
            sublayer_steps = collector.steps
            collector.close()

    texts = tokenizer.batch_decode(cur_ids, skip_special_tokens=True)
    return texts, generated_probs, cur_ids, generated_hidden, generated_logits, sublayer_steps


@torch.no_grad()
def forward_last_token(
    model,
    tokenizer,
    device,
    prompts=None,
    input_ids=None,
    use_cache=True,
    temperature=1.0,
):
    if input_ids is None:
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        return_dict=True,
        use_cache=use_cache,
    )

    logits = outputs.logits
    last_logits = logits[:, -1, :]
    prob_temp = _analysis_temperature(temperature)
    probabilities = F.softmax(last_logits / prob_temp, dim=-1)
    hidden_states = [h[:, -1, :] for h in outputs.hidden_states]
    texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return texts, probabilities, input_ids, hidden_states, logits


@torch.no_grad()
def forward_on_fixed_sequence(
    model,
    full_input_ids,
    prompt_len,
    collect_sublayer=False,
):
    """
    Evaluate next-token outputs on a fixed token trajectory.
    For step t, model sees full_input_ids[:, :t] and predicts token t.
    """
    probs = []
    logits = []
    sublayer_steps = []
    collector = None
    if collect_sublayer:
        collector = SublayerTraceCollector(model)
        collector.attach()

    try:
        total_len = full_input_ids.shape[1]
        for t in range(prompt_len, total_len):
            cur_ids = full_input_ids[:, :t]
            if collector is not None:
                collector.start_step()
            outputs = model(
                input_ids=cur_ids,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            if collector is not None:
                collector.end_step()

            step_logits = outputs.logits[:, -1, :]
            logits.append(step_logits)
            probs.append(F.softmax(step_logits, dim=-1))
    finally:
        if collector is not None:
            sublayer_steps = collector.steps
            collector.close()

    return probs, logits, sublayer_steps
