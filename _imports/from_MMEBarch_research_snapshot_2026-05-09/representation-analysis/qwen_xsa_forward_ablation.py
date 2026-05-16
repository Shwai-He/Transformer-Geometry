from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from packaging.version import InvalidVersion, Version
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

LEGACY_TRANSFORMERS_VERSION = "4.52.4"
QWEN35_TRANSFORMERS_VERSION_FLOOR = "5.6.2"


BUILTIN_TEXTS = [
    """Question: A warehouse has 18 shelves. Each shelf contains 24 boxes, and each box contains 6 notebooks. If 15% of the notebooks are damaged, how many usable notebooks remain? Answer: There are 18 * 24 * 6 = 2592 notebooks. Damaged notebooks are 0.15 * 2592 = 388.8, so approximately 2203 usable notebooks remain if rounded down to whole notebooks.""",
    """Instruction: Summarize the following policy memo in two sentences. Memo: The city plans to replace diesel buses with electric buses over five years, prioritizing routes near schools and hospitals. The plan requires charging depots, driver training, revised maintenance contracts, and an interim budget increase, but it is expected to reduce noise and local air pollution.""",
    """We need to debug a Python function. The function reads a JSONL file, extracts a field named text, skips empty rows, and returns a list of strings. A common bug is forgetting to strip whitespace before checking whether a row is empty, which causes malformed blank records to be parsed.""",
    """In a residual transformer block, the hidden state can be written as x_{l+1} = x_l + f_l(LN(x_l)). If f_l contains a component parallel to x_l and a component orthogonal to x_l, then removing the parallel component changes both the magnitude and direction of the next hidden state.""",
    """Translate and explain: 这个实验的关键不是只看 cosine similarity，而是要同时检查 loss、parallel ratio、perpendicular magnitude 和 layer-wise trend. English: The key point of this experiment is not only to inspect cosine similarity, but also to check loss, parallel ratio, perpendicular magnitude, and the layer-wise trend.""",
    """A researcher compares four interventions on the same validation set: baseline, attention-only projection removal, MLP-only projection removal, and both. If attention-only improves loss but MLP-only degrades loss, the combined intervention may still degrade because the MLP damage dominates the attention benefit.""",
    """Consider this SQL query: SELECT user_id, COUNT(*) AS n FROM events WHERE event_type = 'click' GROUP BY user_id HAVING COUNT(*) > 10 ORDER BY n DESC. The WHERE clause filters rows before aggregation, while the HAVING clause filters groups after aggregation.""",
    """The committee rejected the proposal not because the measurements were noisy, but because the ablation changed two variables at once. A cleaner experiment would keep the model, data order, batch size, and optimizer fixed while modifying only the forward transformation under study.""",
    """Suppose an autoregressive model assigns probabilities 0.50, 0.25, 0.125, and 0.125 to the next four possible tokens. The entropy is lower than a uniform distribution over four tokens because the probability mass is concentrated on the first token.""",
    """Write a concise implementation plan: first add a hook that stores the residual input to each decoder layer; second modify the attention output by subtracting its projection on that residual; third evaluate loss on a fixed text set; fourth compare the result against the unmodified baseline.""",
    """A long-context example: The first paragraph introduces a library API. The second paragraph describes a regression in version 2.1. The third paragraph says the regression only appears when caching is enabled and batch size is greater than one. Therefore, a correct diagnosis should mention cache interaction rather than blaming tokenization alone.""",
    """数学推理：如果一个向量 y 可以分解为 y_parallel + y_perp，其中 y_parallel 与参考向量 x 平行，y_perp 与 x 正交，那么去掉 y_parallel 会降低 y 在 x 方向上的投影，但不一定降低最终状态 x + y 与 x 的 cosine similarity，因为最终状态仍然包含原来的 x.""",
    """Code review note: The hook should return a modified tuple when the original module output is a tuple; otherwise downstream code that expects attention weights or cache objects can break. It should also avoid detaching the replacement tensor during evaluation if gradients are later needed.""",
    """A product manager asks whether a metric improvement is real. The right answer is to run the same ablation over a larger validation set, report confidence intervals or at least multiple seeds when training is involved, and verify that the improvement is not caused by a data preprocessing mismatch.""",
    """In Qwen-style decoder layers, attention and MLP sublayers are separated by RMSNorm modules. This means a change to the attention branch can alter the input distribution seen by the following MLP, so an attention-only intervention may have downstream effects beyond the immediate residual update.""",
    """The model reads: Alice gave Bob three red marbles and two blue marbles. Bob then gave one red marble to Clara. If nobody else changed the marbles, Bob now has two red marbles and two blue marbles. This requires tracking entities rather than matching keywords.""",
    """Formal claim: If an operation removes only the component of f(x) parallel to x, then it preserves the component of f(x) in the orthogonal subspace of x. However, the operation does not generally preserve logits, because subsequent normalization and nonlinear layers can amplify or suppress different directions.""",
    """A bilingual instruction: 请用英文回答，并指出这个实验为什么需要 baseline. Answer in English: The baseline is required because the projection removal changes the forward computation, and only a matched unmodified run tells us whether the change improves or degrades the original model likelihood.""",
    """For a unit test, create a random tensor y and a reference tensor x, subtract proj_x(y), and verify that the dot product between the modified y and x is close to zero along the hidden dimension. Also verify that tensor shapes and dtypes are unchanged.""",
    """The ablation result should be interpreted carefully: a lower validation loss after removing an attention-parallel component suggests that this component is not always useful, but it does not prove that all parallel components are harmful or that the same result will hold for MLP layers."""
]

BUILTIN_TEXTS.extend(
    [
        """Long technical passage: A forward-only XSA ablation is different from adding an auxiliary training loss. In the forward-only setting, the branch output is transformed immediately before it is added to the residual stream, so the next layer receives a modified hidden state on every token and every layer where the hook is active. In the auxiliary-loss setting, the original forward computation remains unchanged, and the optimizer only receives an additional gradient signal that competes with the task loss, weight decay, gradient clipping, learning-rate schedule, and any residual gates. Therefore, if forward-only removal improves validation loss but the loss term does not, the likely explanation is not merely implementation error; it may be that direct geometric surgery changes inference-time computation in a way that the optimizer does not discover under the same training budget.""",
        """Long reasoning example: A hospital scheduling system assigns nurses to shifts under three constraints. First, no nurse can work more than four consecutive night shifts. Second, every intensive-care shift must include at least one senior nurse. Third, nurses who attended training on Monday cannot be assigned to the Monday evening shift. If the schedule violates the second constraint but satisfies the first and third constraints, the correct diagnosis is not that the whole schedule is invalid for all reasons. The specific failure is that at least one intensive-care shift lacks a senior nurse, and the repair should target the senior-nurse assignment rather than reducing all night shifts.""",
        """Long code-oriented prompt: Consider a PyTorch hook registered with register_forward_hook(..., with_kwargs=True). The hook receives module, args, kwargs, and output. If the original output is a tuple, replacing it with a tensor can silently break downstream code because later modules may expect cache objects, attention weights, or auxiliary outputs. A robust hook should modify only output[0] when that first element is the hidden-state tensor, then reconstruct the tuple as (modified_hidden,) + output[1:]. It should also check that the residual tensor and branch tensor have the same shape before subtracting a projection, because grouped-query attention or cache-only decoding can otherwise create shape mismatches.""",
        """Long scientific interpretation: Suppose a pretrained decoder has learned to use attention primarily for contextual mixing and MLP layers primarily for feature transformation and calibration. Removing the residual-parallel component from attention may help if that component duplicates information already carried by the residual stream, especially in deep layers where attention could over-amplify already-present features. Removing the same kind of component from MLP can hurt if the MLP uses the parallel direction to adjust feature magnitude before the final normalization and language-model head. Under this interpretation, the sign of the effect depends on which branch is modified, where in depth it is modified, and whether later normalization can compensate for the removed component.""",
        """Long multilingual sample: 这个测试集合故意混合了英文、中文、数学描述、代码解释和实验分析，因为一个很短的 prompt set 容易给出不稳定的结论。For example, an attention-only ablation may look helpful on a handful of short factual sentences, but the same intervention should also be checked on longer instructions, multi-step reasoning, and implementation details. 如果结果在更长的文本上仍然稳定，那么我们才更有理由相信这个 forward-only geometric modification is capturing a real property of the model rather than noise from a tiny evaluation batch.""",
        """Long analytical prompt: A committee compares two explanations for why an ablation improves loss. Explanation A says that the removed component is useless noise. Explanation B says that the component is useful in some contexts but harmful on average because it competes with a cleaner orthogonal update. To distinguish these explanations, the committee proposes a layer-wise experiment: remove the component only in early layers, only in middle layers, only in late layers, and then compare the loss deltas. If late-layer removal helps while early-layer removal hurts, then the result supports a depth-dependent interpretation rather than a universal claim that all parallel components are noise.""",
        """Long math-style prompt: Let y be a branch update and x be the residual stream at the same token position. The projection of y onto x is proj_x(y) = <y, x> x / ||x||^2, and the perpendicular part is y_perp = y - proj_x(y). By construction, <y_perp, x> is zero up to numerical precision. However, the cosine similarity between x and x + y_perp can still be high, because x remains present in the residual state. This distinction matters: the ablation controls the branch update direction, not the final hidden state direction after the residual addition.""",
        """Long product-and-data prompt: A search ranking team observes that a new model has better click-through rate on mobile but worse conversion rate on desktop. The team should not average the two numbers and declare victory without checking traffic mix, confidence intervals, and whether the desktop drop is concentrated in a specific browser version. Similarly, a representation ablation that improves loss on a small internal prompt list should be treated as a hypothesis generator. The next step is to evaluate on a larger held-out corpus, report token counts, and check whether the improvement persists across model sizes and prompt categories.""",
        """Long implementation plan: First, load the model once and run the baseline loss without hooks. Second, attach attention hooks that store the input hidden state to each decoder layer and subtract the projection of the attention output onto that stored residual. Third, remove the hooks and repeat the evaluation for MLP-only hooks. Fourth, run the combined setting only after the single-branch settings are understood, because a combined degradation can be caused by one harmful branch dominating one helpful branch. Fifth, write all metrics to JSON, including loss, perplexity, token count, and per-layer projection ratios, so the result can be compared across runs.""",
        """Long counterexample: It is tempting to say that lowering the branch parallel ratio must lower the final state cosine similarity, but this is false. If the residual stream x is much larger than the branch update y, then x + y and x can remain highly aligned even when y is purely perpendicular to x. Conversely, if y has a large negative component parallel to x, removing that component can increase the final cosine similarity by preventing cancellation. Therefore, final-state cosine and branch projection ratio measure related but non-equivalent properties, and both should be interpreted with the update magnitude."""
    ]
)


ATTN_MATRIX_TEXTS = [
    """Copy-focused probe: Repeat the exact identifier once and do not explain it. Identifier: ZX-41-KAPPA. Answer:""",
    """Delimiter probe: Complete only the missing closing symbol. Expression: function_call(alpha, beta, gamma[3], {"key": "value" Answer:""",
    """Entity-binding probe: Alice lent Bob her green umbrella. Two minutes later, Bob handed the green umbrella to Clara. Question: Who had the green umbrella immediately before Clara? Answer:""",
    """Local self-reference probe: Rewrite the final word in uppercase and output only that word. Sentence: The debug flag should remain enabled. Final word:""",
    """Repetition probe: Continue the pattern with one more item only. Pattern: red, blue, red, blue, red, blue,""",
    """Needle probe: Return the secret code exactly as written and nothing else. Context: ignore the filler words and keep only the secret code AX7Q-19. Secret code:""",
    """Quote-closure probe: Output the shortest valid continuation. Text: The reviewer wrote, "the ablation changed the diagonal term because""",
    """Positional overwrite probe: In the sequence A1 B2 C3 D4 E5, replace only the third item with C9 and print the full updated sequence.""",
    """Coreference probe: Sarah told Mina that she would submit the patch after she reran the tests. If 'she' in the second clause refers to Sarah, who reruns the tests? Answer:""",
    """Self-token sensitivity probe: Echo the final token exactly once. Tokens: cache, compile, residual, parallel. Final token:""",
    """Attention-mixing probe: The first sentence states a hypothesis. The second sentence gives a counterexample. The third sentence asks for the verdict. Hypothesis: removing a parallel component always lowers state cosine. Counterexample: if the residual dominates, cosine can stay high. Verdict:""",
    """Name-tracking probe: Tom gave Eva a silver key. Eva put the silver key into a wooden box. Later Eva removed the silver key and gave it to Noah. Question: Who has the silver key now? Answer:""",
]


BUILTIN_TEXT_SETS = {
    "builtin": BUILTIN_TEXTS,
    "attn_matrix": ATTN_MATRIX_TEXTS,
}


def _resolve_dtype(name: str):
    name = name.lower()
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _require_transformers_version(model_name_or_path: Optional[str] = None) -> None:
    got_raw = transformers.__version__
    try:
        got = Version(got_raw)
        legacy = Version(LEGACY_TRANSFORMERS_VERSION)
        qwen35_floor = Version(QWEN35_TRANSFORMERS_VERSION_FLOOR)
    except InvalidVersion:
        raise RuntimeError(
            "Could not parse transformers version "
            f"{got_raw!r}. Expected {LEGACY_TRANSFORMERS_VERSION} or >= {QWEN35_TRANSFORMERS_VERSION_FLOOR}."
        )

    if got == legacy or got >= qwen35_floor:
        return

    model_name = (model_name_or_path or "").lower()
    if "qwen3.5" in model_name or "qwen3_5" in model_name:
        raise RuntimeError(
            "Qwen3.5 support requires transformers>="
            f"{QWEN35_TRANSFORMERS_VERSION_FLOOR}, but found {got_raw}. "
            f"Run through the shell wrapper or install transformers>={QWEN35_TRANSFORMERS_VERSION_FLOOR}."
        )
    raise RuntimeError(
        "This Qwen XSA script supports transformers=="
        f"{LEGACY_TRANSFORMERS_VERSION} or >= {QWEN35_TRANSFORMERS_VERSION_FLOOR}, but found {got_raw}. "
        "Run through the shell wrapper or install a supported transformers version."
    )


def _patch_transformers_tp_plan_check() -> bool:
    """Work around Transformers 4.52.x Qwen3 TP-plan checks on torch<2.5.

    Some Qwen3 classes define a default `_tp_plan`. In affected Transformers
    versions, `PreTrainedModel.post_init()` validates that plan against
    `ALL_PARALLEL_STYLES` when torch>=2.3, but `ALL_PARALLEL_STYLES` is only
    initialized for torch>=2.5. We do not use HF tensor parallelism in this
    eval script, so a membership-only set is sufficient for model init.
    """
    try:
        import transformers.modeling_utils as modeling_utils
    except Exception:
        return False

    if getattr(modeling_utils, "ALL_PARALLEL_STYLES", None) is not None:
        return False

    modeling_utils.ALL_PARALLEL_STYLES = {
        "colwise",
        "rowwise",
        "colwise_rep",
        "rowwise_rep",
        "local_colwise",
        "local_rowwise",
        "local",
        "gather",
        "local_packed_rowwise",
        "local_packed_colwise",
        "sequence_parallel",
        "replicate",
    }
    return True


def _find_decoder_layers(model) -> List[torch.nn.Module]:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for parent_name, layers_name in candidates:
        parent = getattr(model, parent_name, None)
        layers = getattr(parent, layers_name, None) if parent is not None else None
        if layers is not None:
            return list(layers)
    raise ValueError("Could not find decoder layers. Expected model.layers, transformer.h, or gpt_neox.layers.")


def _extract_hidden(args, kwargs):
    return kwargs.get("hidden_states", args[0] if args else None)


def _get_token_mixer_kind_and_module(layer) -> Tuple[Optional[str], Optional[torch.nn.Module]]:
    if hasattr(layer, "self_attn"):
        return "full_attention", layer.self_attn
    if hasattr(layer, "linear_attn"):
        return "linear_attention", layer.linear_attn
    return None, None


def _remove_parallel_and_stats(
    y: torch.Tensor,
    ref: torch.Tensor,
    eps: float = 1e-6,
    alpha: float = 1.0,
    perp_scale: float = 1.0,
):
    y_f = y.float()
    r_f = ref.float()
    dot = (y_f * r_f).sum(dim=-1, keepdim=True)
    ref_sq = (r_f * r_f).sum(dim=-1, keepdim=True).clamp_min(eps)
    coeff = dot / ref_sq
    proj = coeff * r_f
    perp = y_f - proj

    proj_sq = (proj * proj).sum(dim=-1)
    perp_sq = (perp * perp).sum(dim=-1)
    y_sq = (y_f * y_f).sum(dim=-1).clamp_min(eps)
    ref_norm = torch.sqrt(ref_sq.squeeze(-1))
    y_norm = torch.sqrt(y_sq)

    stats = {
        "para_ratio": (proj_sq / y_sq).detach(),
        "perp_ratio": (perp_sq / y_sq).detach(),
        "y_over_ref": (y_norm / ref_norm.clamp_min(eps)).detach(),
        "perp_over_ref": (torch.sqrt(perp_sq.clamp_min(0.0)) / ref_norm.clamp_min(eps)).detach(),
        "para_over_ref": (torch.sqrt(proj_sq.clamp_min(0.0)) / ref_norm.clamp_min(eps)).detach(),
    }
    # y = proj + perp; apply directional scales:
    # y_new = perp_scale * perp + (1 - alpha) * proj
    y_new = float(perp_scale) * perp + (1.0 - float(alpha)) * proj
    return y_new.to(dtype=y.dtype), stats


def _replace_first_arg(args, kwargs, new_value):
    if args:
        return (new_value,) + tuple(args[1:]), kwargs
    new_kwargs = dict(kwargs)
    new_kwargs["input"] = new_value
    return args, new_kwargs


def _expand_attn_value_ref(value: torch.Tensor, y_pre: torch.Tensor, attn_module) -> Optional[torch.Tensor]:
    if value.shape == y_pre.shape:
        return value
    if value.dim() != 3 or y_pre.dim() != 3:
        return None
    if value.shape[:2] != y_pre.shape[:2]:
        return None

    head_dim = getattr(attn_module, "head_dim", None)
    num_heads = getattr(attn_module, "num_heads", None)
    num_key_value_heads = getattr(attn_module, "num_key_value_heads", None)
    if head_dim is None or num_heads is None or num_key_value_heads is None:
        if y_pre.size(-1) % value.size(-1) != 0:
            return None
        group = y_pre.size(-1) // value.size(-1)
        return value.repeat_interleave(group, dim=-1)

    if value.size(-1) != num_key_value_heads * head_dim:
        return None
    if y_pre.size(-1) != num_heads * head_dim:
        return None
    if num_heads % num_key_value_heads != 0:
        return None

    group = num_heads // num_key_value_heads
    ref = value.reshape(*value.shape[:-1], num_key_value_heads, head_dim)
    ref = ref.repeat_interleave(group, dim=-2)
    return ref.reshape_as(y_pre)


def _remove_parallel_attn_multihead(
    y_pre: torch.Tensor,
    ref: torch.Tensor,
    attn_module,
    alpha: float = 1.0,
    perp_scale: float = 1.0,
):
    head_dim = getattr(attn_module, "head_dim", None)
    num_heads = getattr(attn_module, "num_heads", None)
    return _remove_parallel_multihead(
        y_pre, ref, num_heads=num_heads, head_dim=head_dim, alpha=alpha, perp_scale=perp_scale
    )


def _remove_parallel_multihead(
    y_pre: torch.Tensor,
    ref: torch.Tensor,
    *,
    num_heads: Optional[int],
    head_dim: Optional[int],
    alpha: float = 1.0,
    perp_scale: float = 1.0,
):
    if head_dim is None or num_heads is None:
        return _remove_parallel_and_stats(y_pre, ref, alpha=alpha, perp_scale=perp_scale)
    if y_pre.dim() != 3 or ref.dim() != 3:
        return _remove_parallel_and_stats(y_pre, ref, alpha=alpha, perp_scale=perp_scale)
    if y_pre.size(-1) != num_heads * head_dim or ref.size(-1) != num_heads * head_dim:
        return _remove_parallel_and_stats(y_pre, ref, alpha=alpha, perp_scale=perp_scale)

    y_heads = y_pre.reshape(*y_pre.shape[:-1], num_heads, head_dim)
    ref_heads = ref.reshape(*ref.shape[:-1], num_heads, head_dim)
    new_heads, stats = _remove_parallel_and_stats(y_heads, ref_heads, alpha=alpha, perp_scale=perp_scale)
    return new_heads.reshape_as(y_pre), stats


def _extract_linear_value_ref(mixed_qkv: torch.Tensor, linear_module) -> Optional[torch.Tensor]:
    value_dim = getattr(linear_module, "value_dim", None)
    key_dim = getattr(linear_module, "key_dim", None)
    if not isinstance(mixed_qkv, torch.Tensor):
        return None
    if mixed_qkv.dim() != 3:
        return None
    if value_dim is None or key_dim is None:
        return None
    qkv_dim = key_dim * 2 + value_dim
    if mixed_qkv.size(-1) != qkv_dim:
        return None
    return mixed_qkv[..., 2 * key_dim :]


class RunningStats:
    def __init__(self, enabled: bool = False, track_layerwise: bool = False):
        self.enabled = bool(enabled)
        self.track_layerwise = bool(track_layerwise)
        self.global_sums = defaultdict(float)
        self.global_counts = defaultdict(int)
        self.global_mins = {}
        self.global_maxs = {}
        self.layer_sums = defaultdict(float)
        self.layer_counts = defaultdict(int)
        self.layer_mins = {}
        self.layer_maxs = {}

    def update_tensor(self, branch: str, layer_idx: int, name: str, value: torch.Tensor) -> None:
        if not self.enabled:
            return
        v = value.detach().float()
        v_min = float(v.min().item())
        v_max = float(v.max().item())
        total = float(v.sum().item())
        count = int(v.numel())

        global_key = (branch, name)
        self.global_sums[global_key] += total
        self.global_counts[global_key] += count
        self.global_mins[global_key] = v_min if global_key not in self.global_mins else min(self.global_mins[global_key], v_min)
        self.global_maxs[global_key] = v_max if global_key not in self.global_maxs else max(self.global_maxs[global_key], v_max)

        if self.track_layerwise:
            layer_key = (branch, int(layer_idx), name)
            self.layer_sums[layer_key] += total
            self.layer_counts[layer_key] += count
            self.layer_mins[layer_key] = v_min if layer_key not in self.layer_mins else min(self.layer_mins[layer_key], v_min)
            self.layer_maxs[layer_key] = v_max if layer_key not in self.layer_maxs else max(self.layer_maxs[layer_key], v_max)

    def summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        if not self.enabled:
            return {}
        layer_branches = {branch for (branch, _, _) in self.layer_sums.keys()}
        global_branches = {branch for (branch, _) in self.global_sums.keys()}
        all_branches = sorted(layer_branches | global_branches)

        layers: Dict[str, Dict[str, Dict[str, float]]] = {branch: {} for branch in all_branches}
        if self.track_layerwise:
            for (branch, layer_idx, name), total in sorted(self.layer_sums.items()):
                count = max(1, self.layer_counts[(branch, layer_idx, name)])
                layer_stats = layers.setdefault(branch, {}).setdefault(str(layer_idx), {})
                layer_stats[name] = total / count
                layer_stats[f"{name}_min"] = self.layer_mins[(branch, layer_idx, name)]
                layer_stats[f"{name}_max"] = self.layer_maxs[(branch, layer_idx, name)]

        overall: Dict[str, Dict[str, float]] = {branch: {} for branch in all_branches}
        for (branch, name), total in self.global_sums.items():
            count = max(1, self.global_counts[(branch, name)])
            v_min = self.global_mins[(branch, name)]
            v_max = self.global_maxs[(branch, name)]
            branch_stats = overall.setdefault(branch, {})
            branch_stats[name] = total / count
            branch_stats[f"{name}_min"] = v_min
            branch_stats[f"{name}_max"] = v_max
        return {"overall": overall, "layers": layers}


class QwenXSAForwardHooks:
    """Forward-only removal of branch-parallel components.

    ``residual_output`` is the older approximation: residual input x -> final
    token-mixer or MLP output y_post. ``xsa_middle`` follows the XSA-style
    middle intervention on the merged-head/value representation before the
    output projection. ``xsa_middle_multihead`` applies the same removal per
    head for token mixers that expose a head structure.
    """

    def __init__(
        self,
        model,
        target: str,
        start_layer: int = 0,
        end_layer: int = -1,
        skip_first_n: int = 0,
        skip_last_n: int = 0,
        intervention_site: str = "xsa_middle_multihead",
        xsa_alpha: float = 1.0,
        xsa_perp_scale: float = 1.0,
        track_stats: bool = False,
        track_layerwise_stats: bool = False,
    ):
        if target not in {"attn", "mlp", "both"}:
            raise ValueError(f"target must be attn/mlp/both, got {target}")
        if intervention_site not in {"residual_output", "xsa_middle", "xsa_middle_multihead"}:
            raise ValueError(
                f"intervention_site must be residual_output/xsa_middle/xsa_middle_multihead, got {intervention_site}"
            )
        if intervention_site in {"xsa_middle", "xsa_middle_multihead"} and target != "attn":
            raise ValueError(
                f"{intervention_site} is attention-only and only supports target=attn, got target={target}"
            )
        self.model = model
        self.target = target
        self.intervention_site = intervention_site
        self.xsa_alpha = float(xsa_alpha)
        self.xsa_perp_scale = float(xsa_perp_scale)
        self.track_stats = bool(track_stats)
        self.track_layerwise_stats = bool(track_layerwise_stats)
        self.layers = _find_decoder_layers(model)
        n_layers = len(self.layers)
        lo = max(0, int(start_layer), int(skip_first_n))
        hi = n_layers if int(end_layer) < 0 else min(n_layers, int(end_layer))
        hi = min(hi, max(0, n_layers - int(skip_last_n)))
        if hi < lo:
            hi = lo
        self.active_layer_indices = set(range(lo, hi))
        self.layer_window = {
            "start_layer": lo,
            "end_layer_exclusive": hi,
            "skip_first_n": int(skip_first_n),
            "skip_last_n": int(skip_last_n),
            "n_layers": n_layers,
            "n_active_layers": len(self.active_layer_indices),
            "intervention_site": self.intervention_site,
            "xsa_alpha": self.xsa_alpha,
            "xsa_perp_scale": self.xsa_perp_scale,
            "track_stats": self.track_stats,
            "track_layerwise_stats": self.track_layerwise_stats,
            "intervention_pair": (
                "x_to_y_post"
                if self.intervention_site == "residual_output"
                else (
                    "token_mixer_value_to_out_proj_input_merged_heads"
                    if self.intervention_site == "xsa_middle"
                    else "token_mixer_value_to_out_proj_input_multihead"
                )
            ),
        }
        self.handles = []
        self.attn_residual = {}
        self.mlp_residual = {}
        self.attn_value = {}
        self.stats = RunningStats(enabled=self.track_stats, track_layerwise=self.track_layerwise_stats)

    def _record_stats(self, branch: str, layer_idx: int, stats: Dict[str, torch.Tensor]) -> None:
        for name, value in stats.items():
            self.stats.update_tensor(branch, layer_idx, name, value)

    def attach(self) -> None:
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx not in self.active_layer_indices:
                continue
            mixer_kind, mixer_module = _get_token_mixer_kind_and_module(layer)
            if self.target in {"attn", "both"} and mixer_module is None:
                raise ValueError(
                    f"Layer {layer_idx} is missing a supported token mixer (self_attn or linear_attn)."
                )
            if self.target in {"mlp", "both"} and not hasattr(layer, "mlp"):
                raise ValueError(f"Layer {layer_idx} is missing mlp; not a supported Qwen-style layer.")

            if self.intervention_site == "residual_output":
                def layer_pre_hook(_mod, args, kwargs, _idx=layer_idx):
                    hidden = _extract_hidden(args, kwargs)
                    if hidden is not None:
                        self.attn_residual[_idx] = hidden.detach()

                self.handles.append(layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))

            if self.target in {"attn", "both"}:
                if mixer_kind == "full_attention" and self.intervention_site == "residual_output":
                    v_proj = getattr(mixer_module, "v_proj", None)
                    o_proj = getattr(mixer_module, "o_proj", None)
                    if v_proj is not None and o_proj is not None:
                        def v_proj_stats_hook(_mod, _args, _kwargs, output, _idx=layer_idx):
                            if isinstance(output, torch.Tensor):
                                self.attn_value[_idx] = output.detach()
                            return output

                        def o_proj_stats_pre_hook(_mod, args, kwargs, _idx=layer_idx, _attn=mixer_module):
                            y_pre = args[0] if args else kwargs.get("input", None)
                            value = self.attn_value.get(_idx)
                            if not isinstance(y_pre, torch.Tensor) or not isinstance(value, torch.Tensor):
                                return None
                            ref = _expand_attn_value_ref(value, y_pre, _attn)
                            if ref is None or ref.shape != y_pre.shape:
                                return None
                            _, stats = _remove_parallel_and_stats(
                                y_pre, ref, alpha=self.xsa_alpha, perp_scale=self.xsa_perp_scale
                            )
                            self._record_stats("attn_pre_o_proj", _idx, stats)
                            self._record_stats("token_mixer_pre_out_proj", _idx, stats)
                            return None

                        self.handles.append(v_proj.register_forward_hook(v_proj_stats_hook, with_kwargs=True))
                        self.handles.append(o_proj.register_forward_pre_hook(o_proj_stats_pre_hook, with_kwargs=True))

                    def attn_hook(_mod, args, kwargs, output, _idx=layer_idx):
                        residual = self.attn_residual.get(_idx)
                        if residual is None:
                            return output
                        attn_out = output[0] if isinstance(output, tuple) else output
                        if not isinstance(attn_out, torch.Tensor) or attn_out.shape != residual.shape:
                            return output
                        new_attn, stats = _remove_parallel_and_stats(
                            attn_out, residual, alpha=self.xsa_alpha, perp_scale=self.xsa_perp_scale
                        )
                        self._record_stats("attn", _idx, stats)
                        self._record_stats("attn_post_o_proj", _idx, stats)
                        self._record_stats("token_mixer", _idx, stats)
                        self._record_stats("token_mixer_post_out_proj", _idx, stats)
                        if isinstance(output, tuple):
                            return (new_attn,) + output[1:]
                        return new_attn

                    self.handles.append(mixer_module.register_forward_hook(attn_hook, with_kwargs=True))
                elif mixer_kind == "full_attention":
                    v_proj = getattr(mixer_module, "v_proj", None)
                    o_proj = getattr(mixer_module, "o_proj", None)
                    if v_proj is None or o_proj is None:
                        raise ValueError(
                            f"Layer {layer_idx} attention lacks v_proj/o_proj; cannot run {self.intervention_site}."
                        )

                    def v_proj_hook(_mod, _args, _kwargs, output, _idx=layer_idx):
                        if isinstance(output, torch.Tensor):
                            self.attn_value[_idx] = output.detach()
                        return output

                    def o_proj_pre_hook(_mod, args, kwargs, _idx=layer_idx, _attn=mixer_module):
                        y_pre = args[0] if args else kwargs.get("input", None)
                        value = self.attn_value.get(_idx)
                        if not isinstance(y_pre, torch.Tensor) or not isinstance(value, torch.Tensor):
                            return None
                        ref = _expand_attn_value_ref(value, y_pre, _attn)
                        if ref is None or ref.shape != y_pre.shape:
                            return None
                        if self.intervention_site == "xsa_middle":
                            new_y_pre, stats = _remove_parallel_and_stats(
                                y_pre, ref, alpha=self.xsa_alpha, perp_scale=self.xsa_perp_scale
                            )
                        else:
                            new_y_pre, stats = _remove_parallel_attn_multihead(
                                y_pre, ref, _attn, alpha=self.xsa_alpha, perp_scale=self.xsa_perp_scale
                            )
                        self._record_stats("attn", _idx, stats)
                        self._record_stats("attn_pre_o_proj", _idx, stats)
                        self._record_stats("token_mixer", _idx, stats)
                        self._record_stats("token_mixer_pre_out_proj", _idx, stats)
                        return _replace_first_arg(args, kwargs, new_y_pre)

                    self.handles.append(v_proj.register_forward_hook(v_proj_hook, with_kwargs=True))
                    self.handles.append(o_proj.register_forward_pre_hook(o_proj_pre_hook, with_kwargs=True))
                elif mixer_kind == "linear_attention" and self.intervention_site == "residual_output":
                    qkv_proj = getattr(mixer_module, "in_proj_qkv", None)
                    out_proj = getattr(mixer_module, "out_proj", None)
                    if qkv_proj is not None and out_proj is not None:
                        def qkv_stats_hook(_mod, _args, _kwargs, output, _idx=layer_idx, _mixer=mixer_module):
                            value = _extract_linear_value_ref(output, _mixer)
                            if isinstance(value, torch.Tensor):
                                self.attn_value[_idx] = value.detach()
                            return output

                        def linear_out_proj_stats_pre_hook(_mod, args, kwargs, _idx=layer_idx):
                            y_pre = args[0] if args else kwargs.get("input", None)
                            value = self.attn_value.get(_idx)
                            if not isinstance(y_pre, torch.Tensor) or not isinstance(value, torch.Tensor):
                                return None
                            if value.shape != y_pre.shape:
                                return None
                            _, stats = _remove_parallel_and_stats(
                                y_pre, value, alpha=self.xsa_alpha, perp_scale=self.xsa_perp_scale
                            )
                            self._record_stats("linear_pre_out_proj", _idx, stats)
                            self._record_stats("token_mixer_pre_out_proj", _idx, stats)
                            return None

                        self.handles.append(qkv_proj.register_forward_hook(qkv_stats_hook, with_kwargs=True))
                        self.handles.append(out_proj.register_forward_pre_hook(linear_out_proj_stats_pre_hook, with_kwargs=True))

                    def linear_hook(_mod, args, kwargs, output, _idx=layer_idx):
                        residual = self.attn_residual.get(_idx)
                        if residual is None:
                            return output
                        linear_out = output[0] if isinstance(output, tuple) else output
                        if not isinstance(linear_out, torch.Tensor) or linear_out.shape != residual.shape:
                            return output
                        new_linear, stats = _remove_parallel_and_stats(
                            linear_out, residual, alpha=self.xsa_alpha, perp_scale=self.xsa_perp_scale
                        )
                        self._record_stats("linear", _idx, stats)
                        self._record_stats("linear_post_out_proj", _idx, stats)
                        self._record_stats("token_mixer", _idx, stats)
                        self._record_stats("token_mixer_post_out_proj", _idx, stats)
                        if isinstance(output, tuple):
                            return (new_linear,) + output[1:]
                        return new_linear

                    self.handles.append(mixer_module.register_forward_hook(linear_hook, with_kwargs=True))
                elif mixer_kind == "linear_attention":
                    qkv_proj = getattr(mixer_module, "in_proj_qkv", None)
                    out_proj = getattr(mixer_module, "out_proj", None)
                    if qkv_proj is None or out_proj is None:
                        raise ValueError(
                            f"Layer {layer_idx} linear attention lacks in_proj_qkv/out_proj; cannot run {self.intervention_site}."
                        )

                    def qkv_hook(_mod, _args, _kwargs, output, _idx=layer_idx, _mixer=mixer_module):
                        value = _extract_linear_value_ref(output, _mixer)
                        if isinstance(value, torch.Tensor):
                            self.attn_value[_idx] = value.detach()
                        return output

                    def linear_out_proj_pre_hook(_mod, args, kwargs, _idx=layer_idx, _mixer=mixer_module):
                        y_pre = args[0] if args else kwargs.get("input", None)
                        value = self.attn_value.get(_idx)
                        if not isinstance(y_pre, torch.Tensor) or not isinstance(value, torch.Tensor):
                            return None
                        if value.shape != y_pre.shape:
                            return None
                        if self.intervention_site == "xsa_middle":
                            new_y_pre, stats = _remove_parallel_and_stats(
                                y_pre, value, alpha=self.xsa_alpha, perp_scale=self.xsa_perp_scale
                            )
                        else:
                            new_y_pre, stats = _remove_parallel_multihead(
                                y_pre,
                                value,
                                num_heads=getattr(_mixer, "num_v_heads", None),
                                head_dim=getattr(_mixer, "head_v_dim", None),
                                alpha=self.xsa_alpha,
                                perp_scale=self.xsa_perp_scale,
                            )
                        self._record_stats("linear", _idx, stats)
                        self._record_stats("linear_pre_out_proj", _idx, stats)
                        self._record_stats("token_mixer", _idx, stats)
                        self._record_stats("token_mixer_pre_out_proj", _idx, stats)
                        return _replace_first_arg(args, kwargs, new_y_pre)

                    self.handles.append(qkv_proj.register_forward_hook(qkv_hook, with_kwargs=True))
                    self.handles.append(out_proj.register_forward_pre_hook(linear_out_proj_pre_hook, with_kwargs=True))

            if self.target in {"mlp", "both"}:
                norm = getattr(layer, "post_attention_layernorm", None)
                if norm is None:
                    raise ValueError(f"Layer {layer_idx} has no post_attention_layernorm; cannot capture MLP residual.")

                def mlp_pre_hook(_mod, args, kwargs, _idx=layer_idx):
                    hidden = args[0] if args else kwargs.get("hidden_states", None)
                    if hidden is not None:
                        self.mlp_residual[_idx] = hidden.detach()

                self.handles.append(norm.register_forward_pre_hook(mlp_pre_hook, with_kwargs=True))

                def mlp_hook(_mod, args, kwargs, output, _idx=layer_idx):
                    residual = self.mlp_residual.get(_idx)
                    if residual is None:
                        return output
                    mlp_out = output[0] if isinstance(output, tuple) else output
                    if not isinstance(mlp_out, torch.Tensor) or mlp_out.shape != residual.shape:
                        return output
                    new_mlp, stats = _remove_parallel_and_stats(
                        mlp_out, residual, alpha=self.xsa_alpha, perp_scale=self.xsa_perp_scale
                    )
                    for name, value in stats.items():
                        self.stats.update_tensor("mlp", _idx, name, value)
                    if isinstance(output, tuple):
                        return (new_mlp,) + output[1:]
                    return new_mlp

                self.handles.append(layer.mlp.register_forward_hook(mlp_hook, with_kwargs=True))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.attn_residual.clear()
        self.mlp_residual.clear()
        self.attn_value.clear()

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _read_jsonl(path: str, text_key: str) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_key)
            if isinstance(text, str) and text.strip():
                texts.append(text)
    return texts


def load_texts(args) -> List[str]:
    if args.jsonl_path:
        texts = _read_jsonl(args.jsonl_path, args.text_key)
    elif args.dataset == "wikitext":
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [x["text"] for x in ds if isinstance(x.get("text"), str) and x["text"].strip()]
    else:
        texts = list(BUILTIN_TEXT_SETS[args.dataset])

    if args.max_samples > 0:
        texts = texts[: args.max_samples]
    if not texts:
        raise ValueError("No evaluation texts were loaded.")
    return texts


def iter_batches(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def iter_batches_with_progress(items: List[str], batch_size: int, desc: str) -> Iterable[List[str]]:
    total_batches = (len(items) + batch_size - 1) // batch_size if batch_size > 0 else 0
    batches = iter_batches(items, batch_size)
    if tqdm is None:
        for batch_idx, batch in enumerate(batches, start=1):
            print(f"[INFO] {desc}: batch {batch_idx}/{total_batches}", flush=True)
            yield batch
        return

    progress = tqdm(
        batches,
        total=total_batches,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )
    try:
        for batch in progress:
            yield batch
    finally:
        progress.close()


def _format_eval_texts(tokenizer, texts: List[str], args) -> List[str]:
    if not args.use_chat_template:
        return texts
    if not getattr(tokenizer, "chat_template", None):
        print("[WARN] Tokenizer has no chat_template; using raw texts.", flush=True)
        return texts

    rendered = []
    for text in texts:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": text})
        rendered.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return rendered


@torch.no_grad()
def evaluate_loss(model, tokenizer, texts: List[str], args) -> Dict[str, float]:
    summary, _ = evaluate_loss_with_per_text(model, tokenizer, texts, args)
    return summary


@torch.no_grad()
def evaluate_loss_with_per_text(model, tokenizer, texts: List[str], args) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    per_text_metrics: List[Dict[str, float]] = []
    progress_desc = f"{args.progress_label}: loss_eval" if getattr(args, "progress_label", "") else "loss_eval"

    for batch in iter_batches_with_progress(texts, args.batch_size, desc=progress_desc):
        model_inputs = _format_eval_texts(tokenizer, batch, args)
        enc = tokenizer(
            model_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        if attention_mask is not None:
            token_mask = attention_mask[:, 1:].contiguous().bool()
        else:
            token_mask = torch.ones_like(labels, dtype=torch.bool)

        valid_tokens = int(token_mask.sum().item())
        if valid_tokens <= 0:
            continue
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        token_losses = token_losses * token_mask.to(token_losses.dtype)
        sample_tokens = token_mask.sum(dim=1)
        sample_loss_sums = token_losses.sum(dim=1)
        sample_mean_losses = sample_loss_sums / sample_tokens.clamp_min(1).to(token_losses.dtype)

        total_loss += float(sample_loss_sums.sum().item())
        total_tokens += valid_tokens
        total_batches += 1
        for sample_loss, sample_token_count in zip(sample_mean_losses, sample_tokens):
            token_count = int(sample_token_count.item())
            if token_count <= 0:
                continue
            mean_loss = float(sample_loss.item())
            per_text_metrics.append(
                {
                    "loss": mean_loss,
                    "ppl": float(math.exp(mean_loss)) if mean_loss < 50 else float("inf"),
                    "tokens": float(token_count),
                    "batches": 1.0,
                }
            )

    mean_loss = total_loss / max(1, total_tokens)
    return {
        "loss": mean_loss,
        "ppl": float(math.exp(mean_loss)) if mean_loss < 50 else float("inf"),
        "tokens": float(total_tokens),
        "batches": float(total_batches),
    }, per_text_metrics


def run_target(model, tokenizer, texts: List[str], target: str, args) -> Dict:
    if target == "none":
        metrics, per_text_metrics = evaluate_loss_with_per_text(model, tokenizer, texts, args)
        return {"target": target, "metrics": metrics, "xsa_stats": {}, "per_text_metrics": per_text_metrics}

    with QwenXSAForwardHooks(
        model,
        target=target,
        start_layer=args.xsa_start_layer,
        end_layer=args.xsa_end_layer,
        skip_first_n=args.xsa_skip_first_n,
        skip_last_n=args.xsa_skip_last_n,
        intervention_site=args.xsa_intervention_site,
        xsa_alpha=args.xsa_alpha,
        xsa_perp_scale=args.xsa_perp_scale,
        track_stats=args.xsa_track_stats,
        track_layerwise_stats=args.xsa_layerwise_stats,
    ) as hooks:
        metrics, per_text_metrics = evaluate_loss_with_per_text(model, tokenizer, texts, args)
        stats = hooks.stats.summary()
        layer_window = hooks.layer_window
    return {
        "target": target,
        "metrics": metrics,
        "xsa_stats": stats,
        "layer_window": layer_window,
        "per_text_metrics": per_text_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen forward-only XSA ablations on a fixed text set.")
    parser.add_argument("--model_name", type=str, required=True, help="HF model id or local model path.")
    parser.add_argument("--targets", type=str, default="none,attn,mlp,both", help="Comma-separated: none,attn,mlp,both.")
    parser.add_argument("--dataset", type=str, default="builtin", choices=["builtin", "attn_matrix", "wikitext"])
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--xsa_start_layer", type=int, default=0, help="Inclusive first layer to modify.")
    parser.add_argument("--xsa_end_layer", type=int, default=-1, help="Exclusive end layer to modify; -1 means all layers.")
    parser.add_argument("--xsa_skip_first_n", type=int, default=0, help="Do not modify the first N decoder layers.")
    parser.add_argument("--xsa_skip_last_n", type=int, default=0, help="Do not modify the last N decoder layers.")
    parser.add_argument("--xsa_alpha", type=float, default=1.0, help="Scale for removed parallel component.")
    parser.add_argument("--xsa_perp_scale", type=float, default=1.0, help="Scale for perpendicular component.")
    parser.add_argument(
        "--xsa_track_stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to compute and save XSA geometric stats. Default is off to reduce overhead.",
    )
    parser.add_argument(
        "--xsa_layerwise_stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to keep per-layer XSA stats. Default is off to reduce output size and overhead.",
    )
    parser.add_argument(
        "--xsa_intervention_site",
        type=str,
        default="xsa_middle_multihead",
        choices=["xsa_middle", "xsa_middle_multihead", "residual_output"],
        help=(
            "xsa_middle is token-mixer-only and modifies the middle pair on merged heads: "
            "projected value -> out_proj input for supported full-attention or linear-attention mixers. "
            "xsa_middle_multihead applies the same idea per head. "
            "residual_output is the older approximation: "
            "residual stream input -> final token-mixer or MLP output for attn/mlp/both."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render each text as a single-turn chat before loss eval. Recommended for Qwen Instruct models.",
    )
    parser.add_argument("--system_prompt", type=str, default="", help="Optional system message when using chat template.")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--per_text_output_jsonl", type=str, default=None)
    parser.add_argument("--text_idx_offset", type=int, default=0)
    parser.add_argument("--progress_label", type=str, default="")
    args = parser.parse_args()

    target_list = [x.strip() for x in args.targets.split(",") if x.strip()]
    bad_targets = sorted(set(target_list) - {"none", "attn", "mlp", "both"})
    if bad_targets:
        raise ValueError(f"Unknown targets: {bad_targets}")

    _require_transformers_version(args.model_name)

    if _patch_transformers_tp_plan_check():
        print(
            "[INFO] Patched Transformers ALL_PARALLEL_STYLES for Qwen TP-plan init check "
            "(eval does not use HF tensor parallelism).",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=_resolve_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    texts = load_texts(args)
    print(
        f"[INFO] Loaded {len(texts)} texts for ablation "
        f"(source={args.jsonl_path or args.dataset}, batch_size={args.batch_size})",
        flush=True,
    )
    results = {
        "model_name": args.model_name,
        "dataset": args.jsonl_path or args.dataset,
        "dataset_note": (
            "attention-matrix-sensitive builtin probes"
            if args.dataset == "attn_matrix"
            else ("general builtin probes" if args.dataset == "builtin" else "")
        ),
        "max_samples": len(texts),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "use_chat_template": bool(args.use_chat_template),
        "system_prompt": args.system_prompt,
        "xsa_layer_selection": {
            "xsa_start_layer": args.xsa_start_layer,
            "xsa_end_layer": args.xsa_end_layer,
            "xsa_skip_first_n": args.xsa_skip_first_n,
            "xsa_skip_last_n": args.xsa_skip_last_n,
        },
        "xsa_track_stats": bool(args.xsa_track_stats),
        "xsa_layerwise_stats": bool(args.xsa_layerwise_stats),
        "xsa_alpha": float(args.xsa_alpha),
        "xsa_perp_scale": float(args.xsa_perp_scale),
        "xsa_intervention_site": args.xsa_intervention_site,
        "xsa_intervention_pair": (
            (
            "token_mixer_value_to_out_proj_input_merged_heads"
            if args.xsa_intervention_site == "xsa_middle"
            else (
                    "token_mixer_value_to_out_proj_input_multihead"
                    if args.xsa_intervention_site == "xsa_middle_multihead"
                    else "x_to_y_post"
                )
            )
        ),
        "analysis_focus": (
            "copy/entity/closure/position-sensitive prompts for studying how XSA perturbs effective self-attention behavior"
            if args.dataset == "attn_matrix"
            else ""
        ),
        "targets": {},
    }

    target_results = {}
    for target in target_list:
        print(f"[INFO] Evaluating target={target}", flush=True)
        result = run_target(model, tokenizer, texts, target, args)
        target_results[target] = result
        results["targets"][target] = {
            "metrics": result["metrics"],
            "xsa_stats": result["xsa_stats"],
            "layer_window": result.get("layer_window", {}),
        }
        print(json.dumps({target: result["metrics"]}, indent=2), flush=True)

    payload = json.dumps(results, indent=2)
    print(payload)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    if args.per_text_output_jsonl:
        per_text_path = Path(args.per_text_output_jsonl)
        per_text_path.parent.mkdir(parents=True, exist_ok=True)
        with per_text_path.open("w", encoding="utf-8") as f:
            for local_idx, text in enumerate(texts):
                rec = {
                    "text_idx": args.text_idx_offset + local_idx,
                    "text": text,
                    "targets": {},
                }
                for target in target_list:
                    rec["targets"][target] = target_results[target]["per_text_metrics"][local_idx]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
