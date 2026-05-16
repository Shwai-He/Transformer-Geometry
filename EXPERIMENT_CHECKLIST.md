# Experiment Checklist (Parallel Control Paper)

## 0. Setup Lock
- [ ] 固定统一训练配方（optimizer, lr schedule, tokens, eval protocol）
- [ ] 固定随机种子集合（建议 3 seeds）
- [ ] 固定两阶段规模：
- [ ] Small: `<500M`
- [ ] Large: `up to 2.7B`

### Current Concrete Setup (nanoGPT / fineweb100bt)

#### Model scales used in this repo

| Size Tag | Approx Params | n_layer | n_head | n_embd | block_size |
|---|---:|---:|---:|---:|---:|
| `0p7b` | ~0.76B | 24 | 6 | 1536 | 2048 |
| `1p4b` | ~1.42B | 24 | 16 | 2048 | 2048 |
| `2p7b` | ~2.65B | 32 | 20 | 2560 | 2048 |

Reference config files:
- `nanoGPT/config/train_xsa_0p7b.py`
- `nanoGPT/config/train_xsa_1p4b.py`
- `nanoGPT/config/train_xsa_2p7b.py`

#### Unified training recipe (current default)

- Dataset: `fineweb100bt`
- Max iters: `200000`
- Warmup iters: `2000`
- LR decay iters: `200000`
- Weight decay: `1e-1`
- Eval interval: `1000`
- Eval iters: `200`
- Log interval: `10`
- Context length: `2048`
- Optimizer: AdamW (fused when available)

#### Learning rate by scale (current default)

| Size Tag | learning_rate |
|---|---:|
| `0p7b` | `5e-4` |
| `1p4b` | `4e-4` |
| `2p7b` | `3e-4` |

#### Notes on global batch alignment

- Tokens/iter formula:  
  `tokens_per_iter = batch_size * block_size * gradient_accumulation_steps * (nnodes * gpus_per_node)`
- In launch scripts, `batch_size` and `gradient_accumulation_steps` are adjusted per node count to keep target global batch comparable.

#### Recommended token budgets (practical default)

- Fast screening (many variants): `10B–20B` tokens
- Main comparison (paper primary table): `30B–50B` tokens
- Final confirmation (few key settings): `~60B` tokens
- Avoid defaulting all runs to `100B` unless specifically doing scaling-law/long-horizon analysis.

Per-scale suggestion:
- `0p7b`: `20B–40B`
- `1p4b`: `30B–50B`
- `2p7b`: `40B–60B`

Execution rule:
- Within one comparison group, keep token budget strictly matched across methods/seeds.

## 1. Zero-shot Core (Must Have)
- [ ] Hard deletion: parallel-only deletion（Attn-only / MLP-only / All）
- [ ] Hard deletion: perpendicular-only deletion（Attn-only / MLP-only / All）
- [ ] Ratio ablation:
- [ ] `gamma_parallel in [0, 0.25, 0.5, 0.75, 1.0]`
- [ ] `gamma_perp in [0, 0.25, 0.5, 0.75, 1.0]`
- [ ] 输出主图：`fig_zeroshot_parallel_perp_sweep`
- [ ] 输出表格：`tab_zeroshot`（PPL + benchmark deltas）

## 2. Method-aligned Validation
- [ ] 验证 unified table 对应的两类方法：
- [ ] value-space parallel control
- [ ] residual-space parallel control
- [ ] 对比两者在相同预算下的 zero-shot 表现
- [ ] 记录 `alpha` 分布统计（层均值、方差、深度趋势）

## 3. Pretraining Stage-1 (<500M)
- [ ] Baseline
- [ ] Residual-space parallel control
- [ ] Value-space parallel control
- [ ] Gated parallel control（固定 gamma sweep）
- [ ] 记录 loss 曲线与 Tok@loss
- [ ] 输出：`fig_pretrain_loss_main`（左图 small）
- [ ] 回填：`tab_pretrain_loss_main`（small列）

## 4. Pretraining Stage-2 (up to 2.7B)
- [ ] 复用 Stage-1 最优设置跑大模型
- [ ] Baseline
- [ ] Residual-space parallel control
- [ ] Value-space parallel control
- [ ] Gated parallel control（可选）
- [ ] 输出：`fig_pretrain_loss_main`（右图 large）
- [ ] 回填：`tab_pretrain_loss_main`（large列）

## Training Matrix (Systematic Plan)

| Exp ID | Scale | Method | Control Strength | Seeds | Fixed Budget (tokens/steps) | Primary Output | Secondary Output | Status |
|---|---|---|---|---|---|---|---|---|
| T1 | <500M | Baseline | N/A | 3 | Matched | Loss curve | Benchmarks | [ ] |
| T2 | <500M | Residual-space parallel control | Hard | 3 | Matched | Loss curve | Alpha stats | [ ] |
| T3 | <500M | Value-space parallel control | Hard | 3 | Matched | Loss curve | Diag/entropy | [ ] |
| T4 | <500M | Gated parallel control | Gamma sweep | 3 | Matched | Loss vs gamma | Diag/entropy vs gamma | [ ] |
| T5 | 2.7B | Baseline | N/A | 3 | Matched | Loss curve | Benchmarks | [ ] |
| T6 | 2.7B | Residual-space parallel control | Best from T2/T4 | 3 | Matched | Loss curve | Alpha stats | [ ] |
| T7 | 2.7B | Value-space parallel control | Best from T3/T4 | 3 | Matched | Loss curve | Diag/entropy | [ ] |
| T8 | 2.7B | Gated parallel control | Key gamma points | 3 | Matched | Loss curve | Benchmarks | [ ] |

### Minimum report fields per training run
- `run_id`, `model_size`, `method`, `gamma` (if any), `seed`
- `final_loss`, `best_loss`, `tok_at_ref_loss`
- `diag_mean`, `offdiag_entropy`, `attn_alpha_mean`, `mlp_alpha_mean`
- `eval_hellaswag`, `eval_mmlu`, `eval_arc_c`, `eval_winogrande`

### Controlled variables (must be constant within a scale)
- optimizer / lr schedule / warmup / weight decay
- global batch size / context length / token budget
- data source and preprocessing
- evaluation harness settings (shot count, prompt template)

## 5. Benchmark Evaluation After Pretraining
- [ ] Small model benchmark评测（统一shot设置）
- [ ] Large model benchmark评测（统一shot设置）
- [ ] 回填：`tab_pretrain_benchmark_main`
- [ ] 当前论文表列：HellaSwag / MMLU / ARC-C / Winogrande
- [ ] 如需替换为你们实际集合，同步改tex表头

## 6. Gated Variant Ablation
- [ ] 固定模型规模（先small）做 gamma sweep
- [ ] 记录：final loss / diag concentration / off-diag entropy
- [ ] 输出：`fig_gate_sweep`
- [ ] 可选：large规模复验1-2个关键gamma点

## 7. Robustness & Consistency
- [ ] 3 seeds统计（均值±std）
- [ ] 校验趋势一致性（small vs large）
- [ ] 检查是否存在“只在单seed有效”

## 8. Paper Fill-in Targets
- [ ] `experiments.tex` 中所有 `--` 数值回填
- [ ] 所有 placeholder 图替换成真实图
- [ ] caption统一术语：
- [ ] `value-space parallel control`
- [ ] `residual-space parallel control`
- [ ] 检查 Method 与 Experiments 符号一致（`alpha`, `gamma`）

## 9. Priority Order (Execution)
1. Zero-shot hard deletion + ratio ablation
2. Pretraining small
3. Gated small ablation
4. Pretraining large
5. Benchmark small/large
6. Seed复验与最终回填
