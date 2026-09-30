# 📐 Transformer-Geometry: 每日前沿文献关联与平行/正交子空间动力学库 (2026-09)

**Document ID:** `TG-LIT-202609` | **Last Updated:** `2026-09-30` | **Target Path:** `docs/frontier_literature_connections_2026_09.md` | **Total Routed Papers:** `26`

> [!IMPORTANT]
> **🔗 跨仓库文献引用链闭环 (Cross-Repository Reference Chain Closure)**
> 本文件由每日 AI 前沿论文精读流水线自动路由生成，专门收录直接引用或印证我们 **EMNLP 2026 Findings (*Transformer-Geometry: Decomposing Transformer Updates into Parallel & Perpendicular Subspaces*, `arXiv:2609.15975`)** 的平行分量（ $\Delta h _ \parallel$ 幅度缩放）与正交分量（ $\Delta h _ \perp$ 方向旋转）、Value-Space (XSA) 正交分解、循环残差方差增长定理（`DeepLoop`, `Fully Looped Transformer`）及无前向谱冗余层剪枝（`WRP`, `SHIFT-LLM`, `CLSE`, `ASL`, `VestigeKV`, `DEE-VLA`, `CAT-Flow`, `MSFM`, `ACPruner`）的最新 arXiv 论文笔记。
> 每一篇收录文献均包含：**核心痛点、底层数学公式、ASCII 架构图、关键实测指标**，以及**与 `transformer-geometry` 仓库具体代码模块和我们已发表代表作（Our Works）的双向锚定**。

---

## 🌟 1. 核心关联文献与本仓库模块映射速查表 (Executive Reference-to-Module Matrix)

| 收录日期 | 论文标题与 arXiv 链接 | 关键实测收益 / 核心结论 | 锚定本仓库代码模块与文档路径 (`Target Module`) | 原始精读归档 |
| :---: | :--- | :--- | :--- | :---: |
| `2026-09-30` | [**ACPruner & SCOPD**](https://arxiv.org/abs/2609.34558) (`arXiv:2609.34558`) | `ACPruner` (`2609.34558`) 保留 64/576 视觉 Token 维持 97.4% 精度；`SCOPD` (`2609.34044`) 10% 视觉 Token 保留率下 13 基准保留率：Vanilla 86.37%、SCOPD 90.49%、SCOPD+ 92.43% | `attention_xsa/value_space_projection.py` (Biased Attention Coverage & Barycentric Value Compensation) | [2026-09-30](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-30_ai_paper_notes.md) |
| `2026-09-30` | [**CAT-Flow & MSFM**](https://arxiv.org/abs/2609.01746) (`arXiv:2609.01746`) | **`CAT-Flow` 免训练少步生成大幅提速**：在 Flux、Stable Diffusion 3、ImageNet SiT 以及机器人流匹配控制策略上，完全免训练的 `CAT-Flow` 在 **4–8 NFE** 低步数... | `probing/residual_decomposition.py` (Directional Velocity Curvature $\kappa(t)$ vs Orthogonal Update Rotation $\Vert \Delta h _ \perp\Vert$ ) | [2026-09-30](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-30_ai_paper_notes.md) |
| `2026-09-29` | [**✂️ CoverPruner & SFPruner**](https://arxiv.org/abs/2609.03158) (`arXiv:2609.03158`) | 在 LLaVA-NeXT、Qwen2.5-VL 与 InternVL-2.5 等高分辨率多模态模型上，当剪除 **80%–88.9% 视觉 Token**（仅保留 64–128 个 Token）时，`CoverPruner` 与... | `attention_xsa/value_space_projection.py` (Value-Space Barycentric Surrogate Compensation) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-29` | [**⚡ VestigeKV**](https://arxiv.org/abs/2609.03949) (`arXiv:2609.03949`) | 在基于 MLA 架构的长上下文大模型上（128K–256K 上下文长度），`VestigeKV` 无需任何重新训练或旁路预测器，在仅加载 **15%–20% KV 潜向量**的稀疏注意力预算下，在 RULER、LongBench... | `attention_xsa/value_space_projection.py` (NoPE-MLA Orthogonal Vestigial Null-Space Norm $\Vert k _ {t,\text{vest}}\Vert _ 2$ ) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-29` | [**🦾 DEE-VLA**](https://arxiv.org/abs/2609.29382) (`arXiv:2609.29382`) | 在 LIBERO（Spatial / Object / Goal / Long）与真机双臂灵巧操作任务上，`DEE-VLA` 在成功率与全深度 10-NFE 基线持平（甚至因减少自由空间过拟合而提升 **+0.8%**）的同时，平... | `probing/residual_decomposition.py` (Cosine Velocity Direction Stability vs $\Vert \Delta h _ \perp\Vert$ Convergence) | [2026-09-29](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-29_ai_paper_notes.md) |
| `2026-09-28` | [**✂️ CLSE**](https://arxiv.org/abs/2606.24165) (`arXiv:2606.24165`) | 在 LLaVA-NeXT-7B/13B 与 Qwen2.5-VL-7B 上，免训练剪除 **75% 视觉 Token** 时，在 DocVQA、ChartQA、TextVQA 与 Video-MME 等 10 项多模态基准上平均保... | `probing/residual_decomposition.py` (Cross-Layer Spectral Entropy Derivative vs $\Delta h _ \perp$ ) | [2026-09-28](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-28_ai_paper_notes.md) |
| `2026-09-28` | [**✂️ ASL**](https://arxiv.org/abs/2601.07667) (`arXiv:2601.07667`) | 在 Llama-3.1-8B/70B 与 Qwen2.5-14B 上，针对 RULER、InfiniteBench 与 Needle-in-a-Haystack（128K 上下文）评测表明：在相同的... | `compression/layer_geometry_pruning.py` (Marginal Information Gain Saturation vs $\Vert \Delta h _ \perp\Vert$ Decay) | [2026-09-28](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-28_ai_paper_notes.md) |
| `2026-09-27` | [**SHAPE**](https://arxiv.org/abs/2606.09886) (`arXiv:2606.09886`) | **跨架构零训练稳健性**：在 **Qwen3-30B-A3B**、**DeepSeek-V2-Lite** 与 **GPT-OSS-20B** 三大主流细粒度 MoE 模型上，仅需 128 条 C4/WikiText2 校准样本... | `probing/` & `attention_xsa/` (`arXiv:2609.15975`) | [2026-09-27](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-27_ai_paper_notes.md) |
| `2026-09-27` | [**L2R**](https://arxiv.org/abs/2601.21349) (`arXiv:2601.21349`) | **语言与视觉双模态全面验证**：在基于 **OLMoE** 的语言模型预训练/微调以及 **ImageNet** 视觉 MoE 骨干网络上，L2R 将路由器参数量削减 **60%–75%**，同时在相同激活专家预算下将下游任务困... | `probing/angular_concentration.py` (Lipschitz Angular Dispersion in MoE Routers) | [2026-09-27](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-27_ai_paper_notes.md) |
| `2026-09-27` | [**OBCache**](https://arxiv.org/abs/2510.07651) (`arXiv:2510.07651`) | **即插即用全面提升主流基线**：在 **Llama-3.1-8B-Instruct**、**Qwen-2.5-7B/14B-Instruct** 与 **Mistral-7B** 上，将 OBCache 的... | `attention_xsa/value_space_projection.py` (Output-Aware Value Jacobian Perturbation) | [2026-09-27](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-27_ai_paper_notes.md) |
| `2026-09-26` | [**🔄 LoopMoE**](https://arxiv.org/abs/2606.04438) (`arXiv:2606.04438`) | **等参数量与等 FLOPs 双向碾压**：在语言建模基准与常识推理任务上，循环 $K=2\sim 4$ 步的 `LoopMoE` 在相同活跃参数量下显著优于标准稠密 Looped 模型，且在相同总参数预算下逼近非共享深层 MoE... | `probing/` & `attention_xsa/` (`arXiv:2609.15975`) | [2026-09-26](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-26_ai_paper_notes.md) |
| `2026-09-26` | [**⚖️ SelKV**](https://arxiv.org/abs/2607.16213) (`arXiv:2607.16213`) | 在 LongBench、RULER 及多轮数学推理基准上，免训练实现 **5x–10x KV Cache 压缩**，通过引入对数分母补偿项，消除了高压缩比下 80% 以上的精度退化。 | `probing/` & `attention_xsa/` (`arXiv:2609.15975`) | [2026-09-26](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-26_ai_paper_notes.md) |
| `2026-09-25` | [**Fully Looped Transformer**](https://arxiv.org/abs/2605.18797) (`arXiv:2605.18797`) | 在完全不增加任何额外参数（0 Extra Parameters）的条件下，Fully Looped Transformer 在 $K=8, 12$ 步循环预训练中完全消除了传统 Looped Transformer 的梯度尖峰（G... | `probing/residual_decomposition.py` (Hyper-Connections Orthogonal Dispersion) | [2026-09-25](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-25_ai_paper_notes.md) |
| `2026-09-25` | [**On the Limits of Layer Pruning in Genera**](https://arxiv.org/abs/2602.01997) (`arXiv:2602.01997`) | 实验精确测定了 Llama-3-8B/70B 与 Qwen-2.5 在不同推理跳数 $m \in \lbrace2, 3, 4, 5\rbrace$ 下的临界剩余层数 $L _ {\text{crit}}(m)$ ，并证明当物理层... | `compression/layer_geometry_pruning.py` (Reasoning Hop Depth Lower Bound) | [2026-09-25](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-25_ai_paper_notes.md) |
| `2026-09-25` | [**How Pruning Attention Layers Affects Int**](https://arxiv.org/abs/2606.24970) (`arXiv:2606.24970`) | 在事实问答（TruthfulQA、haluEval）与医疗/金融高风险推理任务上，该校准修复将深度剪枝模型的 **ECE 降低 68%**，并在基于置信度的拒绝采样（Selective Prediction）中恢复了 98% 的安... | `compression/layer_geometry_pruning.py` (Attention vs MLP Subspace Division) | [2026-09-25](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-25_ai_paper_notes.md) |
| `2026-09-25` | [**SAC**](https://arxiv.org/abs/2604.18392) (`arXiv:2604.18392`) | 在 TB 级长上下文并发推理中，SAC 将跨节点 KV 读取有效带宽利用率从 `15%` 提升至 **`94%`**，P99 尾延迟降低 **3.7x**。 | `attention_xsa/value_space_projection.py` (Output-Aware Value Jacobian Perturbation) | [2026-09-25](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-25_ai_paper_notes.md) |
| `2026-09-24` | [**Training-Free Looped Transformers**](https://arxiv.org/abs/2605.23872) (`arXiv:2605.23872`) | 在完全零训练（Zero Finetuning）的 **Llama-3-8B** 与 **Mistral-7B** 上，对中段 6 层额外循环 $K=2$ 次，在 GSM8K、ARC-Challenge 与逻辑推理任务上直接获得... | `probing/` & `attention_xsa/` (`arXiv:2609.15975`) | [2026-09-24](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-24_ai_paper_notes.md) |
| `2026-09-23` | [**MELT**](https://arxiv.org/abs/2605.07721) (`arXiv:2605.07721`) | 在 $K=4$ 与 $K=8$ 循环配置下，MELT 将长文本解码时的 **KV 缓存显存与带宽读取量直接削减 $75\text{ pct}–87.5$ %（严格降至 $1/K$ ）**，同时在语言建模与数学推理上与保存全套每步... | `probing/residual_decomposition.py` (Terminal-Iteration Latent Convergence) | [2026-09-23](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-23_ai_paper_notes.md) |
| `2026-09-22` | [**SPIN**](https://arxiv.org/abs/2604.26837) (`arXiv:2604.26837`) | 在单台 8 卡服务器上支持 **1M–2M 上下文长度** 并发推理，相比纯 CPU Offloading（Infinite-LLM）实现 **4.8x** 吞吐提升，且恢复 99.7% 全量注意力精度。 | `probing/` & `attention_xsa/` (`arXiv:2609.15975`) | [2026-09-22](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-22_ai_paper_notes.md) |
| `2026-09-21` | [**DeepLoop**](https://arxiv.org/abs/2607.13491) (`arXiv:2607.13491`) | 在循环深度从 $K=2$ 扩展至 ** $K=16$ ** 的语言与数学推理预训练中，标准 Pre-LN 循环架构在 $K \ge 6$ 时完全发散，而 **DeepLoop** 稳定收敛并实现随循环次数 $K$ 对数线性下降的测... | `probing/residual_decomposition.py` ( $\Delta h _ \parallel$ Coherent Variance Growth $O(K^2)$ ) | [2026-09-21](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-21_ai_paper_notes.md) |
| `2026-09-21` | [**RotateK**](https://arxiv.org/abs/2605.19218) (`arXiv:2605.19218`) | 在 **LLaVA-NeXT**、**Qwen2-VL-7B** 与 **InternVL-2** 上，RotateK 剪除 **50%–60% 的 Key 通道**而无需微调，且与视觉 Token 剪枝（如 FastV / VL... | `attention_xsa/value_space_projection.py` (Orthogonal Key/Value Subspace Rotation) | [2026-09-21](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-21_ai_paper_notes.md) |
| `2026-09-20` | [**SHIFT-LLM**](https://arxiv.org/abs/2608.25068) (`arXiv:2608.25068`) | 在 **Llama-3-8B/70B** 与 **Qwen-2.5-14B** 上剪除 **25%–35% 的层**后，无需任何梯度下降微调（仅需 30 秒闭式矩阵求逆），SHIFT-LLM 将 WikiText2 困惑度（PPL... | `compression/layer_geometry_pruning.py` (Closed-Form Orthogonal Drift Compensation) | [2026-09-20](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-20_ai_paper_notes.md) |
| `2026-09-20` | [**Minima-KV**](https://arxiv.org/abs/2608.23834) (`arXiv:2608.23834`) | 在 **Llama-3.1-70B** 与 **Qwen-2.5-32B** 的 128K 长思维链并发服务中，Minima-KV 实现 **4.6x** 真实物理显存节省（零内部页碎片），将最大并发 Batch Size 提升... | `attention_xsa/value_space_projection.py` (Output-Aware Value Jacobian Perturbation) | [2026-09-20](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-20_ai_paper_notes.md) |
| `2026-09-19` | [**WRP**](https://arxiv.org/abs/2609.09883) (`arXiv:2609.09883`) | **秒级零样本层裁剪且跨领域泛化更强**：在 **Llama-3-8B/70B**、**Qwen-2.5-14B** 与 **Mistral-7B** 上，WRP 在完全不运行任何前向传播（耗时不足 8 秒）的情况下剪除... | `compression/layer_geometry_pruning.py` (Weight Spectral Redundancy vs $\Delta h _ \perp$ ) | [2026-09-19](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-19_ai_paper_notes.md) |
| `2026-09-19` | [**REAP**](https://arxiv.org/abs/2510.13999) (`arXiv:2510.13999`) | 在 **Mixtral-8x7B**、**DeepSeek-MoE-16B** 与 **Qwen1.5-MoE-A2.7B** 上，REAP 在 **25%–37.5% 专家剪枝率**下，在 GSM8K 与 HumanEval 生... | `probing/` & `attention_xsa/` (`arXiv:2609.15975`) | [2026-09-19](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-19_ai_paper_notes.md) |
| `2026-09-18` | [**✂️ AnchorPrune**](https://arxiv.org/abs/2609.08842) (`arXiv:2609.08842`) | **评估模型**：Qwen2-VL-7B/72B、LLaVA-NeXT-34B； | `probing/` & `attention_xsa/` (`arXiv:2609.15975`) | [2026-09-18](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/papers/2026-09-18_ai_paper_notes.md) |

---

## 🔎 2. 来源核验、推导边界与复现补充规范 (Source Verification & Reproducibility Notes)

### 🔎 来源核验与研究补充（2026-09-30）

本日实际为 6 个主题组、12 篇论文。本次核对标题与编号，不代表已核对全部公式、实验表或完成复现。

**引用纠正**：SCOPD 的正确编号为 [2609.34044](https://arxiv.org/abs/2609.34044)。原笔记中的 `2609.33918` 实际对应 *Green AI: Cost of LLM-Based Code Completion*，后文涉及 SCOPD 的该编号均以此更正为准。

**指标纠正**：SCOPD 摘要在 10% 视觉 Token 保留率、13 个基准下报告相对未剪枝模型的性能保留率：Vanilla 86.37%、SCOPD 90.49%、SCOPD+ 92.43%。后文“99.5% 恢复率”、5,000 条训练指令、1 Epoch、68% 延迟降低及 79% 缓存压缩未获本次核验支持，撤回这些具体数值。ACPruner 与 SCOPD 的组合应视为研究建议，不能当作论文已报告的联合实验。

| 主题组 | 原始论文来源 |
| :--- | :--- |
| 视觉剪枝与蒸馏 | [ACPruner](https://arxiv.org/abs/2609.34558)、[SCOPD](https://arxiv.org/abs/2609.34044) |
| MoE 服务 | [SlimWise](https://arxiv.org/abs/2609.34117)、[CascadeEP](https://arxiv.org/abs/2609.33252) |
| 静态图与动态剪枝 | [Dynamic Flow, Static Graph](https://arxiv.org/abs/2609.34727)、[DORA](https://arxiv.org/abs/2609.34325) |
| 流匹配 | [CAT-Flow](https://arxiv.org/abs/2609.01746)、[MSFM](https://arxiv.org/abs/2609.35454) |
| 具身与世界模型 | [VLaRL](https://arxiv.org/abs/2609.30868)、[Programmable World Model](https://arxiv.org/abs/2609.10540) |
| 自我改进智能体 | [AutoDataBench](https://arxiv.org/abs/2609.35025)、[SelfOp](https://arxiv.org/abs/2609.22792) |

**推导与实现边界**：后文 KL 公式的方向为教师到学生，不应称为学生到教师的反向 KL；隐状态对齐等组合设计仍需全文逐式核验。次模近似保证需核对非负、单调、归一化与基数约束；流形收缩结论需明确成立区域与扰动假设。跨仓映射表仅为候选适配位置，本次没有检查其他仓库路径或执行跨仓写入。

**建议复现顺序**：先分别复现 ACPruner、SCOPD，再测组合；随后验证 MoE 在长短混合请求下的质量与吞吐，最后测试固定 NFE 下的流匹配误差。记录论文版本、代码 commit、模型与数据版本、随机种子、硬件及预算；同时报告分任务性能、端到端延迟和峰值显存。智能体技能更新应使用独立保留任务，防止验证集泄漏。详细实验建议见[同日新闻](https://github.com/Shwai-He/scholar-odyssey/blob/main/intelligence/news/2026-09-30_daily_news.md)。


---

## 📐 3. 逐篇论文深度机制解构、数学公式与本仓库落地指南 (Per-Paper Deep-Dive Cards)

### 3.1 [2026-09-30] ACPruner & SCOPD: Visual Token Pruning as Biased Attention Coverage Maximization & Sparse-Context On-Policy Self-Distillation (`arXiv:2609.34558` & `arXiv:2609.34044`)
* **论文标题**：
  1. *ACPruner: Visual Token Pruning as Biased Attention Coverage Maximization in LVLMs* (`arXiv:2609.34558`)
  2. *SCOPD: Sparse-Context On-Policy Self-Distillation for Efficient Vision-Language Models* (`arXiv:2609.34044`)
* **核心关键词**：`token pruning`, `visual token pruning`, `acpruner`, `scopd`, `coverage maximization`, `on-policy self-distillation`, `vlm`, `multimodal`

#### 📌 核心痛点与研究动机 (Motivation & Pain Points)
昨天我们精读的 `CoverPruner` (`arXiv:2609.03158`) 证明了纯视觉空间的 $k$ -Medoids 覆盖优化优于盲目 Top- $k$ 显著性截断，但在复杂视觉问答（如细粒度 OCR、图表定位、空间指代推理）中仍面临两大根本瓶颈：
1. **无偏几何覆盖与跨模态任务意图的错位（Unbiased Coverage vs. Query Intent）**：如果所有图像区域按等权重做空间覆盖，大量背景纹理 Token 仍会挤占有限预算 $K$ ，导致与用户文本指令强相关的微小前景区域采样不足；
2. **剪枝后的“表征-利用鸿沟（Representation-Utilization Gap）”**：即使剪枝算法把关键视觉 Token 保留了下来，预训练于稠密完整网格（Full Dense Grid）的 LLM 解码器在面对突然缺失 85%–90% 节点的稀疏上下文（Sparse Context）时，其深层自注意力路由会发生分布偏移，无法有效提取稀疏存活 Token 中的信息。

#### ⚙️ 核心机制与数学公式推导 (Core Mechanism & Mathematical Formulation)
**第一步：`ACPruner` 的偏置注意力覆盖最大化（Biased Attention Coverage Maximization）**  
给定 $N$ 个视觉 Token 表征 $X = \lbrace x _ 1, \dots, x _ N \rbrace$ 及跨模态文本指令对第 $i$ 个视觉 Token 的先验注意力重要性权重 $\pi _ i \in (0, 1)$ （归一化后 $\sum _ {i=1}^N \pi _ i = 1$ ）。定义对称正定的特征-空间复合亲和核 $K _ {ij} = \exp\left(-\frac{1 - \cos(x _ i, x _ j)}{\tau _ f} - \frac{\lVert p _ i - p _ j \rVert _ 2^2}{\tau _ s}\right)$ 。`ACPruner` 将子集选择 $S \subseteq \lbrace 1, \dots, N \rbrace$ （ $|S| = K$ ）构建为最大化**先验注意力加权的饱和覆盖效用函数** $\mathcal{F} _ {\text{AC}}(S)$ ：

$$
\max _ {S \subseteq \mathcal{V}, |S| = K} \mathcal{F} _ {\text{AC}}(S) = \sum _ {i=1}^N \pi _ i \cdot \phi\left(\max _ {j \in S} K _ {ij} + \beta \sum _ {j \in S} K _ {ij}\right)
$$

其中 $\phi(u) = \log(1 + u)$ 为严格凹单调饱和函数， $\beta > 0$ 平衡极值代表性（Facility Location）与局部密度覆盖。由于 $\mathcal{F} _ {\text{AC}}(S)$ 满足单调非负次模性（Monotone Submodularity），在每一步贪心选择中选取使边际覆盖增益 $\Delta _ {\text{AC}}(e \mid S _ t) = \mathcal{F} _ {\text{AC}}(S _ t \cup \lbrace e \rbrace) - \mathcal{F} _ {\text{AC}}(S _ t)$ 最大的 Token，即可保证 $\left(1 - 1/e\right)$ 最优近似界。

**第二步：`SCOPD` 的稀疏上下文在线自蒸馏（Sparse-Context On-Policy Self-Distillation）**  
为弥合“表征-利用鸿沟”，`SCOPD` 不使用任何外部人工标注答案，而是让共享参数 $\theta$ 的模型在**剪枝后的稀疏视觉上下文** $\tilde{V} = \mathrm{Prune}(V; K)$ 下自回归采样生成在线推理序列 $y \sim p _ \theta(\cdot \mid \tilde{V}, Q)$ （On-Policy Rollout）。随后，将同一条自生成序列 $y$ 喂给**拥有完整视觉上下文 $V$ 的冻结教师分支** $p _ {\text{tea}}(\cdot \mid V, Q)$ ，最小化在线轨迹上的逐位置反向 KL 散度与隐状态余弦对齐损失：

$$
\mathcal{L} _ {\text{SCOPD}}(\theta) = \mathbb{E} _ {y \sim p _ \theta(\cdot \mid \tilde{V}, Q)} \left[ \sum _ {t=1}^{|y|} \mathrm{KL}\left( p _ {\text{tea}}(\cdot \mid y _ {<t}, V, Q) \middle\Vert p _ \theta(\cdot \mid y _ {<t}, \tilde{V}, Q) \right) + \lambda _ {\text{hid}} \left(1 - \cos\left(h _ t^{\text{tea}}, h _ t^{\text{stu}}\right)\right) \right]
$$

#### 🎨 算法架构图与实现伪代码 (Architecture & Pseudocode)
```
====================================================================================================
     ACPruner (偏置注意力覆盖选点) + SCOPD (稀疏上下文在线自蒸馏) 协同流水线 (arXiv:2609.34558 & 34044)
====================================================================================================

  [Full Visual Tokens V (N=576)] + [Text Query Q]
                 │
                 ├──► (Branch A: Full-Context Teacher, Frozen) ──────────────────────┐
                 │    完整保留 576 个视觉 Token，仅对学生生成的在线轨迹 y 计算参考分布   │
                 │                                                                   ▼
                 └──► (Branch B: ACPruner Biased Coverage Selection)        [On-Policy KL + Hidden
                      • 计算指令先验权重 π_i 与复合核 K_ij                   Alignment Loss L_SCOPD]
                      • 次模贪心选出 K=64 个兼顾指令焦点与全局覆盖的锚点 ~V          ▲
                                          │                                          │
                                          ▼                                          │
                      [Student On-Policy Rollout: y ~ p_θ(· | ~V, Q)] ───────────────┘
                      在稀疏视觉上下文 ~V 上自回归采样生成推理链，彻底消除训练-推理由稠密转稀疏的分布失配
====================================================================================================
```

#### 📊 实验指标与核心结论 (Experimental Results & Key Takeaways)
* **极低保留率下的性能飞跃**：在 `LLaVA-NeXT-7B` 与 `Qwen2.5-VL-7B` 上，当视觉 Token 剪掉 **88.9%**（仅保留 `64/576` 个 Token）时，单独使用 `ACPruner` 即可在 10 个多模态基准上保留 **97.4%** 的原始精度（超越 `FastV`、`SparseVLM` 与无偏 `CoverPruner` 达 **+1.8–4.6 pp**）。

#### 💡 与我们研究方向的闭环关联 (Connection to Our Research)
* **赋能 `SparseUnifiedModel`、`VLADrop` 与 `Axon V2` (`Pillar 1: RL-HiSTrim`)**：我们在 `VLADrop` 和 `Axon V2` 中对多视角相机图像做 Token 剪枝时，常观察到当保留率压至 ≤ 20% 时动作专家在精细抓取阶段会出现几厘米的定位偏差（正是 `SCOPD` 揭示的“表征-利用鸿沟”）。将 `ACPruner` 的跨模态偏置次模核与 `SCOPD` 的在线稀疏上下文自蒸馏引入 `axon/models/vla_pruner.py` 与 `sparse_umm/token_pruning.py`，可在不改动推理架构的前提下彻底抚平高倍率视觉剪枝带来的特征断层。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`attention_xsa/value_space_projection.py` (Biased Attention Coverage & Barycentric Value Compensation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-30_ai_paper_notes.md`


---

### 3.2 [2026-09-30] CAT-Flow & MSFM: Curvature-Adaptive Steps & Manifold-Stable Contraction Theory for Flow Matching (`arXiv:2609.01746` & `arXiv:2609.35454`)
* **论文标题**：
  1. *CAT-Flow: Curvature-Adaptive sTeps for Flow Matching* (`arXiv:2609.01746`)
  2. *Manifold-Stable Flow Matching* (`arXiv:2609.35454`)
* **核心关键词**：`flow matching`, `cat-flow`, `msfm`, `manifold-stable`, `contraction theory`, `curvature-adaptive`, `snapflow`, `vla`

#### 📌 核心痛点与研究动机 (Motivation & Pain Points)
流匹配（Flow Matching）已成为视觉生成与具身 VLA 动作专家（如 $\pi _ 0$ 、`GR00T`、`Axon V2`）的标准范式，但在少步（Low-NFE）推理与闭环部署中存在两大底层数学缺陷：
1. **均匀时间步长 $\Delta t = 1/N$ 与非均匀轨迹曲率的严重错配（`CAT-Flow` 动机）**：尽管理想条件流匹配期望路径是直线，但边缘化后的实际学习速度场 $v _ \theta(x _ t, t)$ 在 $t \to 0$ （脱离先验噪声分岔区）和 $t \to 1$ （吸附至精细数据流形区）附近具有极大的二阶曲率 $\lVert \ddot{x} _ t \rVert$ ，而在中段 $t \in [0.3, 0.7]$ 近似笔直。均匀欧拉步长在直线段浪费算力，在曲率峰值处产生巨大截断误差；
2. **缺乏法向恢复力导致离流形扰动发散（`MSFM` 动机）**：标准流匹配仅在训练数据流形 $\mathcal{M}$ 的切空间（Tangent Space）上定义了传输速度。一旦在少步积分或机器人传感器噪声下状态 $x _ t$ 发生微小的**法向偏移（Normal Deviation off $\mathcal{M}$ ）**，标准速度场没有任何将状态拉回流形 $\mathcal{M}$ 的法向收缩力，导致误差沿积分步指数级放大。

#### ⚙️ 核心机制与数学公式推导 (Core Mechanism & Mathematical Formulation)
**第一部分：`CAT-Flow` 的曲率自适应步长分配定律（Curvature-Adaptive Step Sizing）**  
由泰勒展开可知，单步欧拉积分 $x _ {t + h _ k} = x _ t + h _ k v _ \theta(x _ t, t)$ 的局部截断误差上界为 $\mathcal{E} _ k \approx \frac{1}{2} h _ k^2 \left\lVert \frac{\mathrm{d}}{\mathrm{d}t} v _ \theta(x _ t, t) \right\rVert _ 2 = \frac{1}{2} h _ k^2 \kappa(t)$ ，其中局部加速度范数（曲率代理） $\kappa(t) = \lVert \dot{v} _ \theta(x _ t, t) \rVert _ 2$ 可由前后步速度有限差分在线无开销估计。为在给定总步数预算 $\sum _ {k=1}^N h _ k = 1$ 下最小化全局累积截断误差上界 $\sum _ {k=1}^N h _ k^2 \kappa(t _ k)$ ，由柯西-施瓦茨不等式（或拉格朗日乘子法）可得最优步长 $h _ k^\star$ 严格反比于局部曲率的平方根（或幂次 $\alpha \in [1/3, 1/2]$ ）：

$$
h _ k^\star = \frac{\left(\kappa(t _ k) + \epsilon\right)^{-\alpha}}{\sum _ {j=1}^N \left(\kappa(t _ j) + \epsilon\right)^{-\alpha}}, \quad \text{where } \hat{\kappa}(t _ k) = \frac{\left\lVert v _ \theta(x _ {t _ k}, t _ k) - v _ \theta(x _ {t _ {k-1}}, t _ {k-1}) \right\rVert _ 2}{h _ {k-1}}
$$

**第二部分：`MSFM` 的切向传输 + 法向收缩正交分解（Manifold-Stable Flow Matching）**  
设目标数据流形 $\mathcal{M}$ 在点 $x$ 处的切空间正交投影算子为 $\Pi _ {\mathcal{T}}(x)$ ，法空间投影算子为 $\Pi _ {\mathcal{N}}(x) = I - \Pi _ {\mathcal{T}}(x)$ ，点 $x$ 到流形的隐式距离/能量梯度为 $\nabla \Psi _ {\mathcal{M}}(x) \in \mathcal{N} _ x \mathcal{M}$ 。`MSFM` 将总速度场 $v _ \theta^{\text{MSFM}}(x, t)$ 正交分解为两部分：

$$
\frac{\mathrm{d}x _ t}{\mathrm{d}t} = v _ \theta^{\text{MSFM}}(x _ t, t) = \underbrace{\Pi _ {\mathcal{T}}(x _ t) u _ \theta^{\parallel}(x _ t, t)} _ {\text{流形内切向传输 (Tangential Transport)}} - \underbrace{\gamma(t) \cdot \Pi _ {\mathcal{N}}(x _ t) \nabla \Psi _ {\mathcal{M}}(x _ t)} _ {\text{流形外法向指数收缩 (Normal Contraction, } \gamma(t) > 0\text{)}}
$$

根据非线性控制中的微分收缩理论（Contraction Theory），当法向收缩增益 $\gamma(t) \ge c _ 0 > 0$ 时，流形外法向距离 $d _ {\mathcal{M}}(x _ t)$ 严格满足李雅普诺夫指数衰减界 $d _ {\mathcal{M}}(x _ t) \le d _ {\mathcal{M}}(x _ 0) \exp\left(-\int _ 0^t \gamma(s) \mathrm{d}s\right)$ 。

#### 🎨 算法架构图与实现伪代码 (Architecture & Pseudocode)
```
====================================================================================================
      CAT-Flow (曲率自适应步长) + MSFM (切向传输与法向收缩流形稳定) 几何原理图 (arXiv:2609.01746 & 35454)
====================================================================================================

                  Off-Manifold Perturbed State x_t
                              ●
                              │  -γ(t) Π_N ∇Ψ_M(x_t)  [MSFM 法向收缩恢复力：指数级拉回流形]
                              ▼
  ═══════════════●════════════●══════════════════════════●══════════════► Data Manifold M_t
               x_{t-1}        proj_M(x_t) ──────────────► x_{t+h_k*}
                                     Π_T u_θ(x_t, t) · h_k*
                                     [CAT-Flow 切向自适应步长：高曲率处步长加密，平坦处大步跨越]
====================================================================================================
```

#### 📊 实验指标与核心结论 (Experimental Results & Key Takeaways)
* **`CAT-Flow` 免训练少步生成大幅提速**：在 Flux、Stable Diffusion 3、ImageNet SiT 以及机器人流匹配控制策略上，完全免训练的 `CAT-Flow` 在 **4–8 NFE** 低步数预算下，生成的 FID 与轨迹跟踪误差比均匀步长欧拉/Heun 求解器降低 **32%–46%**，以 6 步即可媲美标准 15 步均匀采样的质量。
* **`MSFM` 彻底根除离流形误差累积**：在受到环境噪声扰动或分布外（OOD）初始先验测试中，标准 CFM 轨迹迅速发散崩溃，而 `MSFM` 将流形偏离误差压缩了 **1–2 个数量级**，在带强几何约束的分子构象与机械臂运动流形生成上取得 SOTA。

#### 💡 与我们研究方向的闭环关联 (Connection to Our Research)
* **与我们 `Transformer-Geometry` (`arXiv:2609.15975`)、`Axon V2` (`SnapFlow` / `VLALoop`) 及 `vla-distillation` 的完美同构**：
  1. `MSFM` 将流匹配速度场正交分解为切空间 $\Pi _ {\mathcal{T}} u _ \theta^{\parallel}$ 与法空间 $-\gamma \Pi _ {\mathcal{N}} \nabla \Psi _ {\mathcal{M}}$ ，这与我们在 `Transformer-Geometry` 和 `vla-distillation` (`Law G20: Perp-Directional Decomposition`) 中将隐状态更新/速度场分解为平行分量与正交分量的几何哲学**完全一致**！
  2. `CAT-Flow` 的曲率自适应步长 $h _ k^\star \propto \kappa(t _ k)^{-\alpha}$ 可直接作为零成本插件嵌入 `axon/distillation/snapflow_loss.py` 与 `VLADrop/models/pi0.5/` 的 3-NFE / 5-NFE 多挡位求解器中。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/residual_decomposition.py` (Directional Velocity Curvature $\kappa(t)$ vs Orthogonal Update Rotation $\Vert\Delta h _ \perp\Vert$ )  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-30_ai_paper_notes.md`


---

### 3.3 [2026-09-29] ✂️ *CoverPruner & SFPruner: Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization & Single-Forward Ridge Leverage*
> 🏷️ **核心关键词**：Visual Token Pruning · Representational Coverage Maximization (RCM) · Ridge Leverage Score · High-Resolution MLLMs  
> 🔗 **arXiv 链接**：[`arXiv:2609.03158`](https://arxiv.org/abs/2609.03158) (`CoverPruner`) & [`arXiv:2607.23046`](https://arxiv.org/abs/2607.23046) (`SFPruner`)

```
  高分辨率视觉 Token 集合 V = {v_1..v_N}
       ├──► [ CoverPruner (2609.03158) ] : 语义显著性 q_i × 设施选址最大覆盖 max_{j ∈ S} sim(v_i, v_j) ──► 确保每个被剪 Token 有高相似“代言人”
       └──► [ SFPruner    (2607.23046) ] : 语义引导核矩阵岭杠杆分数 τ_i(λ) + 单次方向性排序掩码    ──► 零迭代 O(N d^2) 单次前向去冗余保留子集 S*
```

#### 🎯 背景与痛点 (Problem Statement)
现有视觉语言大模型（VLM / MLLM）的视觉 Token 剪枝算法大多遵循“独立显著性排序（Top- $k$ Salience Ranking）”范式——即根据跨模态注意力得分或范数选出前 $K$ 个最高分的视觉 Token。然而，高分辨率图像中的高显著性 Token 往往高度聚集在少数显著前景物体内部（形成严重的**空间与语义聚团同质化**），导致被保留的 $K$ 个 Token 彼此高度冗余，而大面积次显著但包含关键空间关系、文字细节或环境上下文的视觉区域却“无人代言（No One Speaks for the Pruned）”而被整体抹除；与此同时，传统的子模函数或 $k$ -Medoids 多样性子集选择需要 $O(K N^2)$ 串行贪心迭代，在 GPU 上引入不可接受的推理延迟。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **表征覆盖最大化目标（Representational Coverage Maximization, `CoverPruner`）**：
  设视觉 Token 表征集合为 $\mathcal{V} = \lbrace v _ 1, \dots, v _ N \rbrace \subset \mathbb{R}^d$ ，跨模态指令相关性先验权重为 $w _ i \ge 0$ 。`CoverPruner` 不再最大化保留子集 $\mathcal{S} \subset \mathcal{V}$ （ $\lvert \mathcal{S} \rvert = K$ ）的孤立得分之和，而是最大化**全体原始视觉 Token 在保留子集 $\mathcal{S}$ 上的加权最近邻表征覆盖度（Weighted Facility Location Coverage）**：

$$
\mathcal{S}^{\star} = \arg\max _ {\mathcal{S} \subset \mathcal{V}, \lvert \mathcal{S} \rvert = K} \sum _ {i=1}^{N} w _ i \cdot \max _ {j \in \mathcal{S}} \left( \frac{\langle v _ i, v _ j \rangle}{\lVert v _ i \rVert _ 2 \lVert v _ j \rVert _ 2 + \epsilon} \right)
$$

* **单次前向语义引导岭杠杆打分（Single-Forward Ridge Leverage Score, `SFPruner`）**：
  为彻底消除组合优化的串行迭代开销，`SFPruner` 将语义显著性对角矩阵 $W = \mathrm{diag}(w _ 1, \dots, w _ N)^{1/2}$ 融入特征矩阵 $\tilde{V} = W V \in \mathbb{R}^{N \times d}$ ，通过单次前向闭式计算每个视觉 Token 在主成分子空间中的**统计岭杠杆分数（Ridge Leverage Score）** $\tau _ i(\lambda)$ ，并结合排序方向掩码（Ranking-Based Directional Masking）一步抑制高共线性冗余邻居：

$$
\tau _ i(\lambda) = \tilde{v} _ i^{\top} \left( \tilde{V}^{\top} \tilde{V} + \lambda I _ d \right)^{-1} \tilde{v} _ i, \quad \tilde{\mathcal{S}} _ {\text{SF}}(i) = \tau _ i(\lambda) \cdot \left( 1 - \max _ {j : w _ j > w _ i} \cos(\tilde{v} _ i, \tilde{v} _ j) \right)
$$

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 LLaVA-NeXT、Qwen2.5-VL 与 InternVL-2.5 等高分辨率多模态模型上，当剪除 **80%–88.9% 视觉 Token**（仅保留 64–128 个 Token）时，`CoverPruner` 与 `SFPruner` 在细粒度 OCR（DocVQA、TextVQA）与空间关系推理（MMBench、GQA）上比 FastV、SparseVLM 与 PruMerge 平均提升 **+3.4%–+5.2%**，且算子本身在 GPU 上仅耗时 **< 0.8 ms**（零迭代并行矩阵求逆）。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：直接印证并补充了我们在 ***Understanding and Harnessing Sparsity for Unified Multimodal Models***（`TMLR 2026`, `SparseUnifiedModel`）、***ModelLesion (`EXP-E14` Token 信息密度 + 中心化分岔召回)*** 以及 ***Efficient Ads (`HisTrim`)*** 中的多模态与长序列去冗余思想。
* **落地到 `VLADrop`、`axon_v2`、`ModelLesion` 与 `efficient_ads`**：在 `VLADrop` (`models/pi0.5/`, `models/openvla-oft/`) 与 `ModelLesion` (`token_info_density_compressor.py`) 中，可直接引入 $d \times d$ 协方差矩阵求逆的**单次前向岭杠杆分数 $\tau _ i(\lambda)$ **，用闭式二阶子空间独立性度量替代启发式余弦去重。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`attention_xsa/value_space_projection.py` (Value-Space Barycentric Surrogate Compensation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 3.4 [2026-09-29] ⚡ *VestigeKV: The NoPE-MLA KV Cache Carries Its Own Sparse-Attention Signal in a Vestigial Branch*
> 🏷️ **核心关键词**：Multi-Head Latent Attention (MLA) · NoPE (No Positional Encoding) · Sparse Attention · Training-Free KV Cache Eviction  
> 🔗 **arXiv 链接**：[`arXiv:2609.03949`](https://arxiv.org/abs/2609.03949)

```
  NoPE-MLA 潜向量 c_t^KV ──┬──► 主内容分支 (Content Latent) ──► 吸收进 W_UK 参与语义矩阵乘法
                           └──► 残余解耦分支 k_t^vest (原 RoPE 解耦通道在 NoPE 化后的低维残余信号)
                                      │
                                      ▼ (天然编码与 Query 无关的全局 Token 显著性范数 ||k_t^vest||_2)
                           [ 零额外索引开销 O(1) 稀疏 KV 检索与预算驱逐 ]
```

#### 🎯 背景与痛点 (Problem Statement)
多头潜在注意力（Multi-Head Latent Attention, MLA，如 DeepSeek-V2/V3/R1 所采用）通过将键值联合压缩为低维潜向量 $c _ t^{KV} \in \mathbb{R}^{d _ c}$ 并辅以解耦旋转位置分支 $k _ t^{R} \in \mathbb{R}^{d _ R}$ ，大幅降低了 KV 缓存体积。近期前沿架构进一步发现在深层或特定长上下文头中去除显式位置编码（NoPE-MLA）可显著提升长度外推能力。然而，在长序列推理中对 MLA 实施动态稀疏注意力（Sparse Attention）或 KV 驱逐时，由于 $c _ t^{KV}$ 被紧耦合在矩阵吸收（Matrix Absorption）中，传统方法必须额外维护一套旁路哈希索引或展开高维 Key 向量计算注意力分数，破坏了 MLA 的计算与访存紧凑性。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **残余解耦分支的内生稀疏显著性发现（Vestigial Branch Saliency）**：
  `VestigeKV` 对 NoPE-MLA 架构进行了深入的谱几何剖析，发现原本为解耦位置编码设计的低维分支 $k _ t^{\text{vest}} = W _ {KR} h _ t \in \mathbb{R}^{d _ R}$ （其中 $d _ R \ll d _ c$ ，通常仅 32 或 64 维）在去除 RoPE 旋转约束后，并未退化为无用噪声，而是自发演化为一个**与具体查询向量无关（Query-Independent）的全局注意力先验门控通道**！
* **零开销残余范数打分与混合预算保留准则**：
  设第 $l$ 层历史上下文位置 $t \in \lbrace 1, \dots, S \rbrace$ 的低维残余分支向量为 $k _ t^{\text{vest},(l)}$ 。`VestigeKV` 证明历史 Token $t$ 在后续解码步被关注的期望注意力上界由其残余分支的二阶能量范数与轻量级局部内积共同主导：

$$
\mathcal{I} _ {\text{Vestige}}^{(l)}(t) = \lVert k _ t^{\text{vest},(l)} \rVert _ 2^2 + \eta \cdot \frac{\left\lvert \langle q _ {\text{cur}}^{\text{vest},(l)}, k _ t^{\text{vest},(l)} \rangle \right\rvert}{\lVert q _ {\text{cur}}^{\text{vest},(l)} \rVert _ 2 + \epsilon}
$$

  在固定 KV 预算 $B$ 下，仅需在极低维（ $d _ R = 64$ ）残余分支上按 $\mathcal{I} _ {\text{Vestige}}^{(l)}(t)$ 筛选 Top- $B$ 索引集 $\mathcal{T} _ B$ ，随后仅从 HBM 中Gather 对应的主潜向量 $\lbrace c _ t^{KV} \rbrace _ {t \in \mathcal{T} _ B}$ 参与 MLA 矩阵吸收计算。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在基于 MLA 架构的长上下文大模型上（128K–256K 上下文长度），`VestigeKV` 无需任何重新训练或旁路预测器，在仅加载 **15%–20% KV 潜向量**的稀疏注意力预算下，在 RULER、LongBench 与大海捞针基准上达到 **99.4%** 的全量 MLA 精度，解码阶段 HBM 带宽占用再降 **4.8×**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：与我们在 ***Transformer-Geometry***（`EMNLP 2026`, `arXiv:2609.15975`）中关于注意力子空间正交解耦、***Pruning-on-Representations***（`ICML 2026`）中的隐空间范数相变、以及 ***Efficient Ads (`HisTrim`)*** 和 ***Awesome-LLMs-Pruning*** 的 KV Cache 压缩体系高度契合。
* **落地到 `efficient_ads`、`TraceCraft` 与 `Awesome-LLMs-Pruning`**：在 `TraceCraft` (`tracecraft/spectral_kv.py`) 与 `efficient_ads` (`histrim/kv_compressor.py`) 中，对于采用低秩联合 KV 压缩的推荐/智能体骨干网络，可直接利用低维解耦旁路通道的二阶范数 $\lVert k _ t^{\text{vest}} \rVert _ 2^2$ 作为 O(1) 零开销的静态 KV 驱逐优先级指标。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`attention_xsa/value_space_projection.py` (NoPE-MLA Orthogonal Vestigial Null-Space Norm $\Vert k _ {t,\text{vest}}\Vert _ 2$ )  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 3.5 [2026-09-29] 🦾 *DEE-VLA: Decoupled Early Exits for Task-Dependent Compute Allocation in Flow-Matching VLAs*
> 🏷️ **核心关键词**：Vision-Language-Action (VLA) · Flow Matching · Decoupled Early Exits · Dynamic Compute Allocation  
> 🔗 **arXiv 链接**：[`arXiv:2609.29382`](https://arxiv.org/abs/2609.29382)

```
  多模态观测 (I_t, l) ──► [ VLM 感知主干 (层 1..L_vlm) ] ──► 动态早退层 l_vlm* (自由空间移动早退，精细对准深层)
                                                                  │ (跨模块KV桥接投影)
                                                                  ▼
                     [ Flow-Matching Action Expert (层 1..L_act, 积分步 k=1..K) ]
                                                                  ├──► 动作网络深度早退 l_act*(k)
                                                                  └──► 速度场曲率收敛早退步 K* ──► 实时机器人控制动作 a_t
```

#### 🎯 背景与痛点 (Problem Statement)
现有的视觉-语言-动作（VLA）基础模型（如 $\pi _ 0$ 、 $\pi _ {0.5}$ 、GR00T）通常由百亿级参数的多模态 VLM 主干与数亿参数的流匹配（Flow-Matching）动作专家（Action Expert）组成。以往的早退（Early Exit）或层剪枝方法往往将 VLM 主干深度与动作专家深度**强行绑定（Coupled Depth Scaling）**，或者对一整段轨迹的所有时间步施加相同的静态深度预算。然而，机器人操纵任务具有显著的**时空异质性（Spatio-Temporal Heterogeneity）**：在粗粒度场景理解已完成的抓取接近阶段，VLM 主干仅需浅层表征即可维持语义定位，而进入毫米级插孔（Peg-in-Hole）接触瞬间，VLM 无需重算深层语义但 Action Expert 却需要更深的网络层数与更多的流匹配 ODE 修正步来解析高频接触动力学。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **三轴解耦动态计算空间（Tri-Axis Decoupled Compute Space）**：
  `DEE-VLA` 将单次控制周期的推理算力分解为三个可独立调节的正交自由度：（1）VLM 主干退出层 $l _ {\text{vlm}} \in \lbrace 1, \dots, L _ {\text{vlm}} \rbrace$ ；（2）第 $k$ 个流匹配步中 Action Expert 的退出层 $l _ {\text{act}}^{(k)} \in \lbrace 1, \dots, L _ {\text{act}} \rbrace$ ；（3）流匹配歐拉积分的总终止步数 $K^{\star} \in \lbrace 1, \dots, K _ {\max} \rbrace$ 。
* **跨层隐状态余弦稳定性与速度场曲率早退门控**：
  在 VLM 主干内部，当相邻两层视觉-语言融合隐状态的余弦相似度超过语义收敛阈值 $\tau _ {\text{vlm}}$ 时提前退出并通过轻量级层对齐投影器生成 KV 缓存；在流匹配 Action Expert 内部，同时监测跨层速度预测残差与跨 ODE 步的**流场直线性曲率（Flow Straightness Curvature）**：

$$
\mathcal{E} _ {\text{depth}}^{(k)}(l) = \frac{\lVert v _ {\theta}^{(l)}(x _ {t _ k}, t _ k) - v _ {\theta}^{(l-1)}(x _ {t _ k}, t _ k) \rVert _ 2}{\lVert v _ {\theta}^{(l-1)}(x _ {t _ k}, t _ k) \rVert _ 2 + \epsilon} \le \delta _ {\text{act}}, \quad \mathcal{E} _ {\text{flow}}(k) = \lVert v _ {\theta}^{\star}(x _ {t _ k}, t _ k) - v _ {\theta}^{\star}(x _ {t _ {k-1}}, t _ {k-1}) \rVert _ 2 \le \delta _ {\text{ode}}
$$

  一旦 $\mathcal{E} _ {\text{flow}}(k) \le \delta _ {\text{ode}}$ （表明当前局部流场已呈直线匀速轨迹），立即跳过剩余 ODE 积分步并利用一阶欧拉外推直接输出终端动作块 $\hat{x} _ 1 = x _ {t _ k} + (1 - t _ k) v _ {\theta}^{\star}(x _ {t _ k}, t _ k)$ 。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 LIBERO（Spatial / Object / Goal / Long）与真机双臂灵巧操作任务上，`DEE-VLA` 在成功率与全深度 10-NFE 基线持平（甚至因减少自由空间过拟合而提升 **+0.8%**）的同时，平均削减了 **54.2% 的总 FLOPs** 与 **49.6% 的端到端控制延迟**，自动展现出“自由空间巡航浅层少步、接触操作阶段深层精细积分”的涌现算力分配规律。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：与我们在 `axon_v2` / `VLADrop` 中构建的 **Tri-Orthogonal Depth-Width-Step Compression (`G19`)**、**Once-for-All Switchable Multi-Gear Loops** 以及 **Looped VLA 动态停止准则 $K(s _ t, t, m)$ ** 完全同源！
* **落地到 `axon_v2` 与 `VLADrop` (`VLM-Compression`)**：可在 `axon/layers/looped_ode.py` 与 `VLADrop/models/pi0.5/` 中直接融合 `DEE-VLA` 的双判据 $\left( \mathcal{E} _ {\text{depth}}^{(k)}(l), \mathcal{E} _ {\text{flow}}(k) \right)$ ，将 VLM 编码器深度门控与 Action Expert 的 ODE 步曲率门控彻底解耦。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/residual_decomposition.py` (Cosine Velocity Direction Stability vs $\Vert\Delta h _ \perp\Vert$ Convergence)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-29_ai_paper_notes.md`


---

### 3.6 [2026-09-28] ✂️ *CLSE: Spectral Evolution-Guided Token Pruning in Multimodal Large Language Models*
> 🏷️ **核心关键词**：Multimodal Token Pruning · Cross-Layer Spectral Evolution · Discrete Cosine Transform (DCT) · Training-Free Compression  
> 🔗 **arXiv 链接**：[`arXiv:2606.24165`](https://arxiv.org/abs/2606.24165) (ECCV 2026)

```
  层 l-1 视觉隐状态 H^(l-1) ──► [ 通道维 DCT 频域投影 Φ ] ──► 频谱能量分布 P^(l-1)(ω) ┐
                                                                                      ├──► [ 跨层谱演化散度 D_CLSE(i) ] ──► Top-K 语义活跃视觉 Token 保留
  层 l   视觉隐状态 H^(l)   ──► [ 通道维 DCT 频域投影 Φ ] ──► 频谱能量分布 P^(l)(ω)   ┘
```

#### 🎯 背景与痛点 (Problem Statement)
现有多模态大模型（MLLM / VLM）免训练视觉 Token 剪枝方法（如 FastV、SparseVLM）大多依赖**单层静态注意力分数**（如第 $l$ 层文本对视觉 Token 的注意力权重 $A _ {t, v}^{(l)}$ ）。然而，由于 RoPE 旋转位置编码的远程衰减与视觉 Sink Token 现象，单层注意力极易受到空间位置偏置（Position Bias）误导——许多在当前层注意力得分较高但跨层表征几乎停止更新的“静态背景/锚点冗余 Token”被错误保留，而真正正在经历高频语义整合的关键局部视觉 Token 却被过早裁剪。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **跨层频域映射与归一化能量谱**：
  设第 $l$ 层第 $i$ 个视觉 Token 的隐状态向量为 $h _ i^{(l)} \in \mathbb{R}^d$ 。CLSE 首先通过正交离散余弦变换（DCT）基矩阵 $\Phi \in \mathbb{R}^{d \times d}$ 将特征变化量 $\Delta h _ i^{(l)} = h _ i^{(l)} - h _ i^{(l-1)}$ 投影至频域，计算频谱系数 $c _ i^{(l)} = \Phi h _ i^{(l)}$ ，并构造归一化频域能量分布：

$$
p _ {i, k}^{(l)} = \frac{\left( c _ {i, k}^{(l)} \right)^2}{\sum _ {m=1}^{d} \left( c _ {i, m}^{(l)} \right)^2 + \epsilon}, \quad k \in \lbrace 1, \dots, d \rbrace
$$

* **跨层谱演化散度（Cross-Layer Spectral Evolution Score）**：
  原文发现：真正参与跨模态语义推理的视觉 Token 在穿越浅层到中层 Transformer 时，其能量会从高频局部纹理分量向低频全局语义分量发生剧烈的**谱重分布（Spectral Redistribution）**；而背景冗余 Token 的频谱分布则保持停滞。因此定义第 $i$ 个视觉 Token 在第 $l$ 层的跨层谱演化显著性打分为对称 Jensen-Shannon 谱演化散度与残差平行演化幅度的乘积：

$$
\mathcal{S} _ {\text{CLSE}}^{(l)}(i) = \mathrm{JSD}\left( p _ i^{(l)} \parallel p _ i^{(l-1)} \right) \cdot \frac{\lVert h _ i^{(l)} - h _ i^{(l-1)} \rVert _ 2}{\lVert h _ i^{(l-1)} \rVert _ 2 + \epsilon}
$$

* **谱演化引导渐进剪枝**：
  在预设的剪枝过渡层 $l \in \mathcal{L} _ {\text{prune}}$ ，仅保留 $\mathcal{S} _ {\text{CLSE}}^{(l)}(i)$ 排名前 $K _ l$ 的语义活跃 Token，被裁剪的背景 Token 按频谱相似度加权合并至最近邻保留 Token 中以守恒低频能量。

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 LLaVA-NeXT-7B/13B 与 Qwen2.5-VL-7B 上，免训练剪除 **75% 视觉 Token** 时，在 DocVQA、ChartQA、TextVQA 与 Video-MME 等 10 项多模态基准上平均保留了 **99.1%** 的原始全量精度，显著超越单层注意力剪枝基线（+3.8%），预填充（Prefill）FLOPs 降低 **68%**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：直接呼应我们在 ***Understanding and Harnessing Sparsity for Unified Multimodal Models***（`TMLR 2026`, `SparseUnifiedModel`）、***Demystifying When Pruning Works via Representation Hierarchies***（`ICML 2026`, `Pruning-on-Representations`）以及 ***Transformer-Geometry***（`EMNLP 2026`, `arXiv:2609.15975`）中提出的跨层表征几何演化理论。
* **落地到 `VLADrop` (`VLM-Compression`) 与 `efficient_ads` (`HisTrim`)**：在我们的 `VLADrop` 具身视觉编码器与 `HisTrim` 多阶段分层序列裁剪中，可将单层注意力打分升级为 **跨层平行/正交残差演化率 + 频域谱重分布散度 $\mathcal{S} _ {\text{CLSE}}^{(l)}$ **，用零额外参数的逐层残差差分替代易受位置偏置干扰的静态注意力权重。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/residual_decomposition.py` (Cross-Layer Spectral Entropy Derivative vs $\Delta h _ \perp$ )  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-28_ai_paper_notes.md`


---

### 3.7 [2026-09-28] ✂️ *ASL: Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference*
> 🏷️ **核心关键词**：Layer-Wise Token Pruning · Adaptive Layer Selection · Attention Variance · Long-Context LLM Inference  
> 🔗 **arXiv 链接**：[`arXiv:2601.07667`](https://arxiv.org/abs/2601.07667) (ACL 2026 Findings)

```
  输入长序列 X ──► 逐层前向传播 l=1..L ──► 实时监测注意力熵变与表征漂移率 η_l
                                                    │
                        ┌───────────────────────────┴───────────────────────────┐
                        ▼ (η_l 跌破相变阈值 τ: 语义路由已收敛)                     ▼ (η_l > τ: 仍在剧烈跨位置交互)
          [ 触发 ASL 单次 Token 剪枝 (One-Shot Selection) ]                [ 保持全长序列继续前向传播 ]
```

#### 🎯 背景与痛点 (Problem Statement)
现有的长上下文逐层 Token 剪枝方法（如 PyramidInfer、LazyLLM）通常采用**跨样本固定的剪枝层配置**（例如硬编码在第 4、8、16 层按固定比例裁剪 Token）。然而，不同复杂度与不同上下文长度的输入样本，其跨位置信息汇聚的完成深度截然不同：简单检索任务在第 6 层已完成关键信息聚焦，而多跳推理任务直到第 18 层仍在跨段落聚合线索。静态固定剪枝层要么在困难样本上过早剪断推理链，要么在简单样本上浪费大量冗余计算。

#### 💡 核心方法与数学公式 (Core Methodology & Formulation)
* **跨层注意力方差与路由收敛度度量**：
  设第 $l$ 层查询窗口对上下文 Token 的平均注意力分布为 $\bar{\alpha}^{(l)} \in \Delta^{N-1}$ 。ASL 提出用注意力分布的**二阶方差锐度（Attention Variance Sharpness）**与相邻层注意力分布的 **余弦收敛度** 联合度量当前层是否已完成信息路由聚焦：

$$
\mathcal{C} _ l = \mathrm{Var}\left( \bar{\alpha}^{(l)} \right) \cdot \frac{\left\langle \bar{\alpha}^{(l)}, \bar{\alpha}^{(l-1)} \right\rangle}{\lVert \bar{\alpha}^{(l)} \rVert _ 2 \lVert \bar{\alpha}^{(l-1)} \rVert _ 2 + \epsilon}
$$

* **自适应剪枝层触发准则（Adaptive Layer Selection）**：
  当第 $l$ 层的聚焦收敛指数 $\mathcal{C} _ l$ 首次超过样本自适应阈值 $\tau _ {\text{ASL}}$ 且层间相对增幅趋于平缓（即 $\lvert \mathcal{C} _ l - \mathcal{C} _ {l-1} \rvert \le \delta$ ）时，ASL 判定该样本在层 $l^{\star}$ 已越过“信息收集—语义提纯相变点”，随即在层 $l^{\star}$ 触发 **One-Shot Token Selection**，一次性保留核心上下文子集 $\mathcal{I} _ {\text{keep}}$ ：

$$
l^{\star}(x) = \min \left\lbrace l \in \lbrace l _ {\min}, \dots, L \rbrace \middle| \mathcal{C} _ l(x) \ge \tau _ {\text{ASL}} \land \lvert \mathcal{C} _ l(x) - \mathcal{C} _ {l-1}(x) \rvert \le \delta \right\rbrace
$$

#### 📊 关键实验与结论 (Key Results & Conclusions)
* 在 Llama-3.1-8B/70B 与 Qwen2.5-14B 上，针对 RULER、InfiniteBench 与 Needle-in-a-Haystack（128K 上下文）评测表明：在相同的 **2.4× 端到端推理加速比**下，ASL 比固定层级剪枝基线在多跳问答与长程聚合任务上平均提升 **+4.6 分**，彻底消除了静态早剪导致的“大海捞针丢失”现象。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Connection to Our Works)
* **锚定我们的代表作**：与我们在 ***Uncovering the Redundancy in Transformers via Layer Dropping***（`TMLR 2025`, `LLM-Drop`）、***Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers***（`EMNLP 2025`, `Router-Tuning-Mixture-of-Depths`）以及 ***Demystifying When Pruning Works via Representation Hierarchies***（`ICML 2026`, `Pruning-on-Representations`）中揭示的“语义表征相变层（Phase-Transition Layer）”高度吻合。
* **落地到 `LLM-Drop`、`ModelLesion` 与 `efficient_ads`**：可将 ASL 的样本级在线收敛准则 $\mathcal{C} _ l(x)$ 引入 `efficient_ads` 的 `HisTrim` 多阶段裁剪触发器以及 `LLM-Drop` 的动态跳过门控中，实现**按样本难度自适应推迟或提前剪枝触发层 $l^{\star}(x)$ **。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`compression/layer_geometry_pruning.py` (Marginal Information Gain Saturation vs $\Vert\Delta h _ \perp\Vert$ Decay)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-28_ai_paper_notes.md`


---

### 3.8 [2026-09-27] SHAPE: Coalition-Aware Expert Pruning for Sparse Mixture-of-Experts LLMs

* **论文信息**：`arXiv:2606.09886` (2026-06, 开源仓库：`github.com/Alizen-1009/Shapley-Moe`)
* **核心关键词**：Sparse MoE、Cooperative Game Theory、Shapley Value Attribution、Coalition-Aware Expert Pruning、Quality-Coverage Bisection

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|               SHAPE: Coalition-Aware MoE Expert Pruning Pipeline                  |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [Calibration Corpus D_cal] ---> Layer l Top-k Routing Traces: C_t = {e_i1..e_ik} |
|                                                |                                  |
|                                                v                                  |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Intra-Layer Cooperative Game Formulation (层内专家合作博弈建模)          |  |
|  |    * Players: E_l = {1, ..., N} experts in layer l                          |  |
|  |    * Coalition Utility v_l(S): Expected output reconstruction fidelity      |  |
|  |      when active Top-k coalition C_t is restricted to subset S \cap C_t     |  |
|  +-----------------------------------------------------------------------------+  |
|                                                |                                  |
|                                                v                                  |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Monte-Carlo / Co-Activation Shapley Attribution (Shapley 协同价值归因)    |  |
|  |    \phi_i(v_l) = \sum_{S \subseteq E_l \setminus \{i\}} w(|S|) [v_l(S \cup  |  |
|  |                  \{i\}) - v_l(S)]                                           |  |
|  |    * Captures high-order synergy: preserves "bridge" experts that rarely    |  |
|  |      dominate gate mass alone but are indispensable in Top-k combinations   |  |
|  +-----------------------------------------------------------------------------+  |
|                                                |                                  |
|                                                v                                  |
|  +-----------------------------------------------------------------------------+  |
|  | 3. Quality-Coverage Bisection Selection (全局预算二分质量覆盖率动态分配)    |  |
|  |    Retain minimal subset S_l^* s.t. \sum_{i \in S_l^*} \phi_i^+ >= \alpha(\lambda)|
|  |    Bisection search on \alpha to hit exact global target pruning ratio p    |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **单专家独立打分的“组合盲区”**：现有的免训练 MoE 专家剪枝方法（如基于路由激活频率 Frequency、门控权重均值 Gate-Sum 或单专家一阶重构误差的方法）均隐含了一个错误的**独立性假设（Independence Assumption）**——即每个专家的贡献可以孤立度量。然而，MoE 的前向计算本质上是**组合协同（Coalitional）**的：每个 Token 的输出由激活的 Top- $k$ 专家子集 $C _ t$ 线性叠加生成。
* **协同正交专家的误杀**：在真实 MoE 层中，若两个高激活专家高度共线（功能冗余），同时保留两者的边际增益极低；反之，某些中低频激活的“互补/正交桥接专家（Bridge Experts）”虽然单独门控权重不高，但在特定 Top- $k$ 组合中提供了不可替代的正交残差修正。独立打分会将前者全部保留而误杀后者，导致 20%–40% 剪枝率下模型出现断崖式精度崩塌。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **层内合作博弈定义（Intra-Layer Cooperative Game）**：
   设第 $l$ 层共有 $N$ 个专家 $\mathcal{E} _ l = \lbrace1, \dots, N\rbrace$ 。给定校准集 $\mathcal{D} _ {\text{cal}}$ 上的输入隐状态 $x _ t \in \mathbb{R}^d$ ，原始 Top- $k$ 路由集合为 $C _ t \subseteq \mathcal{E} _ l$ （ $|C _ t|=k$ ），原始层输出为：

$$
y _ t = \sum _ {j \in C _ t} g _ {t,j} E _ j(x _ t)
$$

   当仅保留专家子集 $S \subseteq \mathcal{E} _ l$ 时，受限联盟输出为 $\hat{y} _ t(S) = \sum _ {j \in C _ t \cap S} \tilde{g} _ {t,j}(S) E _ j(x _ t)$ 。定义联盟 $S$ 的特征效用函数（Characteristic Utility Function） $v _ l: 2^{\mathcal{E} _ l} \to \mathbb{R}$ 为相对于空集的输出误差削减量：

$$
v _ l(S) = \mathbb{E} _ {x _ t \sim \mathcal{D} _ {\text{cal}}} \Big[ \Vert y _ t \Vert _ 2^2 - \Vert y _ t - \hat{y} _ t(S) \Vert _ 2^2 \Big]
$$

2. **基于共现轨迹的 Shapley 协同归因（Shapley Value Attribution）**：
   专家 $i \in \mathcal{E} _ l$ 的 Shapley 值定义为其在所有可能专家联盟 $S \subseteq \mathcal{E} _ l \setminus \lbrace i\rbrace$ 中的平均边际贡献：

$$
\phi _ i(v _ l) = \sum _ {S \subseteq \mathcal{E} _ l \setminus \lbrace i\rbrace} \frac{|S|!(N - |S| - 1)!}{N!} \Big( v _ l(S \cup \lbrace i\rbrace) - v _ l(S) \Big)
$$

   由于每个 Token 仅激活 $|C _ t| = k \ll N$ 个专家（例如 $k=2$ 或 $6,8$ ），任何不包含在 $C _ t$ 中的专家对该 Token 边际贡献恒为 $0$ 。因此，原本指数级 $O(2^N)$ 的全局 Shapley 计算可精确降维至局部活跃联盟 $2^{|C _ t|}$ 上的精确求和：

$$
\phi _ i(v _ l) = \mathbb{E} _ {x _ t : i \in C _ t} \left[ \sum _ {A \subseteq C _ t \setminus \lbrace i\rbrace} \frac{|A|!(|C _ t| - |A| - 1)!}{|C _ t|!} \Big( u _ t(A \cup \lbrace i\rbrace) - u _ t(A) \Big) \right]
$$

   其中局部效用 $u _ t(A)$ 度量了子集 $A$ 内专家输出向量的内积交互项 $2 \langle g _ {t,i} E _ i(x _ t), \sum _ {j \in A} g _ {t,j} E _ j(x _ t) \rangle + \Vert g _ {t,i} E _ i(x _ t)\Vert _ 2^2$ ，从而自动惩罚与同联盟其他专家负相关或冗余的专家，奖励提供正交有效增量的专家。
3. **质量覆盖率二分层间分配（Quality-Coverage Selection Rule）**：
   为实现非均匀的层间稀疏率分配，将非负 Shapley 值归一化为质量分布 $\tilde{\phi} _ {l,i} = \frac{\max(\phi _ i(v _ l), 0)}{\sum _ {j=1}^N \max(\phi _ j(v _ l), 0)}$ 。给定阈值 $\alpha \in (0, 1)$ ，每层保留最小专家集合 $S _ l^\star(\alpha)$ 使得累计 Shapley 质量覆盖率不低于 $\alpha$ ：

$$
S _ l^\star(\alpha) = \arg\min _ {S \subseteq \mathcal{E} _ l} |S| \quad \text{s.t.} \quad \sum _ {i \in S} \tilde{\phi} _ {l,i} \ge \alpha
$$

   最后通过一维二分搜索（Bisection Search）求解全局唯一阈值 $\alpha^\star$ ，使得 $\frac{1}{L N}\sum _ {l=1}^L |S _ l^\star(\alpha^\star)| = 1 - p$ （ $p$ 为目标全局剪枝率）。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* **跨架构零训练稳健性**：在 **Qwen3-30B-A3B**、**DeepSeek-V2-Lite** 与 **GPT-OSS-20B** 三大主流细粒度 MoE 模型上，仅需 128 条 C4/WikiText2 校准样本（无需任何微调），在 **20% 剪枝率**下恢复超过 **96.8%** 的原始零样本推理精度，在激进的 **40% 剪枝率**下比独立频次/门控剪枝高出 **5.4%–9.2%**（MMLU、GSM8K、ARC-Challenge）。
* **层间稀疏度自发涌现“沙漏分布”**：二分质量覆盖率准则自动在中间语义整合层保留更多专家，而在浅层词法层与深层输出对齐层裁剪高达 50% 的冗余专家。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
1. **与 *Demystifying When Pruning Works via Representation Hierarchies* (ICML 2026) & *Capacity-Aware Inference* (ICLR 2026) 的理论互证**：
   * 我们在 ICML 2026 中证明了剪枝是否生效取决于层间表示层级（Representation Hierarchy）的有效秩与冗余度分布；SHAPE 的局部 Shapley 展开式 $u _ t(A \cup \lbrace i\rbrace) - u _ t(A)$ 本质上是通过度量专家输出向量之间的交叉内积 $\langle E _ i(x), E _ j(x) \rangle$ 来识别表示子空间的正交性。
2. **与 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) 的几何融合启发**：
   * 在我们的正交/平行场分解框架 $E _ j(x) = E _ {j,\parallel}(x) + E _ {j,\perp}(x)$ 下，SHAPE 的效用函数若直接建立在总输出 $y _ t$ 的欧氏范数上，会被模长占优的平行径向分量 $E _ {j,\parallel}(x)$ 主导！**核心改进点**：将 SHAPE 的联盟效用函数 $v _ l(S)$ 限制在**去除流形平行漂移后的正交切空间分量 $P _ \perp(h _ t) E _ j(x _ t)$ ** 上计算 Shapley 值（即 **Perp-Shapley MoE Pruning**），随后对被剪除专家联盟的正交残差通过 **Woodbury / KKT 闭式补偿** 折叠进保留专家中，有望在 50% 专家剪枝率下实现近乎零损压缩。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/` & `attention_xsa/` (`arXiv:2609.15975`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-27_ai_paper_notes.md`


---

### 3.9 [2026-09-27] L2R: Low-Rank and Lipschitz-Controlled Routing for Mixture-of-Experts

* **论文信息**：Minghao Yang, Ren Togo, Guang Li, Takahiro Ogawa, Miki Haseyama (`arXiv:2601.21349`, 2026-01)
* **核心关键词**：MoE Routing Geometry、Low-Rank Latent Space、Lipschitz Continuity、Saturated Inner-Product Scoring (SIPS)、Multi-Anchor Routing

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|          L2R: Low-Rank & Lipschitz-Controlled MoE Routing Architecture            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|                        Token Hidden State h \in R^d                               |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 1. Shared Low-Rank Latent Projection (低秩路由子空间映射)|                |
|        |    z = P h \in R^r   (r << d, orthogonalized P P^T = I_r)|                |
|        |    Filters out high-dimensional isotropic noise         |                |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 2. Multi-Anchor Expert Prototypes (多锚点专家原型表示)   |                |
|        |    Each Expert e has M low-rank anchors: {u_{e,m}}_{m=1}^M               |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|        +---------------------------------------------------------+                |
|        | 3. Saturated Inner-Product Scoring (SIPS Lipschitz 控制) |                |
|        |    s_{e,m}(z) = \tau \cdot \tanh( <z, u_{e,m}> / (\tau \|z\|_\gamma) )   |
|        |    Explicitly bounds || \nabla_h s_e(h) ||_2 <= L_lip    |                |
|        +---------------------------------------------------------+                |
|                                     |                                             |
|                                     v                                             |
|             SoftMax / Top-k Selection ---> Stable Expert Dispatch                 |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **高维线性路由的三大几何病态**：标准稀疏 MoE 普遍采用单层线性投影 $s(h) = W _ r h \in \mathbb{R}^N$ 作为路由器（Router）。作者从表示几何角度指出高维空间 $d \gg N$ 中的线性内积路由存在三大固有缺陷：
  1. **维度失配与噪声过拟合（Representation Mismatch）**：Token 隐状态 $h \in \mathbb{R}^d$ 包含了大量与任务路由无关的词法/位置高频噪声，全维内积导致路由决策极易受正交噪声方向干扰。
  2. **高维角度集中现象（Angular Concentration）**：随着层深增加，Transformer 隐状态落入狭窄的各向异性锥（Anisotropic Cone），不同专家路由向量与 $h$ 的余弦相似度高度趋同，导致门控分布扁平化或赢家通吃。
  3. **范数敏感与 Lipschitz 失控（Scale Sensitivity）**：当隐状态范数 $\Vert h\Vert _ 2$ 在深层或长序列中剧烈膨胀时，未受控的内积 $w _ e^\top h$ 会使 Softmax 进入指数饱和区，微小输入扰动即可引发离散 Top- $k$ 路由集合翻转（Routing Instability）。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **共享低秩潜空间路由投影（Low-Rank Latent Routing Space）**：
   引入行正交低秩投影矩阵 $P \in \mathbb{R}^{r \times d}$ （ $r \ll d$ ，例如 $d=2048, r=64$ ），将隐状态 $h$ 压缩至低秩判别子空间：

$$
z = P h \in \mathbb{R}^r, \qquad \mathcal{L} _ {\text{orth}} = \Vert P P^\top - I _ r \Vert _ F^2
$$

2. **饱和内积打分与显式 Lipschitz 边界控制（Saturated Inner-Product Scoring, SIPS）**：
   为消除隐状态径向范数 $\Vert h\Vert _ 2$ 暴涨导致的路由震荡，L2R 设计了带阻尼范数归一化与双曲正切饱和的打分算子：

$$
\phi _ {\text{SIPS}}(z, u _ e) = \tau \cdot \tanh\left( \frac{\langle z, u _ e \rangle}{\tau \left(\sqrt{\Vert z\Vert _ 2^2 + \epsilon^2}\right)^\gamma \left(\sqrt{\Vert u _ e\Vert _ 2^2 + \epsilon^2}\right)^\gamma} \right)
$$

   其中 $\tau > 0$ 控制饱和软边界， $\gamma \in [0, 1]$ 控制径向尺度不变性强度（当 $\gamma=1$ 时退化为受控余弦路由）。利用 $\text{sech}^2(x) \le 1$ 及正交投影 $\Vert P\Vert _ 2 = 1$ ，可严格证明打分函数对原始输入 $h$ 的梯度范数（即局部 Lipschitz 常数）存在显式解析上界：

$$
\left\lVert \nabla _ h \phi _ {\text{SIPS}}(P h, u _ e) \right\rVert _ 2 \le \Vert P\Vert _ 2 \cdot \frac{\Vert u _ e\Vert _ 2^{1-\gamma}}{\epsilon^\gamma} = L _ {\text{lip}}
$$

   从而从数学上保证了有界输入扰动 $\Vert\delta h\Vert _ 2 \le \delta$ 不会引发路由分数的剧烈跳变。
3. **多锚点专家表达（Multi-Anchor Routing）**：
   由于单个专家往往需要处理多模态或多子类语义簇，在低秩空间 $\mathbb{R}^r$ 中为每个专家分配 $M$ 个子锚点 $\lbrace u _ {e,m}\rbrace _ {m=1}^M \subset \mathbb{R}^r$ （参数量仅为 $N \times M \times r \ll N \times d$ ），通过 Log-Sum-Exp 软聚合计算专家总得分：

$$
s _ e(h) = \frac{1}{\beta} \log \sum _ {m=1}^M \exp\Big( \beta \cdot \phi _ {\text{SIPS}}(P h, u _ {e,m}) \Big)
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* **语言与视觉双模态全面验证**：在基于 **OLMoE** 的语言模型预训练/微调以及 **ImageNet** 视觉 MoE 骨干网络上，L2R 将路由器参数量削减 **60%–75%**，同时在相同激活专家预算下将下游任务困惑度（PPL）降低 `0.42–0.68`，ImageNet Top-1 准确率提升 `+1.3%`。
* **路由稳定性与负载均衡双升**：在对抗性高斯扰动测试下，L2R 的 Top- $k$ 路由翻转率（Routing Flip Rate）比标准线性 Router 降低 **47%**，专家负载熵（Routing Entropy）更加接近理想均匀分布，无需强依赖破坏主任务梯度的大权重 Load-Balancing 辅助损失。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
1. **与 *Router-Tuning* (EMNLP 2025) & *Capacity-Aware Inference* (ICLR 2026) 的直接耦合**：
   * 我们在 *Router-Tuning* 中提出仅微调轻量路由器即可解锁深层稀疏网络潜力，但在极低资源或长上下文微调中，全维线性路由器容易过拟合表面范数特征。将 L2R 的 **SIPS + 低秩多锚点路由** 作为 *Router-Tuning* 的参数化形式，不仅能将可训练参数再降一个数量级，还能利用 Lipschitz 边界防止微调过程中的路由坍缩。
2. **与 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) & `MerA` SVD 初始化的深刻同构**：
   * L2R 发现的“径向范数敏感性（Scale Sensitivity）”与我们在 *Transformer-Geometry* 及 `ads-rsi`（定律 ADS-RSI-1：Scale-Cancellation）中揭示的**“深层残差流径向范数 $\Vert h\Vert _ 2$ 掩盖切向语义方向 $h / \Vert h\Vert _ 2$ ”**完全一致！此外，在将稠密模型或预训练线性路由器 $W _ r \in \mathbb{R}^{N \times d}$ 转化为 L2R 路由器时，无需随机初始化 $P$ ，可直接调用我们的 **`MerA` 数据感知激活协方差 SVD（Activation-Covariance SVD）** 提取前 $r$ 个主奇异方向初始化 $P$ ，实现零冷启动抖动的低秩 Lipschitz 路由升级。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/angular_concentration.py` (Lipschitz Angular Dispersion in MoE Routers)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-27_ai_paper_notes.md`


---

### 3.10 [2026-09-27] OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference

* **论文信息**：Yuzhe Gu, Xiyu Liang, Jiaojiao Zhao, Enmao Diao (`arXiv:2510.07651`, **ICML 2026**)
* **核心关键词**：KV Cache Eviction、Optimal Brain Damage (OBD)、Second-Order Taylor Perturbation、Output-Aware Saliency、Joint KV Pruning

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|         OBCache: Optimal Brain Damage (OBD) Layer-Wise KV Cache Pruning           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Prefill / Decoding Step: Queries Q \in R^{S_q x d_k}, Cached K, V \in R^{S_k x d}|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Attention Output Perturbation Objective (层输出二阶泰勒扰动建模)         |  |
|  |    Target: Minimize || O - \tilde{O}(\mathcal{M}) ||_F^2 where O = A V       |  |
|  |    Instead of heuristic \sum_i A_{i,j}, expand \Delta O w.r.t. masked K_j,V_j|  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|           +----------------------------+----------------------------+             |
|           v                            v                            v             |
|  +-----------------+          +-----------------+          +-------------------+  |
|  | Isolated Value  |          |  Isolated Key   |          | Joint KV Saliency |  |
|  | Score \Omega_j^V|          |  Score \Omega_j^K|         | Score \Omega_j^{KV}| |
|  | ||A_{:,j}||_2^2 |          | Softmax Jacobian|          | Exact Rank-1      |  |
|  | * ||V_j||_2^2   |          | Coupling Term   |          | Softmax Renorm    |  |
|  +-----------------+          +-----------------+          +-------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Plug-and-Play Eviction Gate (即插即用淘汰门控: 兼容 SnapKV / PyramidKV)  |  |
|  |    Evict tokens with minimal \Omega_j^{KV} -> Retain top-B KV budget        |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **启发式注意力权重累加的理论缺陷**：主流长上下文 KV 缓存淘汰算法（如 H2O、SnapKV、PyramidKV）均使用累积注意力分数 $s _ j = \sum _ {i} A _ {i,j}$ 作为 Token $j$ 的重要性指标。然而，注意力层真正传递给后续残差流的是加权输出矩阵 $O = A V \in \mathbb{R}^{S _ q \times d _ v}$ ：
  1. **忽略 Value 向量范数与方向抵消**：若某个历史 Token $j$ 的注意力权重 $A _ {i,j}$ 较高，但其对应的 Value 向量范数 $\Vert V _ j\Vert _ 2 \approx 0$ ，或者其 $V _ j$ 与当前上下文均值方向完全重合，驱逐它对注意力输出 $O$ 的实际影响极小；反之，注意力权重中等但 $\Vert V _ j\Vert _ 2$ 极大且承载正交关键信息的 Token 被驱逐后会造成严重的输出畸变。
  2. **忽略 Softmax 分母重归一化效应（Denominator Renormalization）**：驱逐第 $j$ 个 Key 相当于将注意力得分 $Z _ {i,j} \to -\infty$ ，这不仅移除了 $A _ {i,j} V _ j$ ，还会通过 Softmax 分母缩放将其余所有保留 Token 的注意力权重放大 $\frac{1}{1 - A _ {i,j}}$ 倍。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **基于 Optimal Brain Damage (OBD) 的二阶输出扰动构建**：
   设某注意力头在查询窗口 $Q \in \mathbb{R}^{S _ q \times d _ k}$ 下的注意力概率矩阵为 $A = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d _ k}}\right) \in \mathbb{R}^{S _ q \times S _ k}$ ，输出为 $O = A V \in \mathbb{R}^{S _ q \times d _ v}$ 。定义驱逐准则为最小化层输出矩阵的 Frobenius 范数平方误差 $\mathcal{E} = \frac{1}{2} \Vert O - \tilde{O} \Vert _ F^2$ 。
2. **单 Value、单 Key 与联合 KV 对的闭式显著性公式（Closed-Form Saliency Scores）**：
   * **孤立 Value 剪枝显著性（Isolated Value Saliency $\Omega _ j^V$ ）**：
     当将第 $j$ 个 Token 的 Value 向量置零（ $V _ j \leftarrow 0$ ）时， $\mathcal{E}$ 对 $V _ j$ 的海森矩阵（Hessian）为 $\mathbf{H} _ {V _ j} = \frac{\partial^2 \mathcal{E}}{\partial V _ j \partial V _ j^\top} = \left(\sum _ {i=1}^{S _ q} A _ {i,j}^2\right) I _ {d _ v}$ 。根据二阶泰勒展开，孤立 Value 显著性得分为：

$$
\Omega _ j^V = \frac{1}{2} V _ j^\top \mathbf{H} _ {V _ j} V _ j = \frac{1}{2} \Vert A _ {:, j} \Vert _ 2^2 \cdot \Vert V _ j \Vert _ 2^2
$$

注意此处注意力权重是**平方和 $\Vert A _ {:,j}\Vert _ 2^2$ **（二阶能量）而非启发式的线性求和 $\Vert A _ {:,j}\Vert _ 1$ ，且显式乘上了 Value 范数平方 $\Vert V _ j\Vert _ 2^2$ ！
   * **联合 KV 剪枝与 Softmax 重归一化修正（Joint KV Saliency $\Omega _ j^{KV}$ ）**：
     当真正从缓存中移除第 $j$ 个 KV 对（即令未归一化 logit $Z _ {i,j} \to -\infty$ ）时，剩余 Token $k \neq j$ 的注意力权重精确变为 $\tilde{A} _ {i,k} = \frac{A _ {i,k}}{1 - A _ {i,j}}$ 。因此，移除第 $j$ 个 KV 对在第 $i$ 个查询位置引起的**精确输出残差**为：

$$
\Delta O _ i^{(-j)} = O _ i - \tilde{O} _ i^{(-j)} = O _ i - \frac{O _ i - A _ {i,j} V _ j}{1 - A _ {i,j}} = \frac{A _ {i,j}}{1 - A _ {i,j}} \big( V _ j - O _ i \big)
$$

对该精确残差在所有查询位置 $i \in \lbrace1, \dots, S _ q\rbrace$ 上求二阶能量，即得到极其优雅的**联合 KV 闭式显著性得分**：

$$
\Omega _ j^{KV} = \frac{1}{2} \sum _ {i=1}^{S _ q} \left( \frac{A _ {i,j}}{1 - A _ {i,j}} \right)^2 \big\Vert V _ j - O _ i \big\Vert _ 2^2
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* **即插即用全面提升主流基线**：在 **Llama-3.1-8B-Instruct**、**Qwen-2.5-7B/14B-Instruct** 与 **Mistral-7B** 上，将 OBCache 的 $\Omega _ j^{KV}$ 闭式打分直接替换 H2O、SnapKV 与 PyramidKV 的启发式打分（零额外超参），在 **LongBench**（16 个长文本任务）与 **RULER**（128K 极限大海捞针与多跳追踪）上，在仅保留 **5%–10% KV 缓存预算**下将平均准确率提升 **`+2.8%` 至 `+6.4%`**。
* **计算开销近乎为零**： $\Vert V _ j - O _ i\Vert _ 2^2 = \Vert V _ j\Vert _ 2^2 - 2 \langle V _ j, O _ i \rangle + \Vert O _ i\Vert _ 2^2$ 可直接复用 FlashAttention 已经算出的输出向量 $O _ i$ ，无需显式物化完整的 $S _ q \times S _ k$ 矩阵，Prefill 延迟增加小于 `1.2%`。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
1. **对我们 `vla-dtr` & `Efficient Ads / HisTrim` 中 `Exclude-Self Value-Space Perpendicular KV Pruning` 的精确二阶理论证明！**
   * 请仔细对比 OBCache 的核心公式 $\Omega _ j^{KV} = \frac{1}{2}\sum _ i \left(\frac{A _ {i,j}}{1 - A _ {i,j}}\right)^2 \Vert V _ j - O _ i\Vert _ 2^2$ 与我们在 `vla-dtr`（定律 5）和 `ads-rsi` 中独立提出的 **`Exclude-Self Value-Space Perpendicular VLM KV Pruning`**：
     * 其中的因子 $\frac{A _ {i,j}}{1 - A _ {i,j}}$ 正是**排除自身注意力权重后的重归一化系数（Exclude-Self Renormalization）**！
     * 其中的 $\Vert V _ j - O _ i\Vert _ 2^2$ 度量的正是第 $j$ 个 Token 的 Value 向量相对于当前聚合输出均值 $O _ i$ 的**偏离能量（即正交/非共线奇异度）**！如果 $V _ j \approx O _ i$ （即该 Token 的 Value 与上下文均值完全共线/冗余），即便 $A _ {i,j}$ 再大， $\Vert V _ j - O _ i\Vert _ 2^2 \approx 0$ ，驱逐它也完全不改变注意力输出！
2. **落地融合方案（Perp-OBCache）**：
   * 在我们的论文撰写与代码实现中，可以直接引用 ICML 2026 的 OBCache 作为二阶泰勒理论背书，并指出我们进一步将 $\Vert V _ j - O _ i\Vert _ 2^2$ 投影到了输出投影矩阵 $W _ O$ 之后的残差切空间 $\Vert(V _ j - O _ i) W _ O P _ \perp(h _ i)\Vert _ 2^2$ ，从而构成了比 OBCache 更进一层的**流形正交切空间二阶最优脑缓存剪枝（Manifold-Orthogonal OBCache）**。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

> **赛道锚点**：前沿研发智能体递归自我改进（Agent Harness RSI）、抗过拟合正则化进化、可执行代码物理世界模型（Code as Worlds）。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`attention_xsa/value_space_projection.py` (Output-Aware Value Jacobian Perturbation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-27_ai_paper_notes.md`


---

### 3.11 [2026-09-26] 🔄 *LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language Modeling*
> **聚焦领域**：Looped Transformers · Mixture of Experts (MoE) · Iterative Depth Scaling · Weight Sharing  
> **arXiv**：[`arXiv:2606.04438`](https://arxiv.org/abs/2606.04438)

```
  输入表征 h^{(0)} ──► [ 循环步 t = 1..K : IterAdaLN(h, t) 轮次特征调制 ]
                                       │
                                       ▼
                     [ 共享 MoE 路由层: Top-k 稀疏专家激活 + 跨循环容量均衡 ]
                                       │
                                       ▼
                     [ 解耦总参数量 P 与单 Token 算力 FLOPs (同参数量 PPL 显著降低) ]
```

#### 🎯 背景与痛点剖析 (Problem Statement)
* **权重复用与轮次角色分化的矛盾**：在 Looped Transformer 中，直接将同一组 Transformer 块重复循环 $K$ 次，虽然能以 $O(1)$ 参数开销换取 $O(K)$ 的等效推理深度，但会导致两个严重退化：（1）不同循环步 $t \in \lbrace1, \dots, K\rbrace$ 缺乏步间身份区分，引发梯度震荡与隐状态平行分量 $\Delta h _ \parallel$ 爆炸；（2）若将循环架构直接与 MoE 结合，不同循环步会争抢同一批头部 Expert，导致严重的跨循环路由坍缩（Cross-Loop Routing Collapse）。

#### 💡 核心方法与底层数学实现 (Mathematical Formulations)
1. **迭代步自适应层归一化 (Iteration-Adaptive LayerNorm, `IterAdaLN`)**：
   - 为第 $t$ 次循环引入轻量级步间嵌入向量 $e _ t \in \mathbb{R}^d$ ，对共享主干的归一化层施加轮次特异性的仿射缩放与偏移调制：

$$
\text{IterAdaLN}(h^{(t)}, t) = \big(1 + \gamma(e _ t)\big) \odot \frac{h^{(t)} - \mu}{\sigma} + \beta(e _ t)
$$

   - 通过仅占总参数量 $<0.1$ % 的步间条件调制参数，赋予共享 MoE 块在不同循环深度下截然不同的几何变换角色。
2. **跨循环容量感知负载均衡 (Iteration-Aware Capacity Balancing)**：
   - 设第 $t$ 步第 $i$ 个专家的路由门控概率为 $p _ i^{(t)}(x)$ ，论文将辅助负载均衡损失扩展至循环时间轴与批次维度的联合分布上，防止特定专家在连续多次循环中被重复饱和激活。

#### 📊 关键实验与结论 (Experiments & Findings)
* **等参数量与等 FLOPs 双向碾压**：在语言建模基准与常识推理任务上，循环 $K=2\sim 4$ 步的 `LoopMoE` 在相同活跃参数量下显著优于标准稠密 Looped 模型，且在相同总参数预算下逼近非共享深层 MoE 模型的困惑度（PPL）上限。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作与在研主线**：
  * [Paper #16: *Disentangling Representation Evolution in Transformers through Directional Decomposition* (EMNLP 2026, `arXiv:2609.15975`)]
  * [Paper #11: *Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts* (ICLR 2026)]
  * [Paper #10: *Router-Tuning for Dynamic Mixture of Experts* (EMNLP 2025)]
  * [Active Line: *Physical AI / VLA-Loop (Stage-Wise Multi-LoRA Residual Boost & Adaptive Layer Looping)*]
* **🔬 机理对比与技术演进**：
  * `LoopMoE` 采用 `IterAdaLN`（逐通道对角缩放 $\gamma(e _ t)$ ）来区分不同循环轮次；而我们在 `VLA-Loop`（见 W39 研发笔记 9/22–9/23）中提出**用极小秩的 Stage-Wise LoRA 去编辑共享主干的每一次循环**，并进一步推进到了**逐层自适应决定是否 Loop**；
  * 从我们 *Transformer-Geometry (EMNLP 26)* 的正交方向分解视角来看，`IterAdaLN` 仅在归一化后施加坐标轴缩放，主要调节平行缩放分量 $\Delta h _ \parallel$ ；而我们的 **共享主干 + 轮次轻量 LoRA ( $\Delta W _ t = B _ t A _ t$ )** 则能直接在子空间中引入低秩正交旋转分量 $\Delta h _ \perp$ ，在表达能力上严格包含 `IterAdaLN`！
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 在撰写 `Physical AI` (MLSys) 论文的 Loop 章节时，可将 `LoopMoE` 的 `IterAdaLN` 作为轻量轮次调制的文献对照基准，用实验展示我们 **“共享主干 + MERA 初始化的轮次小 LoRA + 逐层自适应 Loop 路由”** 相比单纯 LayerNorm 调制的显著几何表达优势。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/` & `attention_xsa/` (`arXiv:2609.15975`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-26_ai_paper_notes.md`


---

### 3.12 [2026-09-26] ⚖️ *SelKV: Selective KV Cache Merging with Per-Token Merge-or-Drop and Attention Compensation*
> **聚焦领域**：KV Cache Compression · Softmax Denominator Compensation · Token Merging vs. Dropping  
> **arXiv**：[`arXiv:2607.16213`](https://arxiv.org/abs/2607.16213)

```
  待压缩历史 Token 序列 ──► [ 软余弦门控 (Soft Cosine Gate) 评估 Value 流形相似度 ]
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
             [ 高相似度: 加权合并 KV ]        [ 低相似度低重要度: 直接丢弃 ]
                        └──────────────┬──────────────┘
                                       ▼
               [ 注意力比率补偿 (Attention-Ratio Logit Compensation) ]
               消除 Softmax 分母塌陷 (Attention Sag) ──► 免训练高压缩保真
```

#### 🎯 背景与痛点剖析 (Problem Statement)
* **为什么免训练剪枝/合并会导致“注意力塌陷（Attention Sag）”**：当我们在推理期丢弃或合并大量历史 Token 后，参与 Softmax 计算的 Key 数量从 $N$ 锐减至 $M$ （ $M \ll N$ ）。若直接对剩余 $M$ 个 Token 的内积得分做标准 Softmax 归一化，原本被大量被删 Token 分担的分母配分函数质量消失，导致剩余 Token（或合并簇）的注意力权重被人为膨胀或失衡，深层表征模长发生剧烈偏移。

#### 💡 核心方法与数学推导 (Mathematical Formulations)
1. **软余弦门控决定“合并还是丢弃” (Soft Cosine Gate for Merge-or-Drop)**：
   - 给定被淘汰候选 Token $i$ 及其在保留集合中的最近邻锚点 $j^\star$ ，计算其 Value 向量的余弦相似度 $s _ i = \cos(v _ i, v _ {j^\star})$ ；
   - 通过平滑门控函数 $g(s _ i) = \sigma(\alpha (s _ i - \tau))$ 动态决定将其特征并入锚点 $j^\star$ （当 $s _ i > \tau$ ）还是直接丢弃（当 $s _ i \le \tau$ ）。
2. **注意力比率对数补偿 (Attention-Ratio Compensation)**：
   - 若锚点 $j^\star$ 吸收了等效计数为 $c _ {j^\star}$ 的历史 Token 质量，则在计算注意力 Logits 时显式加上对数质量补偿项：

$$
\tilde{a} _ {q, j^\star} = \frac{q^\top k _ {j^\star}}{\sqrt{d _ k}} + \ln(c _ {j^\star})
$$

   - 从而保证合并/剪枝前后的 Softmax 分母配分函数 $Z = \sum _ j \exp(\tilde{a} _ {q,j})$ 严格守恒！

#### 📊 关键实验与结论 (Experiments & Findings)
* 在 LongBench、RULER 及多轮数学推理基准上，免训练实现 **5x–10x KV Cache 压缩**，通过引入对数分母补偿项，消除了高压缩比下 80% 以上的精度退化。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作与在研主线**：
  * [Active Line: *Efficient Ads & VLA `HisTrim` (Hierarchical Progressive Token Drop + Softmax Denominator Mass Compensation)*]
  * [Paper #15: *Demystifying When Pruning Works via Representation Hierarchies* (ICML 2026)]
  * [Paper #16: *Transformer-Geometry* (EMNLP 2026, `arXiv:2609.15975`)]
* **🔬 机理对比与技术演进**：
  * **这篇工作独立验证了我们本周在 `Efficient Ads` 与 `axon` FlashAttention 推导中发现的核心机制！** 我们在 W39 周记（9/21）中明确指出：**当丢弃 Token 后，若直接把剩余保留 Token 的注意力权重重新归一化到 100%，会引发 $>1\times$ 的权重膨胀（分母偏差 / Denominator Bias）**，并推导出了 FlashAttention LSE（ $L _ i = m _ i + \ln \ell _ i$ ）下的 `$+\ln(M)$` 对数配分函数补偿与特殊 Token（Attention Sink）保留机制；
  * `SelKV` 在免训练 KV 合并场景下观测到了完全相同的现象（其命名为 *Attention Sag*），并用 $+\ln(c _ {j^\star})$ 予以修正。
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 在正在撰写的 `Efficient Ads`（冲刺 NAACL）正文中，可将 `SelKV` 与我们的分母偏差修正共同作为**“Token 稀疏化中的 Softmax 配分函数守恒定律”**的双向佐证，进一步强化我们把“分母偏差 ↔ 位置编码与 Attention Sink”作为核心机制贡献（而非工程补丁）的理论厚度！

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/` & `attention_xsa/` (`arXiv:2609.15975`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-26_ai_paper_notes.md`


---

### 3.13 [2026-09-25] Fully Looped Transformer: Stabilizing Looped Models via Attention Injection and Residual Scaling

* **论文信息**：`arXiv:2605.18797` (2026-05)
* **核心关键词**：Fully Looped Transformer、Attention Injection、Anchor KV Grounding、Gradient Oscillation Prevention

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       Fully Looped Transformer with Parameter-Free Initial Attention Injection    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Initial Pass (k=0): Input Embedding H^{(0)} ---> Compute Anchor (K^{(0)}, V^{(0)})|
|                                        |                                          |
|                                        v                                          |
|  Loop Iteration k = 1 .. K:                                                       |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Anchor-Injected Multi-Head Attention (零参数初始锚点键值注入)            |  |
|  |    \tilde{K}^{(k)} = (1 - \lambda_k) K^{(k)} + \lambda_k K^{(0)}            |  |
|  |    \tilde{V}^{(k)} = (1 - \lambda_k) V^{(k)} + \lambda_k V^{(0)}            |  |
|  |    Prevents representation drift & provides direct gradient highway to k=0  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Unit-Sphere / Variance-Preserving Residual Update                        |  |
|  |    H^{(k+1)} = \text{Norm}\big( H^{(k)} + \frac{1}{\sqrt{K}} f_\theta(H^{(k)}, \tilde{K}^{(k)}, \tilde{V}^{(k)}) \big)|
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **深层循环中的“初始锚点遗忘”与反向传播雅可比谱半径失控**：当一个循环 Transformer 连续迭代 $K \ge 8$ 步时，第 $k$ 步的隐状态 $H^{(k)}$ 经过反复的非线性自注意力和 FFN 变换后，逐渐丢失了原始输入 Token 的精细词法锚点信息；同时在反向传播（BPTT）中，共享权重连乘 $\prod _ {k=1}^K \big(I + \frac{\partial f _ \theta}{\partial H^{(k)}}\big)$ 极易引发梯度震荡或消失。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **零参数初始注意力注入（Parameter-Free Attention Injection）**：
   缓存首轮（ $k=0$ ）计算得到的初始键值张量 $\left(K^{(0)}, V^{(0)}\right)$ 。在后续任意第 $k \in \lbrace1, \dots, K\rbrace$ 次循环中，通过凸组合或拼接将初始锚点注入当前步的注意力键值中：

$$
O^{(k)} = \text{Softmax}\left( \frac{Q^{(k)} \big( (1-\lambda) K^{(k)} + \lambda K^{(0)} \big)^\top}{\sqrt{d _ k}} \right) \Big( (1-\lambda) V^{(k)} + \lambda V^{(0)} \Big)
$$

   这一设计在计算图上为每一个循环步 $k$ 建立了一条直通初始表征 $\left(K^{(0)}, V^{(0)}\right)$ 的**一阶梯度短路高速通道（Direct Gradient Highway）**：

$$
\frac{\partial \mathcal{L}}{\partial H^{(0)}} = \frac{\partial \mathcal{L}}{\partial H^{(K)}} \prod _ {k=1}^K J _ k + \lambda \sum _ {k=1}^K \frac{\partial \mathcal{L}}{\partial O^{(k)}} \frac{\partial O^{(k)}}{\partial (K^{(0)}, V^{(0)})} \frac{\partial (K^{(0)}, V^{(0)})}{\partial H^{(0)}}
$$

   从而彻底消除了高循环步数下的梯度消失与震荡！

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在完全不增加任何额外参数（0 Extra Parameters）的条件下，Fully Looped Transformer 在 $K=8, 12$ 步循环预训练中完全消除了传统 Looped Transformer 的梯度尖峰（Gradient Spikes），验证集困惑度（PPL）降低 **`1.45`**，下游推理基准提升 **`+4.9%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **直接印证我们 `vla-loop` 定律（Lightweight Dropped-Span VLM Cross-KV Grounding）！**
  * 我们在 `vla-loop` 中发现，当动作专家循环迭代 $K=3,4$ 步时，若每一步都强绑回初始锚点 VLM Prefix KV（即此处的 $\left(K^{(0)}, V^{(0)}\right)$ ），即可完美阻止循环轨迹漂移！该论文的梯度短路公式为我们 `vla-loop` 的 Cross-KV Grounding 提供了极其漂亮的反向传播雅可比谱稳定性证明。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/residual_decomposition.py` (Hyper-Connections Orthogonal Dispersion)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-25_ai_paper_notes.md`


---

### 3.14 [2026-09-25] On the Limits of Layer Pruning in Generative Reasoning LLMs

* **论文信息**：`arXiv:2602.01997` (2026-02)
* **核心关键词**：Limits of Layer Pruning、Sequential Circuit Depth、Multi-Step Arithmetic & Logic Degradation

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       Limits of Layer Pruning: Shallow Knowledge Lookup vs. Compositional Depth   |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Task Type A: Fact Retrieval / Single-Hop QA (MMLU, ARC-Easy, HellaSwag)          |
|    Parallel Associative Memory Circuits ---> Tolerates 30%-40% Layer Pruning!     |
|                                                                                   |
|  Task Type B: Multi-Step Compositional Reasoning (GSM8K, MATH, Symbolic Carry)    |
|    Requires Sequential Circuit Depth D_{\min} >= m \cdot d_{\text{hop}}           |
|    When remaining layers L_{\text{keep}} < D_{\min}:                              |
|    ===> Sharp Cliff Collapse (Even with LoRA recovery!)                           |
|                                        |                                          |
|                                        v                                          |
|  Solution: Convert Pruned Physical Layers into Shared Looped Iterations!          |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **层剪枝评估中的“多项选择幸存者偏差”**：大量层剪枝论文声称剪掉 30% 的层后在 HellaSwag、PIQA、Winogrande 甚至 MMLU 选择题上保留了 95% 性能。然而作者通过系统性压力测试发现，同一批被剪枝模型在自由生成的多步算术、代码执行追踪与符号逻辑推理任务上性能暴跌超过 **40%–65%**。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **基于计算复杂性理论的串行电路深度下界（TC $^0$ Sequential Depth Lower Bound）**：
   单个自注意力+FFN 层属于常数深度阈值电路类 $\text{TC}^0$ 。对于包含 $m$ 步嵌套函数复合 $g _ m \circ g _ {m-1} \circ \dots \circ g _ 1(x)$ （如多位数连加进位链或 $m$ 跳变量代换）的单个前向步推理，若没有外部 CoT Token 展开，模型内部必须至少具备 $L _ {\text{eff}} \ge m \cdot c _ {\text{hop}}$ 个串行非线性消息传递层。
   一旦物理层剪枝使剩余层数 $L _ {\text{keep}} = (1 - p) L < m \cdot c _ {\text{hop}}$ ，任何静态线性适配器或宽度扩容都无法弥补串行电路深度的缺失：

$$
\inf _ {\theta \in \Theta _ {L _ {\text{keep}}}} \mathbb{P}\big( f _ \theta(x) \neq g _ m \circ \dots \circ g _ 1(x) \big) \ge \frac{1}{2} - \exp\big(-\Omega(N^{\epsilon})\big) \quad \text{whenever } L _ {\text{keep}} < m \cdot c _ {\text{hop}}
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 实验精确测定了 Llama-3-8B/70B 与 Qwen-2.5 在不同推理跳数 $m \in \lbrace2, 3, 4, 5\rbrace$ 下的临界剩余层数 $L _ {\text{crit}}(m)$ ，并证明当物理层被剪除后，**唯有通过测试期层循环（Layer Looping）恢复有效串行深度 $L _ {\text{eff}}$ **，才能跨过生成式推理的电路深度下界！

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **为我们为何从单纯的静态层剪枝（`vla-dtr` / *Layer Dropping* TMLR 2025）走向“层剪枝 + 循环精化协同（`vla-loop`）”提供了最坚实的复杂度理论支撑！**
  * 在撰写我们的论文导论（Introduction）与理论动机（Motivation）时，该定理可直接引用：静态深度剪枝省下了显存但突破了串行复合电路深度下界 $L _ {\text{crit}}$ ，而通过 1-Pass 主干 + LoRA 循环级联恰好以零额外主干显存恢复了所需的有效复合深度 $L _ {\text{eff}}$ ！

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`compression/layer_geometry_pruning.py` (Reasoning Hop Depth Lower Bound)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-25_ai_paper_notes.md`


---

### 3.15 [2026-09-25] How Pruning Attention Layers Affects Interpretability, Faithfulness, and Confidence Calibration

* **论文信息**：`arXiv:2606.24970` (2026-06)
* **核心关键词**：Attention Layer Pruning、Confidence Calibration (ECE)、Faithfulness、Overconfident Hallucination

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|     Impact of Attention Layer Pruning on Faithfulness & Confidence Calibration    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Pruned Mid-Deep Attention Layers ---> Loss of "Inhibitory / Suppression Heads"   |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Pathology Diagnosis: Logit Norm Inflation & Entropy Collapse                |  |
|  |    || h^{(L)}_{\text{pruned}} ||_2 > || h^{(L)}_{\text{orig}} ||_2          |  |
|  |    Expected Calibration Error (ECE) spikes by 2.5x - 4.0x!                  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Fix: Inhibitory Subspace Projection + Variance-Matched Logit Rescaling      |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **剪枝后模型的“过度自信幻觉（Overconfident Hallucination）”**：作者发现，许多在中深层被视作“低贡献”而被剪除的注意力层，实际上包含了关键的**抑制头（Suppression / Negative Heads）**——它们的作用是在上下文证据不足或存在冲突时压低错误候选词的 Logit。剪除这些层后，虽然 Top-1 准确率仅轻微下降，但模型的预测分布熵急剧坍缩，期望校准误差（ECE）暴增 3 倍以上！

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **抑制头缺失导致的 Logit 方差膨胀模型**：
   在完整模型中，深层抑制注意力层的输出增量满足 $\langle \Delta h _ {\text{inhib}}^{(l)}, h^{(l-1)} \rangle < 0$ （即对残差流起负反馈阻尼作用）。剪除该层后，终端隐状态平行范数失控放大，导致输出词表概率 $p _ {\text{pruned}}(y \mid x)$ 的期望校准误差（ECE）激增：

$$
\text{ECE} = \sum _ {b=1}^B \frac{|I _ b|}{N} \Big| \text{acc}(I _ b) - \text{conf}(I _ b) \Big|
$$

2. **负反馈阻尼恢复与流形方差对齐**：
   在剪枝切口处引入沿残差主方向的阻尼收缩算子 $\tilde{h} = h - \beta \frac{\langle h, u _ {\text{inhib}} \rangle}{\Vert u _ {\text{inhib}}\Vert _ 2^2} u _ {\text{inhib}}$ 并校准输出层温度 $\tau^\star = \frac{\sigma(\text{logits} _ {\text{pruned}})}{\sigma(\text{logits} _ {\text{orig}})}$ 。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在事实问答（TruthfulQA、haluEval）与医疗/金融高风险推理任务上，该校准修复将深度剪枝模型的 **ECE 降低 68%**，并在基于置信度的拒绝采样（Selective Prediction）中恢复了 98% 的安全边界。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) 的“负平行分量（Negative Parallel Component）”发现完全吻合！**
  * 我们在 *Transformer-Geometry* 中明确观测到中深层部分模块具有 $\Delta h _ \parallel < 0$ 的径向阻尼效应；剪除它们而不做平行范数阻尼补偿，必然导致终端模长膨胀与置信度失真。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`compression/layer_geometry_pruning.py` (Attention vs MLP Subspace Division)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-25_ai_paper_notes.md`


---

### 3.16 [2026-09-25] SAC: Disaggregated KV Cache Architecture for Sparse Attention Serving over CXL

* **论文信息**：`arXiv:2604.18392` (2026-04)
* **核心关键词**：CXL 3.0 Memory Pooling、Disaggregated KV Cache、Sparse Attention Sub-Page Gather

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       SAC: CXL-Disaggregated KV Cache Architecture for Sparse Attention           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  GPU Compute Nodes <--- CXL 3.0 Fabric ---> Shared CXL Memory Pool (TB-Scale KV)  |
|                                                       |                           |
|                                                       v                           |
|  +-----------------------------------------------------------------------------+  |
|  | Near-Memory Sparse Gather Engine on CXL Type-2/3 Controller                 |  |
|  |    Receives Top-k sparse token indices from GPU -> Packs only selected      |  |
|  |    cachelines into dense CXL flits -> 6.5x effective bandwidth amplification|  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **稀疏注意力在 PCIe/CXL 远端内存读取时的粒度放大（Granularity Amplification）**：当稀疏注意力仅需读取分散在不同物理页中的少量关键 Token 时，传统 DMA 以 4KB 页为单位搬运会导致高达 85% 的无效带宽浪费。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **CXL 控制器端近存稀疏聚集与头维度转置存储**：
   在 CXL 内存池侧按缓存行（64B Cacheline）对齐存储单头量化 KV 向量，由 CXL 控制器根据 GPU 下发的稀疏索引列表 $\mathcal{I} _ {\text{top-}k}$ 在远端完成紧密打包（Dense Packing）后再经 CXL.mem 链路回传：

$$
\text{BW} _ {\text{eff}} = \text{BW} _ {\text{CXL}} \cdot \frac{d _ {\text{head}} \cdot b _ {\text{quant}}}{\lceil d _ {\text{head}} \cdot b _ {\text{quant}} / 64\text{B} \rceil \cdot 64\text{B}} \approx 0.94 \cdot \text{BW} _ {\text{CXL}}
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 TB 级长上下文并发推理中，SAC 将跨节点 KV 读取有效带宽利用率从 `15%` 提升至 **`94%`**，P99 尾延迟降低 **3.7x**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **为我们的 SelKV / OBCache 稀疏缓存算法在大规模分布式机架上的部署提供了硬件近存聚集蓝图**。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`attention_xsa/value_space_projection.py` (Output-Aware Value Jacobian Perturbation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-25_ai_paper_notes.md`


---

### 3.17 [2026-09-24] Training-Free Looped Transformers: Test-Time Mid-Stack Layer Looping

* **论文信息**：`arXiv:2605.23872` (2026-05)
* **核心关键词**：Training-Free Looped Transformer、Test-Time Depth Scaling、Mid-Stack Fixed-Point Iteration

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       Training-Free Looped Transformers: Test-Time Mid-Stack Layer Looping        |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Frozen Checkpoint: [Shallow Layers 1..l_a-1]                                     |
|                              |                                                    |
|                              v                                                    |
|        +---> [Mid-Stack Reasoning Span: Layers l_a .. l_b] ---+                   |
|        |                     |                                |                   |
|        |          Loop K times at Test Time                    |                   |
|        +--- Damped Contraction: h <- (1-\eta)h_{\text{in}} + \eta h_{\text{out}}  |
|                              |                                                    |
|                              v                                                    |
|                     [Deep Readout Layers l_b+1..L]                                |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **能否在不重新训练的情况下让现成开源大模型享受循环深度扩展？** 以往工作普遍认为 Looped Transformer 必须从头带循环拓扑预训练，否则直接把某一层重复执行会导致隐状态偏离后续层期望的输入流形。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **中段层块的近似压缩不动点迭代性质（Mid-Stack Contractive Mapping）**：
   作者分析发现，在预训练 Transformer 的中间深层区间 $[l _ a, l _ b]$ （通常位于 $0.4L \sim 0.75L$ ），相邻层的输入输出处于同一缓变语义流形上，复合块算子 $\mathcal{F} _ {l _ a:l _ b}$ 在局部切空间上近似构成压缩不动点精化映射。
2. **阻尼流形拉回循环更新（Damped Manifold-Preserving Loop）**：
   为防止在测试期重复调用 $\mathcal{F} _ {l _ a:l _ b}$ 时隐状态范数越界，在第 $k$ 次额外循环后施加范数匹配与阻尼凸组合：

$$
h^{(k)} = \frac{\Vert h^{(0)}\Vert _ 2}{\Vert\tilde{h}^{(k)}\Vert _ 2} \tilde{h}^{(k)}, \qquad \text{where } \tilde{h}^{(k)} = (1 - \eta) h^{(k-1)} + \eta \mathcal{F} _ {l _ a:l _ b}(h^{(k-1)})
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在完全零训练（Zero Finetuning）的 **Llama-3-8B** 与 **Mistral-7B** 上，对中段 6 层额外循环 $K=2$ 次，在 GSM8K、ARC-Challenge 与逻辑推理任务上直接获得 **`+2.1%` 至 `+3.8%`** 的免费准确率提升。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `vla-loop`（Layer-Specific Span-Bounded Dynamic Halting）及 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) 高度同源**：
  * 该文通过范数重缩放 $\frac{\Vert h^{(0)}\Vert _ 2}{\Vert\tilde{h}^{(k)}\Vert _ 2}$ 抑制测试期循环发散，本质上正是我们在 *Transformer-Geometry* 中指出的**抑制平行径向膨胀、仅保留球面切向正交精化**！

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/` & `attention_xsa/` (`arXiv:2609.15975`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-24_ai_paper_notes.md`


---

### 3.18 [2026-09-23] MELT: Memory-Efficient Looped Transformer — Decoupling Compute from Memory

* **论文信息**：`arXiv:2605.07721` (2026-05)
* **核心关键词**：Memory-Efficient Looped Transformer、Shared Cross-Loop KV Cache、Compute-Memory Decoupling

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|         MELT: Memory-Efficient Looped Transformer (Shared KV Cache Pool)          |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Standard Looped Transformer (K Loops):                                           |
|    Stores separate KV^{(1)}, KV^{(2)}, ..., KV^{(K)} -> K x Memory Footprint!     |
|                                                                                   |
|  MELT Architecture:                                                               |
|    Single Physical KV Cache Buffer \mathcal{C}_{KV} in HBM                        |
|    Loop k=1..K reads & refines \mathcal{C}_{KV} via gated EMA update:             |
|    \mathcal{C}_{KV}^{(k)} = (1 - \alpha_k) \mathcal{C}_{KV}^{(k-1)} + \alpha_k \text{Proj}_{KV}(h^{(k)})|
|    ===> O(K) Compute Depth with strictly O(1) KV Cache Memory!                    |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **循环 Transformer 的“隐性 KV 缓存倍增陷阱”**：虽然 Looped Transformer 通过复用层权重将模型参数显存压缩为 $1/K$ ，但在自回归生成时，如果第 $t$ 个 Token 在第 $k$ 次循环时需要 Attend 到前序 Token $1 \dots t-1$ 在第 $k$ 次循环时的键值状态，就必须为全部 $K$ 次循环分别缓存独立的 $K^{(k)}, V^{(k)}$ ，导致 KV 缓存显存依然随循环步数 $K$ 线性增长！

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **跨循环指数移动平均共享 KV 缓存（Cross-Loop EMA Shared KV Cache）**：
   对于历史已生成的上下文位置 $1 \dots t-1$ ，仅在显存中维护唯一一份最终收敛态的键值缓存 $\left(K _ {\text{shared}}, V _ {\text{shared}}\right)$ （即每个历史 Token 完成第 $K$ 次循环后的稳态 KV）。在当前位置 $t$ 执行第 $k \in \lbrace1, \dots, K\rbrace$ 次内部循环时，当前查询 $q _ t^{(k)}$ 统一读取历史稳态缓存 $K _ {\text{shared}, 1:t-1}$ 并结合当前步自键值 $\left(k _ t^{(k)}, v _ t^{(k)}\right)$ ：

$$
\text{Attn} _ t^{(k)} = \text{Softmax}\left( \frac{q _ t^{(k)} \big[ K _ {\text{shared}, 1:t-1}; k _ t^{(k)} \big]^\top}{\sqrt{d _ k}} \right) \begin{bmatrix} V _ {\text{shared}, 1:t-1} \cr v _ t^{(k)} \end{bmatrix}
$$

   当第 $t$ 个 Token 完成全部 $K$ 步循环后，仅将其终端稳态 $\left(k _ t^{(K)}, v _ t^{(K)}\right)$ 写入共享缓存池！

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 $K=4$ 与 $K=8$ 循环配置下，MELT 将长文本解码时的 **KV 缓存显存与带宽读取量直接削减 $75\text{ pct}–87.5$ %（严格降至 $1/K$ ）**，同时在语言建模与数学推理上与保存全套每步 KV 的基线性能完全持平（差异 `<0.2%`）。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **直接印证我们 `vla-loop` 定律 v19/v20（1-Pass Backbone + Multi-Step LoRA-Only Cascade & Shared KV Grounding）**：在 Looped VLA 中，历史观测与前缀只需保存唯一一份稳态 KV 缓存，多步循环仅更新当前动作查询状态，从而将循环推理的内存带宽开销降到最低。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/residual_decomposition.py` (Terminal-Iteration Latent Convergence)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-23_ai_paper_notes.md`


---

### 3.19 [2026-09-22] SPIN: Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving

* **论文信息**：`arXiv:2604.26837` (2026-04)
* **核心关键词**：Sparse Attention Serving、Hierarchical GPU-CPU Memory、Asynchronous Layer-Ahead Prefetching

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       SPIN: Unifying Sparse Attention with Hierarchical Memory Serving            |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  GPU HBM: [Compact Page Indices + Hot Anchor KV Cache (10%)]                      |
|  CPU DRAM: [Full Cold KV Cache Pool (100%)]                                       |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Layer l-1 Hidden State Speculative Index Prediction                         |  |
|  |    Predict Top-K sparse pages needed by Layer l BEFORE Layer l starts       |  |
|  |    Overlap PCIe/NVLink DMA prefetch of missing cold pages with Layer l-1 FFN|  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **动态稀疏注意力的 PCIe 按需拉取延迟陷阱**：若将全量 KV 缓存卸载至 CPU 内存并在每层动态选出 Top- $k$ 页面后才通过 PCIe 搬运回 GPU，PCIe 传输延迟将远超稀疏注意力节省的计算时间。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **跨层隐状态余弦惯性预取（Cross-Layer Speculative Page Prefetching）**：
   利用相邻层查询向量高度相似的几何惯性（ $\cos(Q^{(l-1)}, Q^{(l)}) > 0.9$ ），在第 $l-1$ 层计算注意力的同时，使用轻量级页中心内积 $\hat{s} _ p^{(l)} = Q^{(l-1)} \bar{K} _ p^{(l)\top}$ 提前预测第 $l$ 层所需的冷页集合 $\mathcal{P} _ {\text{miss}}^{(l)}$ ，实现计算与 PCIe DMA 搬运的完美流水线掩盖：

$$
T _ {\text{step}}^{(l)} = \max\Big( T _ {\text{FFN}}^{(l-1)} + T _ {\text{QKV}}^{(l)}, \frac{|\mathcal{P} _ {\text{miss}}^{(l)}| \cdot B _ {\text{page}}}{\text{BW} _ {\text{PCIe}}} \Big)
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在单台 8 卡服务器上支持 **1M–2M 上下文长度** 并发推理，相比纯 CPU Offloading（Infinite-LLM）实现 **4.8x** 吞吐提升，且恢复 99.7% 全量注意力精度。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 *Transformer-Geometry* (EMNLP 2026) 的层间方向平稳性定理天然契合**：正是因为深层残差流中平行分量占主导、层间角度旋转平缓，才保证了跨层提前 1–2 层预取稀疏 KV 页的高命中率！

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/` & `attention_xsa/` (`arXiv:2609.15975`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-22_ai_paper_notes.md`


---

### 3.20 [2026-09-21] DeepLoop: Depth Scaling for Looped Transformers

* **论文信息**：`arXiv:2607.13491` (2026-07)
* **核心关键词**：Looped Transformers、Residual-Scaling Problem、Coherent Variance Growth、Depth Scaling Law

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|               DeepLoop: Depth Scaling for Looped Transformers                     |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Unrolled Standard Transformer (Independent Weights W_l):                         |
|    \text{Var}(h^{(L)}) \approx \text{Var}(h^{(0)}) + \sum_{l=1}^L \sigma_l^2 = O(L)|
|                                                                                   |
|  Naive Looped Transformer (Shared Weight W reused K times):                       |
|    Coherent alignment \langle f_W(h^{(k)}), f_W(h^{(j)}) \rangle > 0              |
|    ===> \text{Var}(h^{(K)}) = O(K^2)  [Catastrophic Residual & Gradient Explosion]|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | DeepLoop Coherent-Aware Residual Scaling & Step-Conditioned Norm            |  |
|  |    h^{(k)} = h^{(k-1)} + \frac{\alpha_k}{K^{\gamma}} f_W\big(\text{LN}_k(h^{(k-1)})\big)|
|  |    where \gamma \in [1/2, 1] interpolates between diffusive & coherent drift|  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **循环复用的“相干方差爆炸（Coherent Variance Explosion）”**：在标准非循环 Transformer（如 DeepNorm / Pre-LN）中，由于各层权重 $W^{(l)}$ 相互独立，层间残差增量的交叉协方差近似为零，因此 $L$ 层后的隐状态方差按随机游走以 $O(L)$ 线性增长（仅需 $1/\sqrt{L}$ 缩放）。然而在 **Looped Transformer** 中，同一物理层 $f _ W$ 被连续迭代调用 $K$ 次，第 $k$ 步的残差增量 $f _ W(h^{(k-1)})$ 与前一步高度正相关（相干叠加），导致隐状态范数以 ** $O(K^2)$ 二次方速度爆炸**，使得循环步数 $K > 4$ 时训练迅速崩溃！

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **相干循环残差方差增长定理（Coherent Residual Variance Theorem）**：
   设循环块映射为 $h^{(k)} = h^{(k-1)} + \beta _ k f _ W(h^{(k-1)})$ 。令步间余弦相关系数为 $\rho _ {j,k} = \frac{\mathbb{E}[\langle f _ W(h^{(j)}), f _ W(h^{(k)}) \rangle]}{\Vert f _ W(h^{(j)})\Vert _ 2 \Vert f _ W(h^{(k)})\Vert _ 2}$ 。当 $\rho _ {j,k} \ge \bar{\rho} > 0$ 时， $K$ 步循环后的终端方差满足：

$$
\mathbb{E}\big[\Vert h^{(K)} - h^{(0)}\Vert _ 2^2\big] = \sum _ {k=1}^K \beta _ k^2 \sigma _ f^2 + 2 \sum _ {1 \le j < k \le K} \beta _ j \beta _ k \rho _ {j,k} \sigma _ f^2 = \Theta\left( \Big(\sum _ {k=1}^K \beta _ k\Big)^2 \right)
$$

2. **DeepLoop 步间解耦缩放法则（Coherence-Compensated Scaling Law）**：
   为保证无论循环深度 $K$ 如何扩展，终端隐状态流形半径始终保持 $\Theta(1)$ 李雅普诺夫有界，DeepLoop 引入经验相干指数 $\gamma(\bar{\rho}) = \frac{1}{2} + \frac{1}{2}\bar{\rho} \in [\frac{1}{2}, 1]$ ，设定第 $k$ 步残差门控缩放系数为：

$$
\beta _ k(K) = \frac{c _ k}{K^{\gamma(\bar{\rho})}}, \qquad \text{with step-specific affine gain } \text{LN} _ k(h) = \gamma _ k \odot \frac{h - \mu}{\sigma} + b _ k
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在循环深度从 $K=2$ 扩展至 ** $K=16$ ** 的语言与数学推理预训练中，标准 Pre-LN 循环架构在 $K \ge 6$ 时完全发散，而 **DeepLoop** 稳定收敛并实现随循环次数 $K$ 对数线性下降的测试集 Loss，以 **1/4 的物理参数量** 追平同有效深度标准 Transformer 的推理性能。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **为我们 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) 与 `vla-loop`（定律 v18：Continuous Horizon-Phase Terminal Decay）提供精确的二阶统计力学解释！**
  * DeepLoop 发现的“相干叠加 $\rho _ {j,k} > 0$ 导致 $O(K^2)$ 范数爆炸”，从几何上看正是因为共享权重 $f _ W$ 在每次循环中持续向**平行径向分量 $\Delta h _ \parallel$ ** 注入同向推力！这再次证明了我们在 `vla-loop` 与 *Transformer-Geometry* 中剔除平行分量、仅保留正交切空间更新 $\Delta h _ \perp$ （使 $\rho _ {j,k}^{\parallel} \to 0$ ，从而将方差增长压回良性的 $O(K)$ ）并配合终端步长衰减 $\left(1-\tau _ k\right)^\beta$ 的根本必要性。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/residual_decomposition.py` ( $\Delta h _ \parallel$ Coherent Variance Growth $O(K^2)$ )  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-21_ai_paper_notes.md`


---

### 3.21 [2026-09-21] RotateK: Rotation-Aligned Key Channel Pruning for Vision-Language Models

* **论文信息**：`arXiv:2605.19218` (2026-05)
* **核心关键词**：Key Channel Pruning、Orthogonal Rotation Alignment、Vision-Language Models (VLMs)、Head-Dimension Compression

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|       RotateK: Rotation-Aligned Key Channel Pruning for Vision-Language Models    |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Attention Score Invariance under Orthogonal Rotation R \in O(d_k):               |
|    Q K^\top = (Q R)(K R)^\top   where R^\top R = I_{d_k}                          |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Cross-Modal Key-Query Co-Energy SVD (跨模态查询-键联合能量奇异值对齐)    |  |
|  |    Compute covariance C_K = \mathbb{E}[K_{\text{vis}}^\top K_{\text{vis}}]  |  |
|  |    Eigendecompose C_K = R \Lambda R^\top ---> Fold R into W_Q, W_K offline  |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Tail Channel Truncation (尾部低能量通道截断: 兼容 RoPE 2x2 块旋转)       |  |
|  |    Retain top-r channels (r = 0.4 d_k) -> 60% Key Cache & GEMM Reduction    |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **原始坐标轴下的通道能量弥散**：在多模态大模型（VLM）中，除序列长度方向（Token 维度）冗余外，注意力头内部的特征维度 $d _ k$ （如 $d _ k=128$ ）在视觉特征空间中实际上具有极低的本征秩。然而，在原始训练得到的正交基下，信号能量均匀弥散在全部 128 个通道上，直接按坐标轴剪除任何通道都会造成较大的内积误差 $\Vert Q K^\top - \tilde{Q} \tilde{K}^\top\Vert _ F$ 。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **RoPE 兼容的分块正交旋转能量集中（RoPE-Compatible Block-Orthogonal Rotation）**：
   由于旋转位置编码（RoPE）以二维子平面 $\left(2i, 2i+1\right)$ 为单位作用： $R _ \Theta(m) = \text{diag}(R _ {\theta _ 1}^{(m)}, \dots, R _ {\theta _ {d _ k/2}}^{(m)})$ ，为保持与 RoPE 的可交换性，RotateK 将 $d _ k/2$ 个二维频率对按预期内积能量贡献 $\mathcal{E} _ i = \mathbb{E}\big[ \Vert q _ {[2i:2i+1]} \Vert _ 2^2 \cdot \Vert k _ {[2i:2i+1]} \Vert _ 2^2 \big]$ 进行重排，并在每个同频子空间内执行正交主轴对齐 $U _ i \in O(2)$ ：

$$
\tilde{W} _ Q = W _ Q U _ {\text{rot}}, \qquad \tilde{W} _ K = W _ K U _ {\text{rot}}
$$

2. **误差上界最小化通道截断**：
   保留能量最高的前 $r$ 个通道子块，此时注意力 logit 截断误差满足紧上界：

$$
\mathbb{E}\big[ | q^\top k - \tilde{q} _ {1:r}^\top \tilde{k} _ {1:r} |^2 \big] \le \sum _ {i = r/2 + 1}^{d _ k/2} \lambda _ i(C _ Q) \lambda _ i(C _ K)
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **LLaVA-NeXT**、**Qwen2-VL-7B** 与 **InternVL-2** 上，RotateK 剪除 **50%–60% 的 Key 通道**而无需微调，且与视觉 Token 剪枝（如 FastV / VLA-Pruner）**100% 正交兼容**，联合实现 **4.2x** 注意力加速且 VQA 精度损失 `<0.5%`。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `MerA` SVD 初始化及 *Sparsity for Unified Multimodal Models* (TMLR 2026) 的正交协同**：
  * RotateK 在特征通道维度 $d _ k$ 上的正交旋转浓缩与我们在 Token 维度 $N _ {\text{vis}}$ 上的剪枝构成了完整的二维矩阵联合低秩逼近（Row + Column Dual Sparsity），可直接嵌入 `vla-distillation` 的视觉前缀压缩器中。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`attention_xsa/value_space_projection.py` (Orthogonal Key/Value Subspace Rotation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-21_ai_paper_notes.md`


---

### 3.22 [2026-09-20] SHIFT-LLM: Distribution Shift Correction in Depth-Pruned LLMs

* **论文信息**：`arXiv:2608.25068` (2026-08)
* **核心关键词**：Depth Pruning、Distribution Shift Correction、Linear Residual Adapters (LRA)、Closed-Form Ridge Regression、Weight Folding

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|          SHIFT-LLM: Closed-Form Distribution Shift Correction at Cut Sites        |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Original Stack:  h^{(l-1)} ---> [Pruned Block l..l+m] ---> h_{\text{orig}}^{(l+m)}|
|  Pruned Stack:    \tilde{h}^{(l-1)} -----(Identity Skip)---> \tilde{h}^{(l-1)}    |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Covariate Shift Diagnosis at Pruning Cut Site (剪枝切口协变量偏移诊断)   |  |
|  |    \Delta \mu = \mathbb{E}[h_{\text{orig}}^{(l+m)} - \tilde{h}^{(l-1)}],    |  |
|  |    Angular & norm mismatch causes downstream RMSNorm / Attention saturation |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Closed-Form Linear Residual Adapter (LRA) via Woodbury/Ridge             |  |
|  |    \hat{h}^{(l+m)} = \tilde{h}^{(l-1)} + U_r V_r^\top \tilde{h}^{(l-1)} + b |  |
|  |    Solved in closed form on 128 calibration sequences (Training-Free)       |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **层剪枝切口处的“流形断裂（Manifold Fracture）”**：当直接移除 Transformer 中的第 $l$ 至 $l+m$ 层时，第 $l-1$ 层的输出隐状态 $\tilde{h}^{(l-1)}$ 被直接送入原本期望接收 $h _ {\text{orig}}^{(l+m)}$ 的第 $l+m+1$ 层。由于缺失了中间层的残差漂移与旋转，输入分布的一阶均值 $\mu$ 与二阶协方差矩阵 $\Sigma$ 发生剧烈跳变，导致紧随其后的注意力层 Q/K 点积失真并沿着深层指数级放大。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **剪枝切口处的最小二乘残差重构**：
   设剪枝段输入隐状态矩阵为 $X = \tilde{H}^{(l-1)} \in \mathbb{R}^{N \times d}$ ，原始未剪枝模型在该切口输出的目标残差增量为 $\Delta Y = H _ {\text{orig}}^{(l+m)} - \tilde{H}^{(l-1)} \in \mathbb{R}^{N \times d}$ 。SHIFT-LLM 在切口处插入一个低秩线性残差适配器（LRA） $W _ {\text{LRA}} = U _ r V _ r^\top + \mathbf{1} b^\top$ ，通过带 Tikhonov 正则化的岭回归闭式求解全秩最优映射 $W^\star$ ：

$$
W^\star = \arg\min _ {W \in \mathbb{R}^{d \times d}} \big\Vert \Delta Y - (X - \bar{X}) W \big\Vert _ F^2 + \lambda \Vert W \Vert _ F^2 = \Big( \tilde{X}^\top \tilde{X} + \lambda I _ d \Big)^{-1} \tilde{X}^\top \Delta \tilde{Y}
$$

2. **激活协方差加权奇异值截断（Covariance-Weighted Truncated SVD）**：
   为保证适配器自身的计算开销可忽略（或直接折叠进下一层权重），对预测输出空間执行白化 SVD 分解：

$$
\tilde{X} W^\star = \hat{U} \hat{\Sigma} \hat{V}^\top \implies U _ r = (\tilde{X}^\top \tilde{X} + \lambda I _ d)^{-1/2} \hat{U} _ {:, 1:r} \hat{\Sigma} _ {1:r}^{1/2}, \quad V _ r = \hat{V} _ {:, 1:r} \hat{\Sigma} _ {1:r}^{1/2}
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **Llama-3-8B/70B** 与 **Qwen-2.5-14B** 上剪除 **25%–35% 的层**后，无需任何梯度下降微调（仅需 30 秒闭式矩阵求逆），SHIFT-LLM 将 WikiText2 困惑度（PPL）从 `28.4` 恢复至 **`9.1`**，零样本常识与数学推理平均精度恢复 **`+7.9%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与我们 `modellesion-compression-scaffold`、`vla-dtr` (Ortho-MerA) 及 *Layer Dropping* (TMLR 2025) 的直接印证**：
  * SHIFT-LLM 的闭式岭回归校正算子 $W^\star = (\tilde{X}^\top \tilde{X} + \lambda I)^{-1} \tilde{X}^\top \Delta \tilde{Y}$ 与我们在 `modellesion-compression-scaffold` 中使用的 **Depth SVD-LoRA / Woodbury KKT 闭式残差补偿** 数学形式完全一致！更进一步，结合我们的 `vla-dtr`（Ortho-MerA），我们只需对正交切空间残差 $\Delta Y _ \perp = \Delta Y \cdot P _ \perp(X)$ 进行低秩 SVD 拟合，而将平行分量 $\Delta Y _ \parallel$ 简化为标量增益 $\alpha \in \mathbb{R}$ ，即可用一半的秩恢复更高的几何保真度。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`compression/layer_geometry_pruning.py` (Closed-Form Orthogonal Drift Compensation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-20_ai_paper_notes.md`


---

### 3.23 [2026-09-20] Minima-KV: Mixed-Format Paged Attention for Extreme KV Cache Compression

* **论文信息**：`arXiv:2608.23834` (2026-08)
* **核心关键词**：Mixed-Precision KV Cache、PagedAttention、Sub-Page Bit-Packing、Reasoning Continuity

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|        Minima-KV: Mixed-Format Paged Attention for Extreme KV Compression         |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Incoming KV Tokens ---> Saliency Tiering: [Tier-0: FP16] [Tier-1: INT4] [Tier-2: INT2]|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Unified Iso-Byte Physical Page Pool (等字节物理页统一内存池)             |  |
|  |    Each Physical Page = 64 KB fixed size:                                   |  |
|  |    * Can store N_0 FP16 tokens OR 4*N_0 INT4 tokens OR 8*N_0 INT2 tokens    |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Warp-Specialized Mixed-Format PagedAttention Kernel                      |  |
|  |    Single CUDA kernel dispatches dequantization per page descriptor header  |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **混合精度 KV 缓存的“页表碎片化与多核启动开销”**：虽然算法层已证明将关键 Token 存为 FP16、次要 Token 存为 INT4/INT2 可逼近无损压缩，但在 vLLM 等生产级 PagedAttention 系统中，传统的物理页（Page Block）按固定 Token 槽位数划分。若不同位宽的 Token 混存，会导致高达 40% 的页内字节对齐浪费（Internal Fragmentation），或被迫拆分为 3 次独立 CUDA Kernel 启动。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **等字节容量物理页抽象（Iso-Byte Physical Page Abstraction）**：
   固定每个物理页的字节容量为 $B _ {\text{page}}$ （如 64 KB）。对于位宽为 $b \in \lbrace16, 4, 2\rbrace$ 的页类型，其容纳的逻辑 Token 槽位数动态缩放为：

$$
C _ {\text{slots}}(b) = \frac{8 \cdot B _ {\text{page}}}{2 \cdot H _ {kv} \cdot d _ h \cdot b + M _ {\text{meta}}(b)}
$$

   其中 $M _ {\text{meta}}(b)$ 为分组量化缩放因子与零点（Scale & Zero-Point）的紧凑页头字节数。
2. **页描述符驱动的单核融合反量化注意力（Single-Kernel Fused Dequant-Attention）**：
   在逻辑页表中增加 2-bit 格式标签 $\text{fmt}(p) \in \lbrace0, 1, 2\rbrace$ ，CUDA Warp 在读取物理页 $p$ 时根据 $\text{fmt}(p)$ 在寄存器内执行即时位解包（Register-Level Bit Unpacking）：

$$
\hat{K} _ p = \text{Unpack} _ {\text{fmt}(p)}(Q _ p^K) \odot s _ p^K + z _ p^K, \qquad S _ p = Q \hat{K} _ p^\top
$$

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **Llama-3.1-70B** 与 **Qwen-2.5-32B** 的 128K 长思维链并发服务中，Minima-KV 实现 **4.6x** 真实物理显存节省（零内部页碎片），将最大并发 Batch Size 提升 **3.9x**，端到端解码吞吐提升 **2.7x**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **直接解决我们昨日精读的 SelKV 与 `Efficient Ads / HisTrim` 混合位宽生产落地瓶颈**：可将我们的正交价值空间显著性打分（Perp-OBCache）作为 Minima-KV 的三档分层准则（FP16 / INT4 / INT2），直接集成进统一等字节页表内核中。

---

## 🔥 板块二：全球前沿热点精选 (Trending Frontier)

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`attention_xsa/value_space_projection.py` (Output-Aware Value Jacobian Perturbation)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-20_ai_paper_notes.md`


---

### 3.24 [2026-09-19] WRP: Forward-Free LLM Depth Pruning via Weight Redundancy

* **论文信息**：`arXiv:2609.09883` (2026-09)
* **核心关键词**：Forward-Free Depth Pruning、Weight Redundancy、Spectral Subspace Alignment、Calibration-Free Layer Dropping

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|            WRP: Forward-Free LLM Depth Pruning via Weight Redundancy              |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Frozen Pretrained Weights {W_Q^{(l)}, W_K^{(l)}, W_V^{(l)}, W_O^{(l)}, W_FFN^{(l)}}|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Effective Layer Operator Construction (无需前向激活的等效层算子构建)     |  |
|  |    \mathcal{T}_{\text{attn}}^{(l)} = W_O^{(l)} W_V^{(l)},                   |  |
|  |    \mathcal{T}_{\text{ffn}}^{(l)}  = W_{\text{down}}^{(l)} W_{\text{up}}^{(l)}| |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Spectral Concentration & Inter-Layer Subspace Redundancy (谱冗余度量)    |  |
|  |    R_{\text{intra}}(l) = 1 - \frac{\exp(H(\sigma^{(l)}))}{d}                |  |
|  |    R_{\text{inter}}(l) = \| U_{1:r}^{(l)\top} U_{\text{prev}}^{(1:l-1)} \|_F^2|
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | 3. Zero-Pass One-Shot Block Pruning (<10 Seconds on CPU/Single GPU)         |  |
|  |    Prune top-K redundant blocks with highest w_1 R_{\text{intra}} + w_2 R_{\text{inter}}|
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **校准集偏差（Calibration Set Bias）与前向显存开销**：现有的大模型深度/层剪枝方法（如 ShortGPT 的 Block Influence、LaCo、SliceGPT）均依赖在特定校准集（如 WikiText2 或 C4）上运行前向传播以统计输入输出余弦相似度。这不仅在 70B+ 模型上消耗高昂显存与时间，更严重的是层重要性打分高度受制于校准集分布——在通用语料上表现为“弱贡献”的层，往往承载着数学推理或代码生成的关键长尾子空间，剪除后导致严重的领域退化。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **无激活等效残差映射提取**：
   对于第 $l$ 层 Transformer 块，将其对残差流 $h^{(l-1)}$ 的线性主轴作用表征为注意力值-输出合成矩阵 $M _ {\text{attn}}^{(l)} = W _ O^{(l)} W _ V^{(l)} \in \mathbb{R}^{d \times d}$ 与前馈网络合成算子 $M _ {\text{ffn}}^{(l)} = W _ {\text{down}}^{(l)} (W _ {\text{up}}^{(l)} \odot \bar{\sigma} _ {\text{gate}}) \in \mathbb{R}^{d \times d}$ 。
2. **层内有效秩赤字与层间子空间投影重叠度**：
   对合成算子执行奇异值分解 $M^{(l)} = U^{(l)} \Sigma^{(l)} V^{(l)\top}$ ，定义归一化奇异值分布 $p _ i^{(l)} = \frac{\sigma _ i^{(l)}}{\sum _ j \sigma _ j^{(l)}}$ 。层的权重综合冗余度得分 $\mathcal{S} _ {\text{WRP}}(l)$ 由**层内谱坍缩度**与**相对于前序累积子空间的投影冗余度**共同决定：

$$
\mathcal{S} _ {\text{WRP}}(l) = \underbrace{\left( 1 - \frac{\exp\big(-\sum _ {i=1}^d p _ i^{(l)} \log p _ i^{(l)}\big)}{d} \right)} _ {\text{Intra-Layer Spectral Redundancy}} + \lambda \underbrace{\frac{\big\Vert P _ {\text{span}(1:l-1)} U _ {:, 1:r}^{(l)} \big\Vert _ F^2}{r}} _ {\text{Inter-Layer Subspace Overlap}}
$$

   其中 $P _ {\text{span}(1:l-1)}$ 为前 $l-1$ 层输出主奇异子空间的正交投影算子。若第 $l$ 层的输出主奇异方向几乎完全落在前序层已经张成的子空间内（即缺乏新的正交特征扩展），则该层被判定为高度冗余。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* **秒级零样本层裁剪且跨领域泛化更强**：在 **Llama-3-8B/70B**、**Qwen-2.5-14B** 与 **Mistral-7B** 上，WRP 在完全不运行任何前向传播（耗时不足 8 秒）的情况下剪除 **20%–25% 的层**，在 GSM8K 与 HumanEval 等对校准集敏感的生成任务上比 ShortGPT 和 SLEB 高出 **`+3.4%` 至 `+6.1%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与 *Layer Dropping* (TMLR 2025)、*Demystifying When Pruning Works via Representation Hierarchies* (ICML 2026) 及 *Transformer-Geometry* (`arXiv:2609.15975`, EMNLP 2026) 的深度呼应**：
  * WRP 的第二项 $\big\Vert P _ {\text{span}(1:l-1)} U _ {:, 1:r}^{(l)} \big\Vert _ F^2$ 在权重空间精确刻画了我们在 *Transformer-Geometry* 中定义的**平行分量与正交分量之比**——当层权重输出子空间与前序累积子空间高度重合时，该层仅产生平行特征放大而缺乏正交旋转增量！我们可以将 WRP 的纯权重谱重叠指标与单批次激活几何探针结合，作为 `vla-dtr`（VLADrop）的快速层筛选先验。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`compression/layer_geometry_pruning.py` (Weight Spectral Redundancy vs $\Delta h _ \perp$ )  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-19_ai_paper_notes.md`


---

### 3.25 [2026-09-19] REAP: Router-Weighted Expert Activation Pruning for Sparse MoE Models

* **论文信息**：`arXiv:2510.13999` (2025/2026)
* **核心关键词**：MoE Expert Pruning、Router Gate Weighting、Expert Activation Norm、Generative Reasoning Preservation

#### 📐 架构与核心算法流程图 (ASCII Blueprint)

```text
+-----------------------------------------------------------------------------------+
|            REAP: Router-Weighted Expert Activation Pruning Pipeline               |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  Token x_t ---> Router Gate g_{t,e} = Softmax(W_r x_t)_e                          |
|            ---> Active Expert Output E_e(x_t) = W_down (SiLU(W_gate x_t) * W_up x_t)|
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | Joint Multiplicative Saliency Metric (门控权重 x 激活输出范数联合度量)       |  |
|  |    I_{\text{REAP}}(e) = \mathbb{E}_{x_t \in \mathcal{A}_e} [ g_{t,e} \cdot  |  |
|  |                         \| E_e(x_t) \|_2 ] \cdot \hat{P}(e \in \text{Top-}k)|  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|       Prune Lowest-I_{\text{REAP}} Experts ---> Gate Renormalization (Zero-Train) |
+-----------------------------------------------------------------------------------+
```

#### 🎯 背景与痛点 (Background & Pain Points)
* **仅凭路由频率或专家合并（Expert Merging）在生成任务上的失效**：传统 MoE 压缩常根据专家被选中的频次 $\hat{P}(e \in \text{Top-}k)$ 剪枝，或将相似专家权重线性平均（Merging）。作者发现：（1）在代码生成与数学推理等生成任务中，线性合并两个非线性 SwiGLU 专家的权重会破坏内部特征门控对齐，引起特征坍缩；（2）许多高频被选中的专家其输出向量范数 $\Vert E _ e(x _ t)\Vert _ 2$ 极小（充当空操作/恒等缓冲），而真正决定推理跃迁的专家则具有高门控权重乘以高输出激活范数。

#### 💡 核心方法与数学公式 (Core Methodology & Math)
1. **路由器加权激活范数重要性（Router-Weighted Activation Norm）**：
   由于 MoE 层的精确输出增量为 $\Delta h _ t = \sum _ {e \in \text{Top-}k(x _ t)} g _ {t,e} E _ e(x _ t)$ ，单个专家 $e$ 从激活集合中移除时引起的期望一阶残差上界正比于 $g _ {t,e} \Vert E _ e(x _ t)\Vert _ 2$ 。因此 REAP 定义专家 $e$ 的全局重要性为：

$$
\mathcal{I} _ {\text{REAP}}(e) = \frac{1}{|\mathcal{D} _ {\text{cal}}|} \sum _ {t=1}^{|\mathcal{D} _ {\text{cal}}|} \mathbb{I}\big(e \in \text{Top-}k(x _ t)\big) \cdot g _ {t,e} \cdot \big\Vert E _ e(x _ t) \big\Vert _ 2
$$

2. **保留集门控重归一化（Post-Pruning Gate Renormalization）**：
   裁剪掉得分最低的专家集合 $\mathcal{E} _ {\text{prune}}$ 后，对剩余专家集合 $\mathcal{E} _ {\text{keep}}$ 的门控权重执行保和重归一化 $\tilde{g} _ {t,e} = \frac{g _ {t,e}}{\sum _ {j \in \text{Top-}k(x _ t) \cap \mathcal{E} _ {\text{keep}}} g _ {t,j}}$ ，以补偿被移除专家的幅度损失。

#### 📊 关键实验与结论 (Key Experiments & Takeaways)
* 在 **Mixtral-8x7B**、**DeepSeek-MoE-16B** 与 **Qwen1.5-MoE-A2.7B** 上，REAP 在 **25%–37.5% 专家剪枝率**下，在 GSM8K 与 HumanEval 生成基准上大幅超越各类专家合并算法（HC-SMoE、M-SMoE）达 **`+8.5%` 至 `+14.2%`**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发
* **与 *Capacity-Aware Inference* (ICLR 2026) & *Transformer-Geometry* (EMNLP 2026) 的结合**：
  * REAP 揭示了 $\Vert g _ {t,e} E _ e(x _ t)\Vert _ 2$ 相比单纯门控概率 $g _ {t,e}$ 的优越性。结合我们的 *Transformer-Geometry*，我们可以进一步将 $\Vert E _ e(x _ t)\Vert _ 2$ 替换为正交切向范数 $\Vert P _ \perp(h _ t) E _ e(x _ t)\Vert _ 2$ ，避免那些仅沿当前残差方向做无效径向放大的专家占据高分。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/` & `attention_xsa/` (`arXiv:2609.15975`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-19_ai_paper_notes.md`


---

### 3.26 [2026-09-18] ✂️ *AnchorPrune: Geometry-Preserving Representation Hierarchy Compression for Multimodal Large Language Models*
> **聚焦领域**：Multimodal Sparsity · Representation Hierarchies · Layer Dropping · Geometric Manifolds  
> **arXiv**：[`arXiv:2609.08842`](https://arxiv.org/abs/2609.08842)

```
  多模态隐状态流形 ──► [ 1. 局部几何锚点提取 (Anchor SVD) ] ──► 计算流形重构失真率 D_l
                                     │                                      │
                                     ▼                                      ▼
                      [ 2. 层级表征阶梯贡献判定 ]             [ 3. 联合压缩: 40% 层丢弃 + 50% Token 稀疏 ]
                      判为冗余饱和层 ──► 予以跳过               零微调保留 99.2% MMBench 精度
```

#### 🎯 背景与痛点 (Problem Statement)
多模态大模型在深层网络中存在极高比例的视觉表征冗余。现有的 Token 剪枝与 Layer Dropping 往往割裂进行：若先剪 Token 再丢层，会导致跨模态语义对齐发生断崖式崩塌；若仅做静态层丢弃，浅层大量的背景无用 Token 依然占据巨大的显存与 Attention 算力。

#### 💡 核心方法与原文底层数学实现 (Mathematical Formulations)
1. **多模态局部几何锚点矩阵 (Multimodal Geometric Anchors)**：
   - 在第 $l$ 层提取多模态激活流形 $\mathcal{M} _ l$ 上的代表性锚点子集 $\mathcal{A} _ l = \lbrace a _ 1, a _ 2, \dots, a _ K\rbrace \subset \mathbb{R}^{d}$ ；
   - 求解局部切空间的主成分基底，定义层级几何表征流形失真度指标 $\mathcal{D} _ l$ ：

$$
\mathcal{D} _ l \triangleq \frac{1}{K} \sum _ {k=1}^K \left\lVert a _ k - \Pi _ {\mathcal{A} _ {l-1}}(a _ k) \right\rVert _ 2^2
$$

   - 当 $\mathcal{D} _ l < \tau _ {\text{layer}}$ 时，判定该层为表征阶梯中的平坦饱和层，可安全丢弃。
2. **锚点引导的动态 Token 稀疏过滤 (Anchor-Guided Token Sparsification)**：
   - 仅保留与核心几何锚点内积相似度大于动态阈值的 Token，在浅层过滤掉 50% 以上的无用背景 Patch，同时维持深层关键语义边界。

#### 📊 关键实验与结论 (Experiments & Findings)
* **评估模型**：Qwen2-VL-7B/72B、LLaVA-NeXT-34B；
* **压缩指标**：联合跳过 **40% Transformer 层** 并剔除 **50% 视觉 Token**，无需微调，在 MME、MMBench、ChartQA 上平均精度损失仅 **0.8%**，端到端推理提速 **2.7 倍**，显存峰值降低 **62%**。

#### 🔗 与我们工作（Our Works）的直接关联与落地启发 (Relevance & Synergy with Our Works)
* **🎯 锚定代表作**：
  * [Paper #15: *Demystifying When Pruning Works via Representation Hierarchies* (ICML 2026)]
  * [Paper #8: *Understanding and Harnessing Sparsity for Unified Multimodal Models* (TMLR 2026)]
  * [Paper #9: *Uncovering the Redundancy in Transformers via Layer Dropping* (TMLR 2025)]
* **🔬 机理对比与技术演进**：
  * 我们在 *ICML 26* 与 *TMLR 25* 中奠定了从“表征层级阶梯（Representation Hierarchies）”解释剪枝机理的理论基石；
  * *AnchorPrune* 将我们的层级冗余理论推进到了“层丢弃（Layer Dropping）与 Token 动态稀疏（Token Sparsity）的二维联合优化”，提供了具体的几何锚点判据；
* **💡 下一阶段研究（Next Research Directions）落地启发**：
  * 可直接将锚点流形失真度 $\mathcal{D} _ l$ 集成至我们的多模态轻量化评估脚本中，作为我们后续多模态稀疏化大模型训练的正则化损失函数。

---

> [!TIP]
> **🎯 `transformer-geometry` 仓库代码级落地点 (`Target Module`)**：`probing/` & `attention_xsa/` (`arXiv:2609.15975`)  
> **📚 上游精读归档 (`Upstream Source`)**：`scholar-odyssey/intelligence/papers/2026-09-18_ai_paper_notes.md`


---
