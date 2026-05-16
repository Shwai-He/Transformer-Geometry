# Representation Saturation

_Converted from `Representation Saturation.docx`. Figures and drawings are omitted._

## 1. Background and Motivation

Modern Transformer architectures adopt the pre-norm residual Transformer structure:

When:

then:

We refer to this phenomenon as representation saturation, where consecutive layers produce highly similar representations due to the small residual updates.

Deeper layers often exhibit excessively high layer-wise cosine similarity, which may indicate limited transformations in the latent space.

Qwen-2.5-VL

Qwen3-4B-Instruct-2507Qwen3-Next-80B-A3B-Instruct

While this structure is highly effective for generative modeling, recent empirical observations suggest a potential mismatch when it is applied to embedding or metric learning tasks.

For discrete generation tasks, high layer-wise similarity may not necessarily be harmful. This is largely due to the translational invariance of the softmax readout.

Because the softmax function is sensitive to relative logit differences, even small perturbations in the hidden representation hhh can significantly alter the resulting token probabilities. As a result, small residual updates can still produce meaningful functional changes in the output distribution.

However, the situation can be different for continuous embedding spaces. Multiple empirical studies in perception and retrieval tasks have observed that the most informative representations often emerge at intermediate layers rather than at the final layer. For example, perception encoders frequently show that the most discriminative visual embeddings do not appear at the network output. Similarly, recent work such as Making Large Language Models Efficient Dense Retrievers reports that intermediate representations can outperform final-layer embeddings for retrieval tasks.

Perception Encoder: The best visual embeddings are not at the output of the network

Making Large Language Models Efficient Dense Retrievers

These observations raise an important question: if deeper layers exhibit excessively high representation similarity, could this lead to representation saturation? More broadly, do pre-norm Transformers exhibit premature representation saturation when used for metric learning, and if so, how should model architectures be adapted to address this issue?

## 2. Preliminaries

Update Decomposition for Pre-Norm Transformers

t is often very small, since the residual branch typically has a much larger norm.

As a result, the updates introduced at each layer remain minimal, leading to very high cosine similarity between consecutive layer representations.

## 3. Methods to mitigate high similarities

To alleviate the representation saturation caused by minimal residual updates, we explore several architectural modifications that encourage stronger transformations across layers.

Residual Scale Adjustment

One approach is to adjust the scaling of the residual branch. Specifically, we introduce a learnable or depth-dependent scaling factor

Layer-wise Normalization

Another strategy is to modify the behavior of LayerNorm across depth. Instead of using a constant gain parameter, we allow the LayerNorm scaling parameter γl\gamma_lγl​ to increase with depth:

This design restores scale flexibility in higher layers, enabling deeper layers to apply stronger transformations and potentially improving representation diversity.

Hybrid Norm Placement

We also investigate modifying the placement of normalization within the Transformer block by combining pre-norm and post-norm structures:

Pre-norm in early layers, which maintains training stability.

Post-norm or semi-post-norm in deeper layers, which enables stronger representation reshaping.

## 4. Main Experiments

LLMs:

embedding models:

Training the residual scale on a well-trained model may not be useful.

generative SFT:

the gap is minimal,

chain-of-thoughts:

latent-reasoning:

train the compressed model:

With the same random seed, the sample at each step remains identical.

NanoGPT:

tiny version of pretraining.

Embedding training on randomly initialized models:

Vison:

vision encoder:

VLMs:

understanding & generation:

focus on understanding components.

Latent cosine similarity may not be lowered:

we could think about other ways to explicitly lower it.

## 5. Additional Results

Training: SFT models tend to have high angular distance.

Exclusive Self Attention：

Instead of removing it, take it as the optimization objective.
