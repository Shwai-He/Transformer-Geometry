# XSA paper-aligned 0.7B setting.
# Source: arXiv:2603.09078, Table 1 and Sec. 4.1.

wandb_log = False
dataset = 'fineweb100bt'

# XSA: context length 2048, global batch size 256 sequences = 0.5M tokens.
# In this nanoGPT launcher, tokens/iter = batch_size * block_size * gradient_accumulation_steps.
batch_size = 8
block_size = 2048
gradient_accumulation_steps = 32

n_layer = 24
n_head = 6
n_embd = 1536
learning_rate = 5e-4
min_lr = 5e-5

# Rule-based token budget schedule (robust to bsz/ctx/grad_accum changes).
target_tokens = 100_000_000_000  # 100B
tokens_per_iter = batch_size * block_size * gradient_accumulation_steps
max_iters = (target_tokens + tokens_per_iter - 1) // tokens_per_iter
lr_decay_iters = max_iters
warmup_iters = 2000

eval_interval = 1000
eval_iters = 200
log_interval = 10
weight_decay = 1e-1
