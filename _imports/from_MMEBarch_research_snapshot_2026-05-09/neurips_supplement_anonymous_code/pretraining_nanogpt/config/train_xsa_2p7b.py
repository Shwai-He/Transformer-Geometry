# XSA paper-aligned 2.7B setting.
# Paper: n_layer=32, d_model=2560, n_heads=24, d_head=128.

wandb_log = False
dataset = 'fineweb100bt'

batch_size = 8
block_size = 2048
gradient_accumulation_steps = 32

n_layer = 32
n_head = 20
n_embd = 2560
learning_rate = 3e-4
min_lr = 3e-5

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
