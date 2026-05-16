"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import json
import pickle
import re
import random
import threading
import queue
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from model import GPTConfig, GPT
from residual_patch import apply_qwen3_forward_patch

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'auto' # 'auto' or 'scratch' or 'resume' or 'gpt2*'
save_latest_checkpoint = True
save_best_checkpoint = True
checkpoint_interval = 10000 # save ckpt_iter_<step>.pt every N iters; <=0 disables periodic snapshots
checkpoint_every_n_evals = 1 # also save ckpt_iter_<step>.pt every N evals; <=0 disables eval-based snapshots
checkpoint_keep_limit = 0 # number of periodic ckpt_iter_<step>.pt snapshots to keep; <=0 keeps all
best_checkpoint_min_interval_evals = 5 # require at least this many eval intervals between best checkpoint writes; <=0 disables throttling
resume_require_hparam_match = True
resume_hparam_mismatch_fallback_to_scratch = True
checkpoint_interval_min = 100 # when periodic-by-iter is enabled, enforce at least this many iterations
checkpoint_periodic_start_iter = 100 # do not write ckpt_iter_<step>.pt before this step
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
wandb_log_layerwise = True
wandb_log_xsa_per_layer = True
wandb_log_xsa_means = True
wandb_log_aux_losses = False
# residual patch
enable_residual_patch = False
residual_mode = 'sum' # sum | param | wx
residual_attn_param = 1.0
residual_mlp_param = 1.0
residual_wx = 1.0
residual_wx_activation = 'silu' # silu | exp | sigmoid2
residual_wx_learnable = False
residual_branch_gate = False
residual_branch_wx = 1.0
residual_branch_wx_learnable = False
residual_branch_scale_init = 1.0
residual_branch_scale_learnable = False
residual_scale_learnable = False
residual_scale_init = 1.0
residual_scale_depth_slope = 0.0
residual_lr_ratio = 1.0
use_post_norm = False
post_norm_start_layer = 1000000000
# XSA-aligned soft regularization (no inference path modification)
enable_xsa_self_loss = False
xsa_self_loss_lambda = 0.0
xsa_self_loss_warmup_iters = 0
xsa_self_loss_start_layer = 0
xsa_self_loss_stride = 1
xsa_self_loss_objective = 'ratio_detach' # ratio | ratio_detach | parallel
xsa_forward_only = False
xsa_forward_target = 'attn' # attn | both | mlp
xsa_forward_ref = 'self_value' # residual | self_value
xsa_forward_space = 'pre_o_proj' # post_o_proj | pre_o_proj
xsa_forward_op = 'remove_parallel' # remove_parallel | keep_parallel | add_parallel | negate_parallel
xsa_forward_alpha = 1.0
xsa_forward_subtract_scale = 1.0
xsa_forward_learnable_gate = False
xsa_forward_gate_init = 1.0
xsa_forward_gate_mode = 'static' # static | token
xsa_forward_learnable_alpha = False
xsa_forward_alpha_min = 0.0
xsa_forward_alpha_max = 1.0
xsa_forward_learnable_gamma = False
xsa_forward_gamma_per_head = False
xsa_forward_gamma_init = 1.0
xsa_forward_start_layer = 0
xsa_forward_end_layer = -1 # inclusive; -1 means last layer
xsa_forward_stride = 1
xsa_metrics_interval = 0 # 0 => use log_interval, <0 => disable logging-only XSA metrics
task_loss_weight = 1.0
xsa_debug_fixed_batch = False
enable_attnres = False
attnres_eps = 1e-6
# Low-cost diagonal attention suppression: none | hard | bias.
attn_diag_mode = 'none'
attn_diag_bias = 0.0
attn_diag_keep_first = True
use_rope = True
rope_base = 10000.0
embedding_layernorm = True
stop_on_nonfinite = True
deterministic_train = False
activation_checkpointing = False
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_head_dim = 0 # 0 => infer from n_embd // n_head (legacy); >0 enables decoupled head dim
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
compile_debug_recompiles = False
compile_debug_graph_breaks = False
async_batch_prefetch = True
async_batch_prefetch_depth = 2
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Guardrail: avoid excessively frequent periodic checkpoint writes.
if int(checkpoint_interval) > 0 and int(checkpoint_interval) < int(checkpoint_interval_min):
    print(
        f"[WARN] checkpoint_interval={int(checkpoint_interval)} is too small; "
        f"clamped to {int(checkpoint_interval_min)}",
        flush=True,
    )
    checkpoint_interval = int(checkpoint_interval_min)
    config["checkpoint_interval"] = int(checkpoint_interval)

if int(checkpoint_periodic_start_iter) < 0:
    checkpoint_periodic_start_iter = 0
    config["checkpoint_periodic_start_iter"] = 0

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
if master_process:
    print(
        f"[INFO] async_batch_prefetch={bool(async_batch_prefetch)} "
        f"prefetch_depth={int(async_batch_prefetch_depth)}",
        flush=True,
    )

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
np.random.seed(seed + seed_offset)
random.seed(seed + seed_offset)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed + seed_offset)
if bool(deterministic_train):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
if master_process:
    print(
        f"[INFO] seed_base={seed} seed_offset={seed_offset} seed_effective={seed + seed_offset} deterministic_train={bool(deterministic_train)}",
        flush=True,
    )
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Use dedicated RNGs for batch index sampling so data order stays aligned
# across baseline/wx/xsa runs even if model init/patching consumes global RNG.
batch_index_generators = {
    "train": torch.Generator(device="cpu"),
    "val": torch.Generator(device="cpu"),
}
batch_index_generators["train"].manual_seed(seed + seed_offset)
batch_index_generators["val"].manual_seed(seed + seed_offset + 1000003)

# poor man's data loader
data_dir = os.path.join('data', dataset)
_batch_data_cache = {}
_batch_offsets_cache = {}
_batch_tensor_cache = {}
_batch_prefetchers = {}


def _get_split_data(split):
    data = _batch_data_cache.get(split)
    if data is not None:
        return data
    filename = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, filename), dtype=np.uint16, mode='r')
    _batch_data_cache[split] = data
    return data


def _get_batch_offsets():
    offsets = _batch_offsets_cache.get(block_size)
    if offsets is not None:
        return offsets
    offsets = np.arange(block_size, dtype=np.int64)
    _batch_offsets_cache[block_size] = offsets
    return offsets


def _get_batch_buffers(split):
    key = (split, batch_size, block_size, device_type)
    entry = _batch_tensor_cache.get(key)
    if entry is not None:
        return entry
    pin = device_type == 'cuda'
    buffers = []
    for _ in range(2):
        x_buf = torch.empty((batch_size, block_size), dtype=torch.int64, pin_memory=pin)
        y_buf = torch.empty((batch_size, block_size), dtype=torch.int64, pin_memory=pin)
        buffers.append((x_buf, y_buf))
    entry = {"buffers": buffers, "next_idx": 0}
    _batch_tensor_cache[key] = entry
    return entry


def _build_batch_cpu(split):
    data = _get_split_data(split)
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,),
        generator=batch_index_generators[split],
    )
    starts = ix.numpy().astype(np.int64, copy=False)
    offsets = _get_batch_offsets()
    x_idx = starts[:, None] + offsets[None, :]
    y_idx = x_idx + 1
    x_np = np.asarray(data[x_idx], dtype=np.int64)
    y_np = np.asarray(data[y_idx], dtype=np.int64)
    if bool(async_batch_prefetch):
        pin = device_type == 'cuda'
        x_cpu = torch.empty((batch_size, block_size), dtype=torch.int64, pin_memory=pin)
        y_cpu = torch.empty((batch_size, block_size), dtype=torch.int64, pin_memory=pin)
    else:
        buffer_entry = _get_batch_buffers(split)
        buf_idx = int(buffer_entry["next_idx"])
        x_cpu, y_cpu = buffer_entry["buffers"][buf_idx]
        buffer_entry["next_idx"] = 1 - buf_idx
    x_cpu.copy_(torch.from_numpy(x_np), non_blocking=False)
    y_cpu.copy_(torch.from_numpy(y_np), non_blocking=False)
    return x_cpu, y_cpu


class _AsyncBatchPrefetcher:
    def __init__(self, split, depth):
        self.split = split
        self.depth = max(1, int(depth))
        self.ready = queue.Queue(maxsize=self.depth)
        self.stop_event = threading.Event()
        self.error = None
        self.worker = threading.Thread(
            target=self._worker_loop,
            name=f"batch-prefetch-{split}",
            daemon=True,
        )
        self.worker.start()

    def _worker_loop(self):
        try:
            while not self.stop_event.is_set():
                batch = _build_batch_cpu(self.split)
                while not self.stop_event.is_set():
                    try:
                        self.ready.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            self.error = exc
            self.stop_event.set()
            try:
                self.ready.put_nowait(None)
            except queue.Full:
                pass

    def next(self):
        while True:
            if self.error is not None:
                raise RuntimeError(f"async batch prefetch failed for split={self.split}") from self.error
            try:
                batch = self.ready.get(timeout=0.1)
                if batch is None:
                    if self.error is not None:
                        raise RuntimeError(f"async batch prefetch failed for split={self.split}") from self.error
                    continue
                return batch
            except queue.Empty:
                if self.error is not None:
                    raise RuntimeError(f"async batch prefetch failed for split={self.split}") from self.error

    def close(self):
        self.stop_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)


def _get_prefetcher(split):
    prefetcher = _batch_prefetchers.get(split)
    if prefetcher is not None:
        return prefetcher
    prefetcher = _AsyncBatchPrefetcher(split, async_batch_prefetch_depth)
    _batch_prefetchers[split] = prefetcher
    return prefetcher


def get_batch(split):
    if bool(async_batch_prefetch):
        x_cpu, y_cpu = _get_prefetcher(split).next()
    else:
        x_cpu, y_cpu = _build_batch_cpu(split)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
    else:
        x, y = x_cpu, y_cpu
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
tokens_seen = 0
eval_save_counter = 0
_wandb_resume_id = None  # set from checkpoint on resume, used to continue the same W&B run

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
out_dir_abs = os.path.abspath(out_dir)

_periodic_ckpt_pattern = re.compile(r"^ckpt_iter_(\d+)\.pt$")

def _list_checkpoint_candidates():
    candidate_names = ['ckpt.pt', 'ckpt_best.pt']
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if _periodic_ckpt_pattern.match(name):
                candidate_names.append(name)
    candidates = []
    seen = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            if size > 0:
                candidates.append({
                    'name': name,
                    'path': path,
                    'abs_path': os.path.abspath(path),
                    'size': size,
                    'mtime': os.path.getmtime(path),
                })
    return candidates

def _select_latest_checkpoint():
    candidates = _list_checkpoint_candidates()
    if not candidates:
        return None, candidates
    latest = max(candidates, key=lambda item: (item['mtime'], item['name']))
    return latest, candidates

latest_ckpt, ckpt_candidates = _select_latest_checkpoint()
ckpt_path_resolved = latest_ckpt['path'] if latest_ckpt is not None else ckpt_path
ckpt_path_abs = os.path.abspath(ckpt_path_resolved)
ckpt_exists = latest_ckpt is not None
ckpt_size = latest_ckpt['size'] if latest_ckpt is not None else -1
print(
    "[INFO] init_from_check "
    f"cwd={os.getcwd()} "
    f"out_dir={out_dir} "
    f"out_dir_abs={out_dir_abs} "
    f"ckpt_path={ckpt_path_abs} "
    f"ckpt_exists={ckpt_exists} "
    f"ckpt_size={ckpt_size} "
    f"ckpt_candidates={len(ckpt_candidates)} "
    f"init_from={init_from}",
    flush=True,
)
if latest_ckpt is not None:
    print(
        f"[INFO] latest_checkpoint_selected name={latest_ckpt['name']} "
        f"path={latest_ckpt['abs_path']} size={latest_ckpt['size']}",
        flush=True,
    )
resolved_init_from = init_from
if init_from == 'auto':
    if ckpt_exists and ckpt_size > 0:
        resolved_init_from = 'resume'
        print(f"[INFO] Found checkpoint at {ckpt_path_abs}, resuming training.", flush=True)
    else:
        resolved_init_from = 'scratch'
        print(f"[INFO] No checkpoint found at {ckpt_path_abs}, starting from scratch.", flush=True)
elif init_from == 'resume':
    if not (ckpt_exists and ckpt_size > 0):
        raise FileNotFoundError(
            f"init_from=resume but checkpoint is missing or empty: "
            f"ckpt_path={ckpt_path_abs} exists={ckpt_exists} size={ckpt_size}"
        )
    print(f"[INFO] init_from=resume and checkpoint found at {ckpt_path_abs}, resuming training.", flush=True)
elif init_from == 'scratch':
    if ckpt_exists and ckpt_size > 0:
        print(
            f"[WARN] init_from=scratch while checkpoint exists at {ckpt_path_abs}; "
            "existing checkpoint will be ignored and training will restart from scratch.",
            flush=True,
        )
    else:
        print(f"[INFO] init_from=scratch and no checkpoint found at {ckpt_path_abs}.", flush=True)
print(f"[INFO] resolved_init_from={resolved_init_from}", flush=True)

def _cleanup_old_periodic_checkpoints():
    if checkpoint_keep_limit <= 0:
        return
    periodic = []
    for name in os.listdir(out_dir):
        match = _periodic_ckpt_pattern.match(name)
        if match:
            periodic.append((int(match.group(1)), os.path.join(out_dir, name)))
    if len(periodic) <= checkpoint_keep_limit:
        return
    periodic.sort()
    for _, path in periodic[:-checkpoint_keep_limit]:
        try:
            os.remove(path)
            print(f"[INFO] removed old periodic checkpoint {path}", flush=True)
        except FileNotFoundError:
            pass


def _checkpoint_sidecar_payload(checkpoint_name, iter_num, is_best):
    return {
        "checkpoint_name": checkpoint_name,
        "run_name": os.path.basename(os.path.abspath(out_dir)),
        "out_dir": os.path.abspath(out_dir),
        "iter_num": int(iter_num),
        "tokens_seen": int(tokens_seen),
        "is_best": bool(is_best),
        "saved_at_unix": int(time.time()),
        "model": {
            "n_layer": int(n_layer),
            "n_head": int(n_head),
            "n_embd": int(n_embd),
            "block_size": int(block_size),
            "vocab_size": int(model_args.get("vocab_size", 0) or 0),
        },
        "schedule": {
            "learning_rate": float(learning_rate),
            "min_lr": float(min_lr),
            "warmup_iters": int(warmup_iters),
            "lr_decay_iters": int(lr_decay_iters),
            "max_iters": int(max_iters),
            "decay_lr": bool(decay_lr),
        },
        "batching": {
            "batch_size": int(batch_size),
            "gradient_accumulation_steps_local": int(gradient_accumulation_steps),
            "world_size": int(ddp_world_size),
            "global_batch_sequences": int(batch_size * gradient_accumulation_steps * ddp_world_size),
            "tokens_per_iter": int(tokens_per_iter),
        },
        "run": {
            "dataset": str(dataset),
            "seed": int(seed),
            "wandb_run_name": str(wandb_run_name),
        },
    }


def _write_checkpoint_sidecar(checkpoint_path, iter_num, is_best):
    sidecar_path = checkpoint_path + ".json"
    payload = _checkpoint_sidecar_payload(
        checkpoint_name=os.path.basename(checkpoint_path),
        iter_num=iter_num,
        is_best=is_best,
    )
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[INFO] saved checkpoint metadata to {sidecar_path}", flush=True)


def _normalize_hparam_value(v):
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    if isinstance(v, str):
        return str(v)
    return v


def _resume_hparam_keys():
    return [
        'dataset',
        'batch_size',
        'block_size',
        'gradient_accumulation_steps',
        'learning_rate',
        'min_lr',
        'warmup_iters',
        'lr_decay_iters',
        'max_iters',
        'decay_lr',
        'weight_decay',
        'beta1',
        'beta2',
        'grad_clip',
        'n_layer',
        'n_head',
        'n_head_dim',
        'n_embd',
        'dropout',
        'bias',
        'use_rope',
        'rope_base',
        'embedding_layernorm',
        'attn_diag_mode',
        'attn_diag_bias',
        'attn_diag_keep_first',
        'enable_attnres',
        'attnres_eps',
        'enable_residual_patch',
        'residual_mode',
        'residual_attn_param',
        'residual_mlp_param',
        'residual_wx',
        'residual_wx_activation',
        'residual_wx_learnable',
        'residual_branch_gate',
        'residual_branch_wx',
        'residual_branch_wx_learnable',
        'residual_branch_scale_init',
        'residual_branch_scale_learnable',
        'residual_scale_learnable',
        'residual_scale_init',
        'residual_scale_depth_slope',
        'residual_lr_ratio',
        'use_post_norm',
        'post_norm_start_layer',
        'enable_xsa_self_loss',
        'xsa_self_loss_lambda',
        'xsa_self_loss_warmup_iters',
        'xsa_forward_only',
        'xsa_forward_target',
        'xsa_forward_ref',
        'xsa_forward_space',
        'xsa_forward_op',
        'xsa_forward_alpha',
        'xsa_forward_subtract_scale',
        'xsa_forward_learnable_gate',
        'xsa_forward_gate_init',
        'xsa_forward_gate_mode',
        'xsa_forward_learnable_alpha',
        'xsa_forward_alpha_min',
        'xsa_forward_alpha_max',
        'xsa_forward_learnable_gamma',
        'xsa_forward_gamma_per_head',
        'xsa_forward_gamma_init',
        'xsa_forward_start_layer',
        'xsa_forward_end_layer',
        'xsa_forward_stride',
    ]


def _check_resume_hparam_match(checkpoint_config, current_config):
    mismatches = []
    for key in _resume_hparam_keys():
        if key not in checkpoint_config or key not in current_config:
            continue
        ckpt_v = _normalize_hparam_value(checkpoint_config[key])
        curr_v = _normalize_hparam_value(current_config[key])
        if ckpt_v != curr_v:
            mismatches.append((key, ckpt_v, curr_v))
    return mismatches

def save_training_checkpoint(checkpoint, iter_num, is_best, save_periodic=False):
    if save_latest_checkpoint:
        latest_path = os.path.join(out_dir, 'ckpt.pt')
        torch.save(checkpoint, latest_path)
        print(f"[INFO] saved latest checkpoint to {latest_path}", flush=True)
        _write_checkpoint_sidecar(latest_path, iter_num, is_best=False)
    if is_best and save_best_checkpoint:
        best_path = os.path.join(out_dir, 'ckpt_best.pt')
        torch.save(checkpoint, best_path)
        print(f"[INFO] saved best checkpoint to {best_path}", flush=True)
        _write_checkpoint_sidecar(best_path, iter_num, is_best=True)
    periodic_by_iter = checkpoint_interval > 0 and iter_num > 0 and iter_num % checkpoint_interval == 0
    periodic_allowed = int(iter_num) >= int(checkpoint_periodic_start_iter)
    if (save_periodic or periodic_by_iter) and periodic_allowed:
        periodic_path = os.path.join(out_dir, f'ckpt_iter_{iter_num:07d}.pt')
        torch.save(checkpoint, periodic_path)
        print(f"[INFO] saved periodic checkpoint to {periodic_path}", flush=True)
        _write_checkpoint_sidecar(periodic_path, iter_num, is_best=False)
        _cleanup_old_periodic_checkpoints()
    elif (save_periodic or periodic_by_iter) and (not periodic_allowed):
        print(
            f"[INFO] skip periodic checkpoint at iter={int(iter_num)} "
            f"(checkpoint_periodic_start_iter={int(checkpoint_periodic_start_iter)})",
            flush=True,
        )


last_best_checkpoint_iter = None

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  n_head_dim=n_head_dim,
                  bias=bias, vocab_size=None, dropout=dropout,
                  activation_checkpointing=activation_checkpointing,
                  enable_attnres=enable_attnres, attnres_eps=attnres_eps,
                  attn_diag_mode=attn_diag_mode, attn_diag_bias=attn_diag_bias,
                  attn_diag_keep_first=attn_diag_keep_first,
                  xsa_forward_learnable_gate=xsa_forward_learnable_gate,
                  xsa_forward_learnable_alpha=xsa_forward_learnable_alpha,
                  xsa_forward_learnable_gamma=xsa_forward_learnable_gamma,
                  use_rope=use_rope, rope_base=rope_base,
                  embedding_layernorm=embedding_layernorm) # start with model_args from command line
if resolved_init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif resolved_init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    checkpoint = torch.load(ckpt_path_resolved, map_location=device)
    checkpoint_config = checkpoint.get('config', None)
    resume_fallback_to_scratch = False
    if bool(resume_require_hparam_match):
        if not isinstance(checkpoint_config, dict):
            raise RuntimeError(
                "resume_require_hparam_match=True but checkpoint has no saved config; "
                "cannot verify hyperparameter compatibility."
            )
        hparam_mismatches = _check_resume_hparam_match(checkpoint_config, config)
        if hparam_mismatches:
            mismatch_lines = [
                f"{key}: checkpoint={ckpt_v} current={curr_v}"
                for key, ckpt_v, curr_v in hparam_mismatches
            ]
            if bool(resume_hparam_mismatch_fallback_to_scratch):
                print(
                    "[WARN] Resume checkpoint hparams mismatch; starting a NEW run from scratch instead:\n  "
                    + "\n  ".join(mismatch_lines),
                    flush=True,
                )
                resume_fallback_to_scratch = True
            else:
                raise RuntimeError(
                    "Refusing to resume because critical hyperparameters do not match:\n  "
                    + "\n  ".join(mismatch_lines)
                )
        if not resume_fallback_to_scratch:
            print("[INFO] resume hparam check passed.", flush=True)
    if resume_fallback_to_scratch:
        resolved_init_from = 'scratch'
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        iter_num = 0
        best_val_loss = 1e9
        tokens_seen = 0
        eval_save_counter = 0
        _wandb_resume_id = None
    else:
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        resume_model_arg_keys = [
            'n_layer',
            'n_head',
            'n_head_dim',
            'n_embd',
            'block_size',
            'bias',
            'vocab_size',
            'use_rope',
            'embedding_layernorm',
            'rope_base',
        ]
        missing_resume_model_arg_keys = []
        for k in resume_model_arg_keys:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
            else:
                missing_resume_model_arg_keys.append(k)
        if missing_resume_model_arg_keys:
            print(
                "[INFO] resume checkpoint missing model_args keys; "
                "using current defaults for: "
                + ", ".join(missing_resume_model_arg_keys),
                flush=True,
            )
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing_suffixes = (
            '.attn.bias',
            '.attn.attn_additive_mask',
            '.xsa_forward_alpha_raw',
            '.xsa_forward_gamma_raw',
            '.xsa_forward_gate_raw',
            '.xsa_forward_gate_proj.weight',
            '.xsa_forward_gate_proj.bias',
        )
        unexpected = [
            key for key in incompatible.unexpected_keys if not key.endswith(allowed_missing_suffixes)
        ]
        disallowed_missing = [
            key for key in incompatible.missing_keys if not key.endswith(allowed_missing_suffixes)
        ]
        if disallowed_missing or unexpected:
            raise RuntimeError(
                "Failed to resume checkpoint cleanly: "
                f"missing={disallowed_missing} unexpected={unexpected}"
            )
        if incompatible.missing_keys:
            print(
                "[INFO] resume tolerated missing non-parameter attention buffers: "
                + ", ".join(incompatible.missing_keys),
                flush=True,
            )
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        tokens_seen = int(checkpoint.get('tokens_seen', int(iter_num) * int(tokens_per_iter)))
        eval_save_counter = int(checkpoint.get('eval_save_counter', 0))
        _wandb_resume_id = checkpoint.get('wandb_id', None)
        print(
            f"[INFO] resume state loaded: iter_num={int(iter_num)} "
            f"tokens_seen={int(tokens_seen)} "
            f"best_val_loss={float(best_val_loss):.6f} "
            f"wandb_id={_wandb_resume_id}",
            flush=True,
        )
elif resolved_init_from.startswith('gpt2'):
    print(f"Initializing from GPT-2 weights: {resolved_init_from}")
    # initialize from GPT-2 weights
    override_args = dict(dropout=dropout, use_rope=use_rope, embedding_layernorm=embedding_layernorm)
    model = GPT.from_pretrained(resolved_init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_head_dim', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"Unsupported init_from={init_from} (resolved={resolved_init_from})")
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
# nanoGPT uses n_embd naming; expose hidden_size for shared residual patch compatibility.
if not hasattr(model.config, 'hidden_size'):
    model.config.hidden_size = model.config.n_embd
if enable_residual_patch:
    patch_cfg = type("ResidualPatchConfig", (), config)()
    patched = apply_qwen3_forward_patch(model, patch_cfg)
    if not patched:
        raise RuntimeError("Residual patch enabled but no compatible decoder layers were patched.")
    print(f"[INFO] Residual patch enabled (mode={residual_mode}).")
    # Re-seed after patching so dropout masks stay aligned with the baseline run.
    # (residual_patch_core already saves/restores RNG inside _set_wx_linear, but
    # other patch steps such as register_forward_hook may also advance state.)
    torch.manual_seed(seed + seed_offset)
    np.random.seed(seed + seed_offset)
    random.seed(seed + seed_offset)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + seed_offset)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay,
    learning_rate,
    (beta1, beta2),
    device_type,
    residual_lr_ratio=residual_lr_ratio,
)
if resolved_init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile and (bool(compile_debug_recompiles) or bool(compile_debug_graph_breaks)):
    try:
        import torch._logging as torch_logging
        log_kwargs = {}
        if bool(compile_debug_recompiles):
            log_kwargs["recompiles"] = True
        if bool(compile_debug_graph_breaks):
            log_kwargs["graph_breaks"] = True
        torch_logging.set_logs(**log_kwargs)
        if master_process:
            print(
                f"[INFO] torch.compile diagnostics enabled: "
                f"recompiles={bool(compile_debug_recompiles)} "
                f"graph_breaks={bool(compile_debug_graph_breaks)}",
                flush=True,
            )
    except Exception as exc:
        if master_process:
            print(f"[WARN] Failed to enable torch.compile diagnostics: {exc}", flush=True)
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
residual_stats_model = unoptimized_model if compile else model


@torch.no_grad()
def collect_residual_param_stats(module):
    attn_means = []
    mlp_means = []
    attn_branch_means = []
    mlp_branch_means = []
    layer_attn = {}
    layer_mlp = {}
    layer_attn_branch = {}
    layer_mlp_branch = {}

    layers = []
    if hasattr(module, "transformer") and hasattr(module.transformer, "h"):
        layers = list(module.transformer.h)

    for lid, layer in enumerate(layers):
        idx = int(getattr(layer, "_residual_layer_idx", lid))

        attn_alpha = None
        if hasattr(layer, "residual_attn_scale"):
            t = layer.residual_attn_scale
            attn_alpha = t.detach().float().mean()
        elif hasattr(layer, "residual_attn_wx"):
            wx = layer.residual_attn_wx
            if hasattr(wx, "bias") and wx.bias is not None:
                b = wx.bias.detach().float()
                wx_act = str(getattr(layer, "_residual_wx_activation", "silu")).lower()
                if wx_act == "exp":
                    attn_alpha = torch.exp(b).mean()
                elif wx_act == "sigmoid2":
                    attn_alpha = (2.0 * torch.sigmoid(b)).mean()
                else:
                    attn_alpha = (1.0 + torch.nn.functional.silu(b)).mean()

        mlp_alpha = None
        if hasattr(layer, "residual_mlp_scale"):
            t = layer.residual_mlp_scale
            mlp_alpha = t.detach().float().mean()
        elif hasattr(layer, "residual_mlp_wx"):
            wx = layer.residual_mlp_wx
            if hasattr(wx, "bias") and wx.bias is not None:
                b = wx.bias.detach().float()
                wx_act = str(getattr(layer, "_residual_wx_activation", "silu")).lower()
                if wx_act == "exp":
                    mlp_alpha = torch.exp(b).mean()
                elif wx_act == "sigmoid2":
                    mlp_alpha = (2.0 * torch.sigmoid(b)).mean()
                else:
                    mlp_alpha = (1.0 + torch.nn.functional.silu(b)).mean()

        if isinstance(attn_alpha, torch.Tensor):
            v = float(attn_alpha.item())
            attn_means.append(v)
            layer_attn[idx] = v
        if isinstance(mlp_alpha, torch.Tensor):
            v = float(mlp_alpha.item())
            mlp_means.append(v)
            layer_mlp[idx] = v

        attn_gamma = None
        if hasattr(layer, "residual_attn_branch_scale"):
            t = layer.residual_attn_branch_scale
            attn_gamma = t.detach().float().mean()
        elif hasattr(layer, "residual_attn_branch_wx"):
            wx = layer.residual_attn_branch_wx
            if hasattr(wx, "bias") and wx.bias is not None:
                b = wx.bias.detach().float()
                wx_act = str(getattr(layer, "_residual_wx_activation", "silu")).lower()
                if wx_act == "exp":
                    attn_gamma = torch.exp(b).mean()
                elif wx_act == "sigmoid2":
                    attn_gamma = (2.0 * torch.sigmoid(b)).mean()
                else:
                    attn_gamma = (1.0 + torch.nn.functional.silu(b)).mean()

        mlp_gamma = None
        if hasattr(layer, "residual_mlp_branch_scale"):
            t = layer.residual_mlp_branch_scale
            mlp_gamma = t.detach().float().mean()
        elif hasattr(layer, "residual_mlp_branch_wx"):
            wx = layer.residual_mlp_branch_wx
            if hasattr(wx, "bias") and wx.bias is not None:
                b = wx.bias.detach().float()
                wx_act = str(getattr(layer, "_residual_wx_activation", "silu")).lower()
                if wx_act == "exp":
                    mlp_gamma = torch.exp(b).mean()
                elif wx_act == "sigmoid2":
                    mlp_gamma = (2.0 * torch.sigmoid(b)).mean()
                else:
                    mlp_gamma = (1.0 + torch.nn.functional.silu(b)).mean()

        if isinstance(attn_gamma, torch.Tensor):
            v = float(attn_gamma.item())
            attn_branch_means.append(v)
            layer_attn_branch[idx] = v
        if isinstance(mlp_gamma, torch.Tensor):
            v = float(mlp_gamma.item())
            mlp_branch_means.append(v)
            layer_mlp_branch[idx] = v

    out = {}
    if len(attn_means) > 0:
        out["attn_alpha_mean"] = float(sum(attn_means) / len(attn_means))
    if len(mlp_means) > 0:
        out["mlp_alpha_mean"] = float(sum(mlp_means) / len(mlp_means))
    if ("attn_alpha_mean" in out) and ("mlp_alpha_mean" in out):
        out["alpha_mean"] = 0.5 * (out["attn_alpha_mean"] + out["mlp_alpha_mean"])
    if len(attn_branch_means) > 0:
        out["attn_branch_gamma_mean"] = float(sum(attn_branch_means) / len(attn_branch_means))
    if len(mlp_branch_means) > 0:
        out["mlp_branch_gamma_mean"] = float(sum(mlp_branch_means) / len(mlp_branch_means))
    if ("attn_branch_gamma_mean" in out) and ("mlp_branch_gamma_mean" in out):
        out["branch_gamma_mean"] = 0.5 * (out["attn_branch_gamma_mean"] + out["mlp_branch_gamma_mean"])
    out["layers"] = {
        "attn": layer_attn,
        "mlp": layer_mlp,
        "attn_branch": layer_attn_branch,
        "mlp_branch": layer_mlp_branch,
    }
    return out


@torch.no_grad()
def collect_residual_runtime_stats(module):
    out = {"layers": {"attn": {}, "mlp": {}, "attn_branch": {}, "mlp_branch": {}}}
    layers = []
    if hasattr(module, "transformer") and hasattr(module.transformer, "h"):
        layers = list(module.transformer.h)

    attn_vals = []
    mlp_vals = []
    attn_branch_vals = []
    mlp_branch_vals = []

    for lid, layer in enumerate(layers):
        idx = int(getattr(layer, "_residual_layer_idx", lid))
        kv = (
            ("_residual_last_attn_alpha_mean", "attn", attn_vals),
            ("_residual_last_mlp_alpha_mean", "mlp", mlp_vals),
            ("_residual_last_attn_branch_gamma_mean", "attn_branch", attn_branch_vals),
            ("_residual_last_mlp_branch_gamma_mean", "mlp_branch", mlp_branch_vals),
        )
        for attr, branch, vals in kv:
            if not hasattr(layer, attr):
                continue
            v = float(getattr(layer, attr))
            if not math.isfinite(v):
                continue
            out["layers"][branch][idx] = v
            vals.append(v)

    if len(attn_vals) > 0:
        out["attn_alpha_mean"] = float(sum(attn_vals) / len(attn_vals))
    if len(mlp_vals) > 0:
        out["mlp_alpha_mean"] = float(sum(mlp_vals) / len(mlp_vals))
    if ("attn_alpha_mean" in out) and ("mlp_alpha_mean" in out):
        out["alpha_mean"] = 0.5 * (out["attn_alpha_mean"] + out["mlp_alpha_mean"])
    if len(attn_branch_vals) > 0:
        out["attn_branch_gamma_mean"] = float(sum(attn_branch_vals) / len(attn_branch_vals))
    if len(mlp_branch_vals) > 0:
        out["mlp_branch_gamma_mean"] = float(sum(mlp_branch_vals) / len(mlp_branch_vals))
    if ("attn_branch_gamma_mean" in out) and ("mlp_branch_gamma_mean" in out):
        out["branch_gamma_mean"] = 0.5 * (out["attn_branch_gamma_mean"] + out["mlp_branch_gamma_mean"])
    return out


def compute_xsa_self_loss(xsa_layer_branch_proj_ratio):
    if not isinstance(xsa_layer_branch_proj_ratio, list) or len(xsa_layer_branch_proj_ratio) == 0:
        return None, None, None
    objective = str(xsa_self_loss_objective).lower().strip()
    if objective not in {"ratio", "ratio_detach", "parallel"}:
        objective = "ratio"

    def _pick(row: dict, branch: str):
        if objective == "ratio":
            return row.get(branch, None)
        if objective == "ratio_detach":
            return row.get(f"{branch}_ratio_detach", None)
        return row.get(f"{branch}_proj_sq", None)

    n_layers = len(xsa_layer_branch_proj_ratio)
    start = max(0, min(int(xsa_self_loss_start_layer), n_layers - 1))
    stride = max(1, int(xsa_self_loss_stride))
    vals = []
    attn_vals = []
    mlp_vals = []
    for i in range(start, n_layers, stride):
        v = xsa_layer_branch_proj_ratio[i]
        if not isinstance(v, dict):
            continue
        a = _pick(v, "attn")
        m = _pick(v, "mlp")
        if isinstance(a, torch.Tensor):
            vals.append(a)
            attn_vals.append(a)
        if isinstance(m, torch.Tensor):
            vals.append(m)
            mlp_vals.append(m)
    if len(vals) == 0:
        return None, None, None
    loss_total = torch.stack(vals).mean()
    loss_attn = torch.stack(attn_vals).mean() if len(attn_vals) > 0 else None
    loss_mlp = torch.stack(mlp_vals).mean() if len(mlp_vals) > 0 else None
    return loss_total, loss_attn, loss_mlp


def compute_xsa_layer_perp_para_ratio(xsa_layer_branch_proj_ratio, eps=1e-8):
    out = {"attn": {}, "mlp": {}}
    if not isinstance(xsa_layer_branch_proj_ratio, list):
        return out
    for lid, layer_vals in enumerate(xsa_layer_branch_proj_ratio):
        if not isinstance(layer_vals, dict):
            continue
        for branch in ("attn", "mlp"):
            v = layer_vals.get(branch, None)
            if not isinstance(v, torch.Tensor):
                continue
            # v = para_sq / total_sq (cos^2). Convert to ||dz_perp|| / ||dz_para||.
            para_sq_ratio = v.detach().float().clamp_min(eps)
            perp_sq_ratio = (1.0 - para_sq_ratio).clamp_min(0.0)
            dz_para = torch.sqrt(para_sq_ratio)
            dz_perp = torch.sqrt(perp_sq_ratio)
            ratio = dz_perp / dz_para
            out[branch][lid] = {
                "perp_para": float(ratio.item()),
            }
    return out


def compute_xsa_layer_state_cos(xsa_layer_branch_state_cos):
    out = {"attn": {}, "mlp": {}}
    if not isinstance(xsa_layer_branch_state_cos, list):
        return out
    for lid, layer_vals in enumerate(xsa_layer_branch_state_cos):
        if not isinstance(layer_vals, dict):
            continue
        for branch in ("attn", "mlp"):
            v = layer_vals.get(branch, None)
            if isinstance(v, torch.Tensor):
                out[branch][lid] = float(v.detach().float().item())
    return out


def compute_xsa_layer_magnitude_ratio(xsa_layer_branch_proj_ratio, eps=1e-8):
    out = {"attn": {}, "mlp": {}}
    def _to_scalar(v):
        if isinstance(v, torch.Tensor):
            return float(v.detach().float().item())
        if isinstance(v, (int, float)):
            return float(v)
        return None
    if not isinstance(xsa_layer_branch_proj_ratio, list):
        return out
    for lid, layer_vals in enumerate(xsa_layer_branch_proj_ratio):
        if not isinstance(layer_vals, dict):
            continue
        for branch in ("attn", "mlp"):
            y_sq = layer_vals.get(f"{branch}_y_sq", None)
            x_sq = layer_vals.get(f"{branch}_x_sq", None)
            yv = _to_scalar(y_sq)
            xv = _to_scalar(x_sq)
            if (yv is None) or (xv is None):
                continue
            ratio = math.sqrt(max(yv, 0.0) / max(xv, eps))
            out[branch][lid] = {"y_over_x": float(ratio)}
    return out


def compute_xsa_layer_para_perp_magnitude(xsa_layer_branch_proj_ratio):
    out = {"attn": {}, "mlp": {}}
    def _to_scalar(v):
        if isinstance(v, torch.Tensor):
            return float(v.detach().float().item())
        if isinstance(v, (int, float)):
            return float(v)
        return None
    if not isinstance(xsa_layer_branch_proj_ratio, list):
        return out
    for lid, layer_vals in enumerate(xsa_layer_branch_proj_ratio):
        if not isinstance(layer_vals, dict):
            continue
        for branch in ("attn", "mlp"):
            proj_sq = _to_scalar(layer_vals.get(f"{branch}_proj_sq", None))
            y_sq = _to_scalar(layer_vals.get(f"{branch}_y_sq", None))
            if (proj_sq is None) or (y_sq is None):
                continue
            para = math.sqrt(max(proj_sq, 0.0))
            perp = math.sqrt(max(y_sq - proj_sq, 0.0))
            out[branch][lid] = {"para": float(para), "perp": float(perp)}
    return out


def compute_attn_preproj_self_value_perp_para_ratio(xsa_layer_branch_proj_ratio, eps=1e-8):
    out = {}
    if not isinstance(xsa_layer_branch_proj_ratio, list):
        return out
    for lid, layer_vals in enumerate(xsa_layer_branch_proj_ratio):
        if not isinstance(layer_vals, dict):
            continue
        v = layer_vals.get("attn_preproj_self_value", None)
        if not isinstance(v, torch.Tensor):
            continue
        para_sq_ratio = v.detach().float().clamp_min(eps)
        perp_sq_ratio = (1.0 - para_sq_ratio).clamp_min(0.0)
        dz_para = torch.sqrt(para_sq_ratio)
        dz_perp = torch.sqrt(perp_sq_ratio)
        out[lid] = {"perp_para": float((dz_perp / dz_para).item())}
    return out


def compute_attn_preproj_self_value_para_perp_magnitude(xsa_layer_branch_proj_ratio):
    out = {}
    def _to_scalar(v):
        if isinstance(v, torch.Tensor):
            return float(v.detach().float().item())
        if isinstance(v, (int, float)):
            return float(v)
        return None
    if not isinstance(xsa_layer_branch_proj_ratio, list):
        return out
    for lid, layer_vals in enumerate(xsa_layer_branch_proj_ratio):
        if not isinstance(layer_vals, dict):
            continue
        proj_sq = _to_scalar(layer_vals.get("attn_preproj_self_value_proj_sq", None))
        y_sq = _to_scalar(layer_vals.get("attn_preproj_self_value_y_sq", None))
        if (proj_sq is None) or (y_sq is None):
            continue
        para = math.sqrt(max(proj_sq, 0.0))
        perp = math.sqrt(max(y_sq - proj_sq, 0.0))
        out[lid] = {"para": float(para), "perp": float(perp)}
    return out


def compute_layerwise_sim(layer_outputs):
    out = {}
    if not isinstance(layer_outputs, list):
        return out
    prev = None
    for lid, h in enumerate(layer_outputs):
        if not isinstance(h, torch.Tensor):
            prev = None
            continue
        if isinstance(prev, torch.Tensor):
            sim = F.cosine_similarity(prev.float(), h.float(), dim=-1).mean()
            out[lid] = float(sim.detach().item())
        prev = h
    return out


def format_token_count(n: int) -> str:
    n = int(n)
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B tok"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.3f}M tok"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.3f}K tok"
    return f"{n} tok"


@torch.no_grad()
def collect_logging_metrics(module, x, y, need_layer_outputs=False):
    if not (hasattr(module, "transformer") and hasattr(module.transformer, "h")):
        return None, None, None, None, None, None, None
    layers = list(module.transformer.h)
    prev_flags = [bool(getattr(blk, "track_xsa_self_metric", False)) for blk in layers]
    for blk in layers:
        blk.track_xsa_self_metric = True
    try:
        with ctx:
            _, _, extras = module(
                x,
                y,
                return_layer_outputs=bool(need_layer_outputs),
                return_xsa_metrics=True,
            )
        if not isinstance(extras, dict):
            extras = {}
        xsa_layer_branch_proj_ratio = extras.get("xsa_layer_branch_proj_ratio", None)
        xsa_layer_branch_state_cos = extras.get("xsa_layer_branch_state_cos", None)
        layer_outputs = extras.get("layer_outputs", None) if bool(need_layer_outputs) else None
        return (
            compute_xsa_layer_perp_para_ratio(xsa_layer_branch_proj_ratio),
            compute_xsa_layer_state_cos(xsa_layer_branch_state_cos),
            compute_xsa_layer_magnitude_ratio(xsa_layer_branch_proj_ratio),
            compute_xsa_layer_para_perp_magnitude(xsa_layer_branch_proj_ratio),
            compute_attn_preproj_self_value_perp_para_ratio(xsa_layer_branch_proj_ratio),
            compute_attn_preproj_self_value_para_perp_magnitude(xsa_layer_branch_proj_ratio),
            compute_layerwise_sim(layer_outputs),
        )
    finally:
        for blk, flag in zip(layers, prev_flags):
            blk.track_xsa_self_metric = flag


@torch.no_grad()
def collect_grad_norms(module):
    total_sq = 0.0
    residual_sq = 0.0
    has_any = False
    has_residual = False
    for name, p in module.named_parameters():
        g = p.grad
        if g is None:
            continue
        has_any = True
        gg = g.detach().float()
        n2 = float((gg * gg).sum().item())
        total_sq += n2
        if ("residual" in name) or ("wx" in name):
            residual_sq += n2
            has_residual = True
    out = {}
    if has_any:
        out["grad_norm"] = float(math.sqrt(max(total_sq, 0.0)))
    if has_residual:
        out["grad_norm_residual"] = float(math.sqrt(max(residual_sq, 0.0)))
    return out

# NOTE: DDP wrapping is intentionally deferred until after XSA/residual
# per-layer requires_grad flags are fully configured.

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def ddp_any_nonfinite(local_nonfinite: bool):
    if not local_nonfinite and not ddp:
        return False
    flag = torch.tensor(
        1 if local_nonfinite else 0,
        device=device if device_type == "cuda" else "cpu",
        dtype=torch.int32,
    )
    if ddp:
        all_reduce(flag, op=ReduceOp.MAX)
    return bool(flag.item() > 0)

# logging
_wandb_run_id = None  # set after init, persisted in checkpoint for future resumes
if wandb_log and master_process:
    import wandb
    if _wandb_resume_id is not None:
        if int(iter_num) <= 0:
            print(
                "[WARN] iter_num<=0 while checkpoint carries wandb_id; "
                "starting a NEW W&B run to avoid corrupting old curves.",
                flush=True,
            )
            _wandb_resume_id = None
        else:
            print(f"[INFO] Resuming W&B run id={_wandb_resume_id} from iter={int(iter_num)}", flush=True)
            wandb.init(project=wandb_project, id=_wandb_resume_id, resume="must", config=config)
    if _wandb_resume_id is None:
        print(f"[INFO] Starting new W&B run name={wandb_run_name} from iter={int(iter_num)}", flush=True)
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    wandb.define_metric("iter", summary="max")
    for _wandb_prefix in ("train/*", "val/*", "eval/*"):
        wandb.define_metric(_wandb_prefix, step_metric="iter")
    _wandb_run_id = wandb.run.id

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if isinstance(model, DDP) else model # unwrap DDP container if needed
if hasattr(raw_model, "transformer") and hasattr(raw_model.transformer, "h"):
    _xsa_ref = str(xsa_forward_ref).lower().strip()
    _xsa_target = str(xsa_forward_target).lower().strip()
    _xsa_space = str(xsa_forward_space).lower().strip()
    _xsa_op = str(xsa_forward_op).lower().strip()
    _xsa_alpha = float(xsa_forward_alpha)
    _xsa_subtract_scale = float(xsa_forward_subtract_scale)
    _xsa_learnable_gate = bool(xsa_forward_learnable_gate)
    _xsa_gate_init = float(xsa_forward_gate_init)
    _xsa_gate_mode = str(xsa_forward_gate_mode).lower().strip()
    _xsa_learnable_alpha = bool(xsa_forward_learnable_alpha)
    _xsa_alpha_min = float(xsa_forward_alpha_min)
    _xsa_alpha_max = float(xsa_forward_alpha_max)
    _xsa_learnable_gamma = bool(xsa_forward_learnable_gamma)
    _xsa_gamma_per_head = bool(xsa_forward_gamma_per_head)
    _xsa_gamma_init = float(xsa_forward_gamma_init)
    if _xsa_gamma_init <= 0.0:
        raise ValueError("xsa_forward_gamma_init must be > 0")
    _xsa_gamma_raw_init = math.log(_xsa_gamma_init)
    if _xsa_alpha_max <= _xsa_alpha_min:
        raise ValueError("xsa_forward_alpha_max must be greater than xsa_forward_alpha_min")
    _xsa_alpha_target = min(max(_xsa_alpha, _xsa_alpha_min + 1e-6), _xsa_alpha_max - 1e-6)
    _xsa_alpha01 = (_xsa_alpha_target - _xsa_alpha_min) / (_xsa_alpha_max - _xsa_alpha_min)
    _xsa_alpha_raw_init = math.log(_xsa_alpha01 / (1.0 - _xsa_alpha01))
    if _xsa_space not in {"post_o_proj", "pre_o_proj"}:
        raise ValueError("xsa_forward_space must be one of: post_o_proj, pre_o_proj")
    if _xsa_op not in {"remove_parallel", "keep_parallel", "add_parallel", "negate_parallel"}:
        raise ValueError(
            "xsa_forward_op must be one of: remove_parallel, keep_parallel, add_parallel, negate_parallel"
        )
    if _xsa_subtract_scale < 0.0:
        raise ValueError("xsa_forward_subtract_scale must be >= 0")
    if _xsa_gate_mode not in {"static", "token"}:
        raise ValueError("xsa_forward_gate_mode must be one of: static, token")
    if _xsa_learnable_gate:
        if _xsa_gate_init <= 0.0 or _xsa_gate_init >= 1.0:
            raise ValueError("xsa_forward_gate_init must be in (0, 1) when xsa_forward_learnable_gate=True")
        _xsa_gate_raw_init = math.log(_xsa_gate_init / (1.0 - _xsa_gate_init))
    else:
        _xsa_gate_raw_init = 0.0
    if _xsa_space == "pre_o_proj" and _xsa_ref != "self_value":
        if master_process:
            print(
                f"[WARN] xsa_forward_space=pre_o_proj only applies to xsa_forward_ref=self_value; "
                f"got ref={_xsa_ref}, falling back to post_o_proj.",
                flush=True,
            )
        _xsa_space = "post_o_proj"
    if _xsa_ref == "self_value" and _xsa_target != "attn":
        if master_process:
            print(
                f"[WARN] xsa_forward_ref=self_value requires xsa_forward_target=attn; "
                f"got target={_xsa_target}, forcing to attn.",
                flush=True,
        )
        _xsa_target = "attn"
    _xsa_layers = list(raw_model.transformer.h)
    _xsa_n_layers = len(_xsa_layers)
    _xsa_start_layer = max(0, min(int(xsa_forward_start_layer), max(_xsa_n_layers - 1, 0)))
    _xsa_end_layer_raw = int(xsa_forward_end_layer)
    _xsa_end_layer = _xsa_n_layers - 1 if _xsa_end_layer_raw < 0 else _xsa_end_layer_raw
    _xsa_end_layer = max(0, min(_xsa_end_layer, max(_xsa_n_layers - 1, 0)))
    _xsa_stride = max(1, int(xsa_forward_stride))
    if master_process and bool(xsa_forward_only):
        print(
            f"[INFO] xsa_forward target={_xsa_target} ref={_xsa_ref} "
            f"space={_xsa_space} op={_xsa_op} alpha={_xsa_alpha} subtract_scale={_xsa_subtract_scale} "
            f"layers={_xsa_start_layer}..{_xsa_end_layer} stride={_xsa_stride}",
            flush=True,
        )
    if master_process and bool(xsa_forward_only) and _xsa_learnable_gate:
        print(
            f"[INFO] xsa_forward learnable_gate=True mode={_xsa_gate_mode} init={_xsa_gate_init}",
            flush=True,
        )
        print(
            "[INFO] wandb gate keys: train/xsa_gate_mean and train/xsa_gate/layer_{k}",
            flush=True,
        )
    if master_process and bool(xsa_forward_only) and _xsa_learnable_alpha:
        print(
            f"[INFO] xsa_forward learnable_alpha=True range=[{_xsa_alpha_min}, {_xsa_alpha_max}] init={_xsa_alpha_target}",
            flush=True,
        )
    if master_process and bool(xsa_forward_only) and _xsa_learnable_gamma:
        print(
            f"[INFO] xsa_forward learnable_gamma=True per_head={_xsa_gamma_per_head} init={_xsa_gamma_init}",
            flush=True,
        )
        print(
            "[INFO] wandb gamma keys: train/xsa_gamma_mean, train/xsa_gamma/layer_{k}, "
            "and train/xsa_gamma/layer_{k}/head_{h} when per_head=True",
            flush=True,
        )
    _need_logging_only_xsa_metrics = bool(wandb_log) and (
        bool(wandb_log_layerwise) or bool(wandb_log_xsa_per_layer) or bool(wandb_log_xsa_means)
    )
    if int(xsa_metrics_interval) == 0:
        xsa_metrics_interval = int(log_interval) if _need_logging_only_xsa_metrics else -1
    for _lid, _blk in enumerate(_xsa_layers):
        _enabled = (
            bool(xsa_forward_only)
            and (_lid >= _xsa_start_layer)
            and (_lid <= _xsa_end_layer)
            and (((_lid - _xsa_start_layer) % _xsa_stride) == 0)
        )
        _blk.track_xsa_self_metric = False
        _blk.xsa_forward_only = bool(xsa_forward_only)
        _blk.xsa_forward_enabled = _enabled
        _blk.xsa_forward_target = _xsa_target
        _blk.xsa_forward_ref = _xsa_ref
        _blk.xsa_forward_space = _xsa_space
        _blk.xsa_forward_op = _xsa_op
        _blk.xsa_forward_alpha = _xsa_alpha
        _blk.xsa_forward_subtract_scale = _xsa_subtract_scale
        _blk.xsa_forward_learnable_gate = _xsa_learnable_gate
        _blk.xsa_forward_gate_mode = _xsa_gate_mode
        _blk.xsa_forward_learnable_alpha = _xsa_learnable_alpha
        _blk.xsa_forward_alpha_min = _xsa_alpha_min
        _blk.xsa_forward_alpha_max = _xsa_alpha_max
        _blk.xsa_forward_learnable_gamma = _xsa_learnable_gamma
        _blk.xsa_forward_gamma_per_head = _xsa_gamma_per_head
        if hasattr(_blk, "xsa_forward_alpha_raw"):
            _blk.xsa_forward_alpha_raw.requires_grad_(bool(_enabled and _xsa_learnable_alpha))
            if bool(_enabled and _xsa_learnable_alpha) and int(iter_num) == 0:
                _blk.xsa_forward_alpha_raw.data.fill_(_xsa_alpha_raw_init)
        if hasattr(_blk, "xsa_forward_gamma_raw"):
            _blk.xsa_forward_gamma_raw.requires_grad_(bool(_enabled and _xsa_learnable_gamma))
            if bool(_enabled and _xsa_learnable_gamma) and int(iter_num) == 0:
                _blk.xsa_forward_gamma_raw.data.fill_(_xsa_gamma_raw_init)
        if hasattr(_blk, "xsa_forward_gate_raw"):
            _blk.xsa_forward_gate_raw.requires_grad_(bool(_enabled and _xsa_learnable_gate and _xsa_gate_mode == "static"))
            if bool(_enabled and _xsa_learnable_gate) and int(iter_num) == 0:
                _blk.xsa_forward_gate_raw.data.fill_(_xsa_gate_raw_init)
        if hasattr(_blk, "xsa_forward_gate_proj"):
            _blk.xsa_forward_gate_proj.weight.requires_grad_(bool(_enabled and _xsa_learnable_gate and _xsa_gate_mode == "token"))
            _blk.xsa_forward_gate_proj.bias.requires_grad_(bool(_enabled and _xsa_learnable_gate and _xsa_gate_mode == "token"))
            if bool(_enabled and _xsa_learnable_gate and _xsa_gate_mode == "token") and int(iter_num) == 0:
                _blk.xsa_forward_gate_proj.weight.data.zero_()
                _blk.xsa_forward_gate_proj.bias.data.fill_(_xsa_gate_raw_init)

# wrap model into DDP container (after all dynamic requires_grad toggles)
if ddp and not isinstance(model, DDP):
    ddp_find_unused = bool(xsa_forward_only) and (
        bool(xsa_forward_learnable_alpha) or bool(xsa_forward_learnable_gamma) or bool(xsa_forward_learnable_gate)
    )
    if master_process and ddp_find_unused:
        print("[INFO] DDP find_unused_parameters=True (XSA learnable alpha/gamma/gate enabled)", flush=True)
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=ddp_find_unused)
    raw_model = model.module
running_mfu = -1.0
last_task_loss = None
last_xsa_self_loss = None
last_xsa_self_attn_loss = None
last_xsa_self_mlp_loss = None
last_xsa_lambda = 0.0
last_xsa_weighted_loss = None
last_xsa_perp_para = {"attn": {}, "mlp": {}}
last_xsa_state_cos = {"attn": {}, "mlp": {}}
last_xsa_magnitude = {"attn": {}, "mlp": {}}
last_xsa_para_perp_mag = {"attn": {}, "mlp": {}}
last_attn_preproj_self_value_perp_para = {}
last_attn_preproj_self_value_para_perp_mag = {}
last_layerwise_sim = {}
last_metrics_batch = None
last_grad_norm = None
last_grad_norm_residual = None
if device_type == "cuda":
    torch.cuda.reset_peak_memory_stats()
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * float(param_group.get('lr_scale', 1.0))

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "eval/tokens": tokens_seen,
                "eval/tokens_b": tokens_seen / 1e9,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "eval/lr_next": lr,
            })
        is_best_checkpoint = losses['val'] < best_val_loss
        should_save_best_checkpoint = is_best_checkpoint
        if is_best_checkpoint:
            best_val_loss = losses['val']
            if int(best_checkpoint_min_interval_evals) > 0 and last_best_checkpoint_iter is not None:
                min_best_interval_steps = int(best_checkpoint_min_interval_evals) * int(eval_interval)
                if (int(iter_num) - int(last_best_checkpoint_iter)) < min_best_interval_steps:
                    should_save_best_checkpoint = False
                    print(
                        f"[INFO] best checkpoint improved at step {iter_num} but skipped save "
                        f"due to best_checkpoint_min_interval_evals={int(best_checkpoint_min_interval_evals)} "
                        f"(last_best_step={last_best_checkpoint_iter})",
                        flush=True,
                    )
        if should_save_best_checkpoint or always_save_checkpoint:
            eval_save_counter = int(eval_save_counter) + 1
            periodic_by_eval = (
                int(checkpoint_every_n_evals) > 0
                and (int(eval_save_counter) % int(checkpoint_every_n_evals) == 0)
            )
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'tokens_seen': int(tokens_seen),
                'eval_save_counter': int(eval_save_counter),
                'best_val_loss': best_val_loss,
                'config': config,
                'wandb_id': _wandb_run_id,
            }
            save_training_checkpoint(
                checkpoint,
                iter_num,
                should_save_best_checkpoint,
                save_periodic=bool(periodic_by_eval),
            )
            if should_save_best_checkpoint:
                last_best_checkpoint_iter = int(iter_num)
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    stop_training_nonfinite = False
    nonfinite_reason = ""
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logging_xsa_metrics = (
                bool(wandb_log)
                and (bool(wandb_log_layerwise) or bool(wandb_log_xsa_per_layer) or bool(wandb_log_xsa_means))
                and
                int(xsa_metrics_interval) > 0
                and (((iter_num + 1) % int(xsa_metrics_interval)) == 0)
                and (micro_step == gradient_accumulation_steps - 1)
            )
            need_layer_outputs = False
            need_xsa_metrics = bool(enable_xsa_self_loss)
            if hasattr(raw_model, "transformer") and hasattr(raw_model.transformer, "h"):
                for _blk in raw_model.transformer.h:
                    _blk.track_xsa_self_metric = need_xsa_metrics
            if need_layer_outputs or need_xsa_metrics:
                logits, task_loss, extras = model(
                    X,
                    Y,
                    return_layer_outputs=need_layer_outputs,
                    return_xsa_metrics=need_xsa_metrics,
                )
                if not isinstance(extras, dict):
                    extras = {}
            else:
                logits, task_loss = model(X, Y)
                extras = {}

            total_loss = task_loss * float(task_loss_weight)
            last_task_loss = task_loss.detach()

            xsa_layer_branch_proj_ratio = extras.get("xsa_layer_branch_proj_ratio", None)
            xsa_loss, xsa_attn_loss, xsa_mlp_loss = compute_xsa_self_loss(xsa_layer_branch_proj_ratio)
            if xsa_loss is None:
                xsa_loss = task_loss.new_zeros(())
            if int(xsa_self_loss_warmup_iters) > 0:
                xsa_warmup = min(float(iter_num + 1) / float(xsa_self_loss_warmup_iters), 1.0)
            else:
                xsa_warmup = 1.0
            xsa_lambda = float(xsa_self_loss_lambda) * float(xsa_warmup) if enable_xsa_self_loss else 0.0
            if (
                enable_xsa_self_loss
                and xsa_lambda > 0.0
                and isinstance(xsa_loss, torch.Tensor)
                and (not xsa_loss.requires_grad)
                and master_process
                and iter_num < 5
            ):
                print(
                    "[WARN][xsa] xsa_loss.requires_grad=False while xsa is enabled. "
                    "xsa objective will not affect optimization.",
                    flush=True,
                )
            total_loss = total_loss + xsa_lambda * xsa_loss
            last_xsa_self_loss = xsa_loss.detach()
            last_xsa_self_attn_loss = xsa_attn_loss.detach() if isinstance(xsa_attn_loss, torch.Tensor) else None
            last_xsa_self_mlp_loss = xsa_mlp_loss.detach() if isinstance(xsa_mlp_loss, torch.Tensor) else None
            last_xsa_lambda = xsa_lambda
            last_xsa_weighted_loss = (xsa_loss.detach() * float(xsa_lambda)) if isinstance(xsa_loss, torch.Tensor) else None
            if bool(stop_on_nonfinite):
                local_nonfinite = not bool(torch.isfinite(total_loss.detach()).item())
                if ddp_any_nonfinite(local_nonfinite):
                    stop_training_nonfinite = True
                    nonfinite_reason = "non-finite total_loss detected"
                    break
            loss = total_loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        if logging_xsa_metrics:
            last_metrics_batch = (X.detach(), Y.detach())
        # For xsa debug runs, keep a fixed batch to verify pure optimization behavior.
        if not bool(xsa_debug_fixed_batch):
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    if stop_training_nonfinite:
        optimizer.zero_grad(set_to_none=True)
        if master_process:
            print(f"[STOP] iter {iter_num}: {nonfinite_reason}. Terminating training early.")
        break
    # unscale once so grad norms and clipping are on true (unscaled) grads
    scaler.unscale_(optimizer)
    should_log_now = (iter_num % log_interval == 0)
    if should_log_now or bool(stop_on_nonfinite):
        grad_stats = collect_grad_norms(raw_model)
        last_grad_norm = grad_stats.get("grad_norm", None)
        last_grad_norm_residual = grad_stats.get("grad_norm_residual", None)
    if bool(stop_on_nonfinite):
        local_nonfinite_grad = (
            (isinstance(last_grad_norm, (int, float)) and (not math.isfinite(float(last_grad_norm))))
            or (isinstance(last_grad_norm_residual, (int, float)) and (not math.isfinite(float(last_grad_norm_residual))))
        )
        if ddp_any_nonfinite(local_nonfinite_grad):
            optimizer.zero_grad(set_to_none=True)
            if master_process:
                print(f"[STOP] iter {iter_num}: non-finite grad norm detected. Terminating training early.")
            break
    # clip the gradient
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    if (
        bool(wandb_log)
        and (bool(wandb_log_layerwise) or bool(wandb_log_xsa_per_layer) or bool(wandb_log_xsa_means))
        and
        int(xsa_metrics_interval) > 0
        and (((iter_num + 1) % int(xsa_metrics_interval)) == 0)
        and isinstance(last_metrics_batch, tuple)
    ):
        metrics = collect_logging_metrics(
            raw_model,
            last_metrics_batch[0],
            last_metrics_batch[1],
            need_layer_outputs=bool(wandb_log_layerwise),
        )
        if metrics is not None:
            (
                last_xsa_perp_para,
                last_xsa_state_cos,
                last_xsa_magnitude,
                last_xsa_para_perp_mag,
                last_attn_preproj_self_value_perp_para,
                last_attn_preproj_self_value_para_perp_mag,
                last_layerwise_sim,
            ) = metrics
    last_metrics_batch = None

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        total_lossf = loss.item() * gradient_accumulation_steps
        task_lossf = total_lossf
        if isinstance(last_task_loss, torch.Tensor):
            task_lossf = float(last_task_loss.float().item())
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        peak_mem_gb = None
        if device_type == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        metrics_suffix = ""
        grad_parts = []
        if isinstance(last_grad_norm, (int, float)):
            grad_parts.append(f"g {float(last_grad_norm):.3f}")
        if isinstance(last_grad_norm_residual, (int, float)):
            grad_parts.append(f"g_res {float(last_grad_norm_residual):.3f}")
        if isinstance(peak_mem_gb, float):
            grad_parts.append(f"mem {peak_mem_gb:.2f}GB")
        if grad_parts:
            metrics_suffix += ", " + ", ".join(grad_parts)
        completed_iter = int(iter_num) + 1
        tokens_seen_after_step = int(tokens_seen) + int(tokens_per_iter)
        tokens_seen_print = tokens_seen_after_step
        print(
            f"iter {completed_iter}: "
            f"loss {task_lossf:.4f}, "
            f"lr(step) {lr:.2e}, "
            f"tok/step {format_token_count(tokens_per_iter)}, "
            f"tokens_seen {format_token_count(tokens_seen_print)}, "
            f"time {dt*1000:.2f}ms, "
            f"mfu {running_mfu*100:.2f}%"
            f"{metrics_suffix}"
        )
        if wandb_log:
            log_dict = {
                "iter": completed_iter,
                "train/tokens": tokens_seen_after_step,
                "train/tokens_b": tokens_seen_after_step / 1e9,
                "train/tokens_per_iter": int(tokens_per_iter),
                "train/tokens_per_iter_m": float(tokens_per_iter) / 1e6,
                # Keep this comparable with baseline task CE.
                "train/loss_iter": task_lossf,
                "train/loss_total": float(total_lossf),
                "train/lr": lr,
                "train/mfu": running_mfu * 100,
            }
            if isinstance(last_grad_norm, (int, float)):
                log_dict["train/grad_norm"] = float(last_grad_norm)
            if isinstance(last_grad_norm_residual, (int, float)):
                log_dict["train/grad_norm_residual"] = float(last_grad_norm_residual)
            if isinstance(peak_mem_gb, float):
                log_dict["train/max_memory_allocated_gb"] = float(peak_mem_gb)
            residual_group_lrs = [float(pg["lr"]) for pg in optimizer.param_groups if float(pg.get("lr_scale", 1.0)) != 1.0]
            if len(residual_group_lrs) > 0:
                log_dict["train/lr_residual"] = sum(residual_group_lrs) / len(residual_group_lrs)
            if bool(wandb_log_aux_losses):
                log_dict["train/xsa_self_lambda"] = float(last_xsa_lambda)
                _xsa_obj = str(xsa_self_loss_objective).lower().strip()
                _xsa_obj_id = 0
                if _xsa_obj == "ratio_detach":
                    _xsa_obj_id = 1
                elif _xsa_obj == "parallel":
                    _xsa_obj_id = 2
                log_dict["train/xsa_loss_objective_id"] = float(_xsa_obj_id)
                if isinstance(last_xsa_self_loss, torch.Tensor):
                    log_dict["train/loss_xsa_self"] = float(last_xsa_self_loss.float().item())
                if isinstance(last_xsa_weighted_loss, torch.Tensor):
                    log_dict["train/loss_xsa_weighted"] = float(last_xsa_weighted_loss.float().item())
                if isinstance(last_xsa_self_attn_loss, torch.Tensor):
                    log_dict["train/loss_xsa_self_attn"] = float(last_xsa_self_attn_loss.float().item())
                if isinstance(last_xsa_self_mlp_loss, torch.Tensor):
                    log_dict["train/loss_xsa_self_mlp"] = float(last_xsa_self_mlp_loss.float().item())
            if bool(wandb_log_xsa_means) and isinstance(last_xsa_magnitude, dict):
                for branch in ("attn", "mlp"):
                    vals = last_xsa_magnitude.get(branch, {})
                    if not isinstance(vals, dict):
                        continue
                    mag_vals = [float(v["y_over_x"]) for v in vals.values() if isinstance(v, dict) and ("y_over_x" in v)]
                    if len(mag_vals) > 0:
                        log_dict[f"train/xsa_mag_ratio_{branch}_mean"] = float(sum(mag_vals) / len(mag_vals))
            if bool(wandb_log_xsa_means) and isinstance(last_xsa_para_perp_mag, dict):
                for branch in ("attn", "mlp"):
                    vals = last_xsa_para_perp_mag.get(branch, {})
                    if not isinstance(vals, dict):
                        continue
                    para_vals = [float(v["para"]) for v in vals.values() if isinstance(v, dict) and ("para" in v)]
                    perp_vals = [float(v["perp"]) for v in vals.values() if isinstance(v, dict) and ("perp" in v)]
                    if len(para_vals) > 0:
                        log_dict[f"train/xsa_para_abs_{branch}_mean"] = float(sum(para_vals) / len(para_vals))
                    if len(perp_vals) > 0:
                        log_dict[f"train/xsa_perp_abs_{branch}_mean"] = float(sum(perp_vals) / len(perp_vals))
            if bool(wandb_log_xsa_means) and isinstance(last_attn_preproj_self_value_para_perp_mag, dict):
                para_vals = [float(v["para"]) for v in last_attn_preproj_self_value_para_perp_mag.values() if isinstance(v, dict) and ("para" in v)]
                perp_vals = [float(v["perp"]) for v in last_attn_preproj_self_value_para_perp_mag.values() if isinstance(v, dict) and ("perp" in v)]
                if len(para_vals) > 0:
                    log_dict["train/xsa_preproj_self_value_para_abs_attn_mean"] = float(sum(para_vals) / len(para_vals))
                if len(perp_vals) > 0:
                    log_dict["train/xsa_preproj_self_value_perp_abs_attn_mean"] = float(sum(perp_vals) / len(perp_vals))
            if bool(wandb_log_xsa_per_layer) and isinstance(last_xsa_perp_para, dict):
                for branch in ("attn", "mlp"):
                    vals = last_xsa_perp_para.get(branch, {})
                    if isinstance(vals, dict):
                        for lid, v in vals.items():
                            if isinstance(v, dict):
                                if "perp_para" in v:
                                    vv = float(v["perp_para"])
                                    # Grouped keys for easier W&B multi-line visualization.
                                    log_dict[f"train/xsa_perp_para/{branch}/layer_{int(lid)}"] = vv
            if bool(wandb_log_xsa_per_layer) and isinstance(last_xsa_magnitude, dict):
                for branch in ("attn", "mlp"):
                    vals = last_xsa_magnitude.get(branch, {})
                    if isinstance(vals, dict):
                        for lid, v in vals.items():
                            if isinstance(v, dict) and ("y_over_x" in v):
                                log_dict[f"train/xsa_mag_ratio/{branch}/layer_{int(lid)}"] = float(v["y_over_x"])
            if bool(wandb_log_xsa_per_layer) and isinstance(last_xsa_para_perp_mag, dict):
                for branch in ("attn", "mlp"):
                    vals = last_xsa_para_perp_mag.get(branch, {})
                    if isinstance(vals, dict):
                        for lid, v in vals.items():
                            if isinstance(v, dict):
                                if "para" in v:
                                    log_dict[f"train/xsa_para_abs/{branch}/layer_{int(lid)}"] = float(v["para"])
                                if "perp" in v:
                                    log_dict[f"train/xsa_perp_abs/{branch}/layer_{int(lid)}"] = float(v["perp"])
            if bool(wandb_log_xsa_per_layer) and isinstance(last_attn_preproj_self_value_perp_para, dict):
                for lid, v in last_attn_preproj_self_value_perp_para.items():
                    if isinstance(v, dict) and ("perp_para" in v):
                        log_dict[f"train/xsa_preproj_self_value_perp_para/attn/layer_{int(lid)}"] = float(v["perp_para"])
            if bool(wandb_log_xsa_per_layer) and isinstance(last_attn_preproj_self_value_para_perp_mag, dict):
                for lid, v in last_attn_preproj_self_value_para_perp_mag.items():
                    if isinstance(v, dict):
                        if "para" in v:
                            log_dict[f"train/xsa_preproj_self_value_para_abs/attn/layer_{int(lid)}"] = float(v["para"])
                        if "perp" in v:
                            log_dict[f"train/xsa_preproj_self_value_perp_abs/attn/layer_{int(lid)}"] = float(v["perp"])
            if bool(wandb_log_layerwise) and isinstance(last_layerwise_sim, dict):
                if len(last_layerwise_sim) > 0:
                    log_dict["train/layerwise_sim_mean"] = float(
                        sum(float(v) for v in last_layerwise_sim.values()) / len(last_layerwise_sim)
                    )
                for lid, v in last_layerwise_sim.items():
                    log_dict[f"train/layerwise_sim/layer_{int(lid)}"] = float(v)
            if bool(wandb_log_xsa_per_layer) and isinstance(last_xsa_state_cos, dict):
                for branch in ("attn", "mlp"):
                    vals = last_xsa_state_cos.get(branch, {})
                    if isinstance(vals, dict):
                        for lid, v in vals.items():
                            vv = float(v)
                            log_dict[f"train/latent_sim/{branch}/layer_{int(lid)}"] = vv
            if hasattr(raw_model, "transformer") and hasattr(raw_model.transformer, "h"):
                alpha_vals = []
                for lid, blk in enumerate(raw_model.transformer.h):
                    if not bool(getattr(blk, "xsa_forward_enabled", False)):
                        continue
                    v = None
                    if bool(getattr(blk, "xsa_forward_learnable_alpha", False)) and hasattr(blk, "xsa_forward_alpha_raw"):
                        v = torch.sigmoid(blk.xsa_forward_alpha_raw.float())
                        amin = float(getattr(blk, "xsa_forward_alpha_min", 0.0))
                        amax = float(getattr(blk, "xsa_forward_alpha_max", 1.0))
                        v = amin + (amax - amin) * v
                    else:
                        v = getattr(blk, "_xsa_forward_alpha_value", None)
                    if isinstance(v, torch.Tensor):
                        vv = float(v.float().mean().item())
                    elif isinstance(v, (int, float)):
                        vv = float(v)
                    else:
                        continue
                    alpha_vals.append(vv)
                    log_dict[f"train/xsa_alpha/layer_{int(lid)}"] = vv
                if len(alpha_vals) > 0:
                    log_dict["train/xsa_alpha_mean"] = float(sum(alpha_vals) / len(alpha_vals))
            if hasattr(raw_model, "transformer") and hasattr(raw_model.transformer, "h"):
                gate_vals = []
                for lid, blk in enumerate(raw_model.transformer.h):
                    if not bool(getattr(blk, "xsa_forward_enabled", False)):
                        continue
                    v = None
                    gate_mode = str(getattr(blk, "xsa_forward_gate_mode", "static")).lower().strip()
                    if (
                        bool(getattr(blk, "xsa_forward_learnable_gate", False))
                        and gate_mode == "static"
                        and hasattr(blk, "xsa_forward_gate_raw")
                    ):
                        v = torch.sigmoid(blk.xsa_forward_gate_raw.float())
                    else:
                        v = getattr(blk, "_xsa_forward_gate_value", None)
                        # For token-wise gate, runtime cache can be missing in some graph/runtime paths.
                        # Fall back to parameterized bias so W&B keys remain present and trackable.
                        if (
                            v is None
                            and bool(getattr(blk, "xsa_forward_learnable_gate", False))
                            and gate_mode == "token"
                            and hasattr(blk, "xsa_forward_gate_proj")
                            and hasattr(blk.xsa_forward_gate_proj, "bias")
                            and blk.xsa_forward_gate_proj.bias is not None
                        ):
                            v = torch.sigmoid(blk.xsa_forward_gate_proj.bias.float())
                    if isinstance(v, torch.Tensor):
                        vv = float(v.float().mean().item())
                    elif isinstance(v, (int, float)):
                        vv = float(v)
                    else:
                        continue
                    gate_vals.append(vv)
                    log_dict[f"train/xsa_gate/layer_{int(lid)}"] = vv
                if len(gate_vals) > 0:
                    log_dict["train/xsa_gate_mean"] = float(sum(gate_vals) / len(gate_vals))
            if hasattr(raw_model, "transformer") and hasattr(raw_model.transformer, "h"):
                gamma_vals = []
                for lid, blk in enumerate(raw_model.transformer.h):
                    if not bool(getattr(blk, "xsa_forward_enabled", False)):
                        continue
                    v = None
                    if bool(getattr(blk, "xsa_forward_learnable_gamma", False)) and hasattr(blk, "xsa_forward_gamma_raw"):
                        v = torch.exp(blk.xsa_forward_gamma_raw.float())
                    else:
                        v = getattr(blk, "_xsa_forward_gamma_value", None)
                    if isinstance(v, torch.Tensor):
                        if v.ndim == 0:
                            vv = float(v.float().item())
                            gamma_vals.append(vv)
                            log_dict[f"train/xsa_gamma/layer_{int(lid)}"] = vv
                        else:
                            v_flat = v.float().view(-1)
                            if v_flat.numel() > 0:
                                vv = float(v_flat.mean().item())
                                gamma_vals.append(vv)
                                log_dict[f"train/xsa_gamma/layer_{int(lid)}"] = vv
                                for hid, hv in enumerate(v_flat):
                                    log_dict[f"train/xsa_gamma/layer_{int(lid)}/head_{int(hid)}"] = float(hv.item())
                        continue
                    elif isinstance(v, (int, float)):
                        vv = float(v)
                    else:
                        continue
                    gamma_vals.append(vv)
                    log_dict[f"train/xsa_gamma/layer_{int(lid)}"] = vv
                if len(gamma_vals) > 0:
                    log_dict["train/xsa_gamma_mean"] = float(sum(gamma_vals) / len(gamma_vals))
            wandb.log(log_dict)
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()
    tokens_seen += int(tokens_per_iter)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    for _prefetcher in list(_batch_prefetchers.values()):
        _prefetcher.close()
    destroy_process_group()
else:
    for _prefetcher in list(_batch_prefetchers.values()):
        _prefetcher.close()
