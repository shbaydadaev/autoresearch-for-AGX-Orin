"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: python3 train.py
"""

import os
import platform
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

IS_ARM64 = platform.machine() == "aarch64"
DEVICE_PEAK_FLOPS = float(os.getenv("AUTORESEARCH_DEVICE_PEAK_FLOPS", "0"))
ATTN_MASK_CACHE = {}
ALLOW_UNSTABLE_FP16 = os.getenv("AUTORESEARCH_ALLOW_FP16", "0").lower() in {"1", "true"}
USE_VALUE_EMBEDS = os.getenv("AUTORESEARCH_USE_VALUE_EMBEDS", "0" if IS_ARM64 else "1").lower() in {"1", "true"}
LOG_PATH = os.getenv("AUTORESEARCH_LOG_PATH", "run.log")
LOG_FILE = None
RESULTS_PATH = Path(os.getenv("AUTORESEARCH_RESULTS_PATH", "results.tsv"))
RESULT_STATUS = os.getenv("AUTORESEARCH_RESULT_STATUS", "keep")
RESULT_DESCRIPTION = os.getenv("AUTORESEARCH_RUN_DESCRIPTION", "manual run")
RESULT_RECORDED = False


def init_log_file():
    global LOG_FILE
    if LOG_FILE is None:
        LOG_FILE = open(LOG_PATH, "w", buffering=1)
    return LOG_FILE


def log_line(message=""):
    print(message)
    log_fp = init_log_file()
    print(message, file=log_fp, flush=True)


def log_step_line(message):
    print(f"\r{message}    ", end="", flush=True)
    log_fp = init_log_file()
    print(message, file=log_fp, flush=True)


def get_git_commit_short():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "nogit"


def ensure_results_header():
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
        return
    RESULTS_PATH.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n")


def append_result_row(val_bpb, memory_gb, status, description):
    global RESULT_RECORDED
    if RESULT_RECORDED:
        return
    ensure_results_header()
    with RESULTS_PATH.open("a") as f:
        f.write(
            f"{get_git_commit_short()}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"
        )
    RESULT_RECORDED = True
    log_line(f"Recorded result to {RESULTS_PATH}: status={status} val_bpb={val_bpb:.6f} mem={memory_gb:.1f}GB desc={description}")


def resolve_autocast_dtype():
    requested = os.getenv("AUTORESEARCH_AMP_DTYPE", "fp32" if IS_ARM64 else "auto").lower()
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        if IS_ARM64 and not ALLOW_UNSTABLE_FP16:
            return None
        return torch.float16
    if requested == "fp32":
        return None
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


LOW_PRECISION_DTYPE = resolve_autocast_dtype()
OPTIMIZER_MODE = os.getenv("AUTORESEARCH_OPTIMIZER", "adamw" if IS_ARM64 else "hybrid").lower()
MODEL_STORAGE_DTYPE = torch.float32


def make_autocast_context():
    if LOW_PRECISION_DTYPE is None:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=LOW_PRECISION_DTYPE)


def resolve_attention_backend():
    requested = os.getenv("AUTORESEARCH_ATTENTION_BACKEND", "eager" if IS_ARM64 else "auto").lower()
    backend = {"name": "eager", "interface": None, "warning": None}
    if requested not in {"auto", "sdpa", "kernel", "eager"}:
        raise ValueError(f"Unsupported attention backend: {requested}")
    if requested == "eager":
        return backend
    if requested == "sdpa":
        backend["name"] = "sdpa"
        return backend
    try:
        from kernels import get_kernel

        cap = torch.cuda.get_device_capability()
        repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
        backend["name"] = "kernel"
        backend["interface"] = get_kernel(repo).flash_attn_interface
        return backend
    except Exception as exc:
        if requested == "kernel":
            raise
        backend["warning"] = f"Kernel attention unavailable ({exc}); falling back to torch SDPA."
        return backend


def get_sdpa_mask(seq_len, window_size, device):
    left_window, right_window = window_size
    cache_key = (seq_len, left_window, right_window, device.type, device.index)
    mask = ATTN_MASK_CACHE.get(cache_key)
    if mask is not None:
        return mask
    q_positions = torch.arange(seq_len, device=device).unsqueeze(1)
    k_positions = torch.arange(seq_len, device=device).unsqueeze(0)
    allowed = k_positions <= q_positions
    if left_window >= 0:
        allowed = allowed & (k_positions >= (q_positions - left_window))
    if right_window >= 0:
        allowed = allowed & (k_positions <= (q_positions + right_window))
    mask = torch.zeros((seq_len, seq_len), dtype=torch.float32, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    mask = mask.unsqueeze(0).unsqueeze(0)
    ATTN_MASK_CACHE[cache_key] = mask
    return mask


def run_attention(q, k, v, window_size, backend):
    if backend["name"] == "kernel":
        return backend["interface"].flash_attn_func(q, k, v, causal=True, window_size=window_size)

    original_dtype = q.dtype
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    full_causal = window_size[1] == 0 and window_size[0] >= q.size(-2)
    if backend["name"] == "sdpa" and hasattr(F, "scaled_dot_product_attention"):
        if full_causal:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn_mask = get_sdpa_mask(q.size(-2), window_size, q.device)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
        q = q.float()
        k = k.float()
        v = v.float()
        scale = q.size(-1) ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        if full_causal:
            attn_mask = get_sdpa_mask(q.size(-2), (q.size(-2), 0), q.device)
        else:
            attn_mask = get_sdpa_mask(q.size(-2), window_size, q.device)
        scores = scores + attn_mask
        probs = F.softmax(scores, dim=-1)
        y = probs @ v
    return y.transpose(1, 2).to(dtype=original_dtype)


ATTENTION_BACKEND = resolve_attention_backend()

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.size(-1),))
    return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(dtype=x.dtype)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    if not USE_VALUE_EMBEDS:
        return False
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = run_attention(q, k, v, window_size, ATTENTION_BACKEND)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Keep the largest tables in reduced precision on small Jetson memory budgets.
        self.transformer.wte.to(dtype=MODEL_STORAGE_DTYPE)
        for ve in self.value_embeds.values():
            ve.to(dtype=MODEL_STORAGE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos = cos.to(dtype=MODEL_STORAGE_DTYPE)
        sin = sin.to(dtype=MODEL_STORAGE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        log_line(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        if OPTIMIZER_MODE == "adamw":
            param_groups = [
                dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-8, weight_decay=0.0),
                dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-8, weight_decay=0.0),
                dict(params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-8, weight_decay=0.0),
                dict(params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-8, weight_decay=0.0),
                dict(params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-8, weight_decay=0.0),
                dict(params=matrix_params, lr=matrix_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-8, weight_decay=weight_decay),
            ]
            optimizer = torch.optim.AdamW(param_groups)
            for group in optimizer.param_groups:
                group["kind"] = "adamw"
                group["initial_lr"] = group["lr"]
            log_line("Optimizer mode: AdamW-only")
            return optimizer
        if OPTIMIZER_MODE != "hybrid":
            raise ValueError(f"Unsupported AUTORESEARCH_OPTIMIZER value: {OPTIMIZER_MODE}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        log_line("Optimizer mode: Muon + AdamW hybrid")
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 30
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    step_t = float(step_t)
    lr_t = float(lr_t)
    beta1_t = float(beta1_t)
    beta2_t = float(beta2_t)
    eps_t = float(eps_t)
    wd_t = float(wd_t)
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum_t = float(momentum_t)
    lr_t = float(lr_t)
    wd_t = float(wd_t)
    beta2_t = float(beta2_t)
    # Nesterov momentum
    momentum_buffer.lerp_(stacked_grads, 1.0 - momentum_t)
    g = stacked_grads.lerp(momentum_buffer, momentum_t)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1.0 - beta2_t)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t
    wd = wd_t
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            state['step'], group['lr'], group['betas'][0],
                            group['betas'][1], group['eps'], group['weight_decay'])

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        group["momentum"], group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5, group["weight_decay"],
                        group["beta2"] if group["beta2"] is not None else 0.0, group["ns_steps"], red_dim)
        for param, updated in zip(params, stacked_params.unbind(0)):
            param.copy_(updated)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture (Jetson AGX Orin / R35-friendly defaults)
ASPECT_RATIO = 32       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 64           # smaller heads fit Orin memory and SDPA fallback better
WINDOW_PATTERN = "L"    # full causal attention; simplest and most reliable on Jetson

# Optimization
TOTAL_BATCH_SIZE = 2**14  # 16K tokens per optimizer step
EMBEDDING_LR = 0.02       # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.003    # learning rate for lm_head (Adam)
MATRIX_LR = 0.003         # learning rate for matrix parameters
SCALAR_LR = 0.02          # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.01       # modest weight decay for the safe Jetson baseline
ADAM_BETAS = (0.9, 0.95)  # Adam beta1, beta2
WARMUP_RATIO = 0.05       # short warmup helps smaller Jetson runs
WARMDOWN_RATIO = 0.20     # preserve more of the run at peak LR
FINAL_LR_FRAC = 0.1       # do not decay all the way to zero on short runs

# Model size
DEPTH = 6                 # number of transformer layers
DEVICE_BATCH_SIZE = 16    # per-device batch size (reduce if OOM)
GRAD_CLIP_NORM = 1.0      # gradient clipping keeps the first Jetson run stable


def maybe_compile_model(model):
    requested = os.getenv("AUTORESEARCH_USE_COMPILE", "0" if IS_ARM64 else "1").lower()
    if requested not in {"0", "1", "false", "true"}:
        raise ValueError(f"Unsupported AUTORESEARCH_USE_COMPILE value: {requested}")
    enabled = requested in {"1", "true"}
    if not enabled or not hasattr(torch, "compile"):
        return model, False
    try:
        return torch.compile(model, dynamic=False), True
    except Exception as exc:
        log_line(f"torch.compile unavailable ({exc}); continuing without compilation.")
        return model, False


def build_model_for_device(config, device):
    try:
        with torch.device("meta"):
            model = GPT(config)
        if hasattr(model, "to_empty"):
            model.to_empty(device=device)
        else:
            model = GPT(config).to(device)
        return model, "meta"
    except Exception as exc:
        log_line(f"Meta-device init unavailable ({exc}); falling back to direct CUDA init.")
        return GPT(config).to(device), "direct"

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if not torch.cuda.is_available():
    raise RuntimeError("autoresearch requires a CUDA-capable NVIDIA GPU. Jetson container must be started with --runtime nvidia or --gpus all.")
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_dtype = LOW_PRECISION_DTYPE
autocast_ctx = make_autocast_context()

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
log_line(f"Vocab size: {vocab_size:,}")
log_line(f"Autocast dtype: {autocast_dtype if autocast_dtype is not None else 'torch.float32'}")
log_line(f"Attention backend: {ATTENTION_BACKEND['name']}")
if ATTENTION_BACKEND["warning"] is not None:
    log_line(f"Attention note: {ATTENTION_BACKEND['warning']}")
log_line(f"Optimizer mode: {OPTIMIZER_MODE}")
log_line(f"Value embeddings: {'enabled' if USE_VALUE_EMBEDS else 'disabled'}")
if IS_ARM64 and os.getenv("AUTORESEARCH_AMP_DTYPE", "").lower() == "fp16" and not ALLOW_UNSTABLE_FP16:
    log_line("Autocast note: fp16 requested on Jetson, but the stable baseline forces fp32 unless AUTORESEARCH_ALLOW_FP16=1.")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
log_line(f"Model config: {asdict(config)}")

model, init_mode = build_model_for_device(config, device)
log_line(f"Model init mode: {init_mode}")
model.init_weights()

param_counts = model.num_scaling_params()
log_line("Parameter counts:")
for key, value in param_counts.items():
    log_line(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
log_line(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

model, compile_enabled = maybe_compile_model(model)
log_line(f"torch.compile: {'enabled' if compile_enabled else 'disabled'}")

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

log_line(f"Time budget: {TIME_BUDGET}s")
log_line(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f):
        append_result_row(0.0, torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, "crash", f"{RESULT_DESCRIPTION} (nan loss)")
        log_line(f"FAIL: train_loss became NaN at step {step}")
        exit(1)
    if math.isinf(train_loss_f):
        append_result_row(0.0, torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, "crash", f"{RESULT_DESCRIPTION} (inf loss)")
        log_line(f"FAIL: train_loss became Inf at step {step}")
        exit(1)
    if train_loss_f > 100:
        append_result_row(0.0, torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, "crash", f"{RESULT_DESCRIPTION} (loss explosion)")
        log_line(f"FAIL: train_loss exploded to {train_loss_f:.6f} at step {step}")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    if DEVICE_PEAK_FLOPS > 0:
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / DEVICE_PEAK_FLOPS
    else:
        mfu = 0.0
    remaining = max(0, TIME_BUDGET - total_training_time)

    status_line = f"step {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s"
    log_step_line(status_line)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 0.0
if total_training_time > 0 and DEVICE_PEAK_FLOPS > 0:
    steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / DEVICE_PEAK_FLOPS
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

log_line("---")
log_line(f"val_bpb:          {val_bpb:.6f}")
log_line(f"training_seconds: {total_training_time:.1f}")
log_line(f"total_seconds:    {t_end - t_start:.1f}")
log_line(f"peak_vram_mb:     {peak_vram_mb:.1f}")
log_line(f"mfu_percent:      {steady_state_mfu:.2f}")
log_line(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
log_line(f"num_steps:        {step}")
log_line(f"num_params_M:     {num_params / 1e6:.1f}")
log_line(f"depth:            {DEPTH}")
append_result_row(val_bpb, peak_vram_mb / 1024, RESULT_STATUS, RESULT_DESCRIPTION)
