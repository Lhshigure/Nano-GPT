from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from muon import Muon

#----------------------------------------------------------------------------------
# Grouped Attention (GQA) helper function
def repeat_kv(x: torch.Tensor, n_rep : int) -> torch.Tensor:
    
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return(
            # (B, Seq_Len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )   

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = getattr(config, 'n_kv_heads', config.n_head)
        self.n_rep = config.n_rep
        self.head_dim = config.head_dim
        self.block_size = config.block_size
        
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # residual network flag
        self.c_proj.RES_SCALE_INIT = 1

        self.rotary = Rotary(dim=self.head_dim, block_size=config.block_size)
        # QK-Normalization 层：用于提高超大规模模型训练的稳定性
        # 这种做法可以防止点积注意力的值在训练初期溢出
        self.q_norm = RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-5)

        # KV-Cache 仅在推理时有效,节省显存
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)


    def forward(self, x, start_pos = None):
        B, T, C = x.size()
       
        # 投影与归一化 shape = (B, T, nh, hs)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # RoPE 旋转
        # 训练：start_pos=0, T=max_length, 旋转 [0:T]
        # 推理：start_pos=current, T=1, 旋转 [start_pos]
        current_pos = start_pos if start_pos is not None else 0
        q = self.rotary(q, current_pos)
        k = self.rotary(k, current_pos)

        if start_pos is not None:
            
            # 防越界
            if start_pos + T > self.block_size:
                raise ValueError(f"KV cache overflow: start_pos({start_pos})+T({T}) > block_size({self.block_size})")
            
            # Inference
            if (self.cache_k is None or 
                self.cache_k.size(0) < B or 
                self.cache_k.device != k.device or 
                self.cache_k.dtype != k.dtype):
            
                self.cache_k = torch.empty(
                    (B, self.block_size, self.n_kv_heads, self.head_dim),
                    device=k.device, dtype=k.dtype
                )
                self.cache_v = torch.empty_like(self.cache_k)

            # 的 k/v 写入 cache
            self.cache_k[:B, start_pos:start_pos+T] = k
            self.cache_v[:B, start_pos:start_pos+T] = v

            k = self.cache_k[:B, :start_pos+T]
            v = self.cache_v[:B, :start_pos+T]

        #GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        k = k.transpose(1, 2) # (B, nh, T, hs)
        q = q.transpose(1, 2) # (B, nh, T, hs)
        v = v.transpose(1, 2) #(B, nh, T, hs)

        #Flash Attention, masked
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim)) 
    
    def _norm(self, x:torch.Tensor):
        rsqrt = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # (B, T, 1)
        #(B, T, dim) * (B, T, 1) = (B, T, dim)
        return x * rsqrt

    def forward(self, x:torch.Tensor):
        #(B, T, dim) * (dim) = (B, T, dim)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Rotary(nn.Module):
    def __init__(self, block_size: int, dim: int, base: int = 10000):
        """
        初始化 RoPE 旋转位置编码
        :param dim: 每个注意力头的维度 (head_dim)
        :param block_size: 支持的最大序列长度(context window)
        """
        super().__init__()
        self.dim = dim
        # --- 步骤 1: 计算角频率 (Angular Frequencies) ---
        # 按照公式计算频率：theta_i = 10000^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        # --- 步骤 2: 构建位置矩阵 ---
        # t 是位置索引 [0, 1, 2, ..., max_seq_len-1]
        t = torch.arange(block_size, dtype=torch.float32)
        # 使用外积计算所有位置的 theta 值: theta = t * angular_freq
        # 结果维度为 (max_seq_len, dim//2)
        theta = torch.einsum("i,j -> ij", t, inv_freq)
        # 预先计算并缓存 cos 和 sin 值，提高推理效率
        # persistent=False 表示这些缓存不会被保存在模型的 state_dict 中
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)

    
    def forward(self, x: torch.Tensor, start_pos: int = 0):
        # x: (B, T, nh, hs)
        T = x.size(1)
        cos = self.cos[None, start_pos:start_pos + T, None, :]  # (1, T, 1, hs/2)
        sin = self.sin[None, start_pos:start_pos + T, None, :]  # (1, T, 1, hs/2)

        x_fp32 = x.float()
        # 核心修改：改为偶/奇维度交错配对 (0,1), (2,3)...
        x_even = x_fp32[..., 0::2]   # (B, T, nh, hs/2)
        x_odd  = x_fp32[..., 1::2]   # (B, T, nh, hs/2)

        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos

        # 重新拼接回 (B, T, nh, hs)
        y = torch.stack((y_even, y_odd), dim=-1).flatten(-2)
        return y.type_as(x)


class SwiGLUFeedForward(nn.Module):

    def __init__(self, n_embd: int, dim_ff:int):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, dim_ff, bias=False)
        self.down_proj = nn.Linear(dim_ff, n_embd, bias=False)
        self.up_proj = nn.Linear(n_embd, dim_ff, bias=False)

        # only down_proj in Residual Stream
        self.down_proj.RES_SCALE_INIT = 1

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.gate_proj(x))
        x_v = self.up_proj(x)
        x = swish * x_v
        return self.down_proj(x)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.SwiGLUffd= SwiGLUFeedForward(config.n_embd, config.dim_ff)

    def forward(self, x, start_pos=None):
        x = x + self.attn(self.ln_1(x), start_pos)
        x = x + self.SwiGLUffd(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_kv_heads: int = 12
    n_embd: int = 768
    rms_norm_eps:float = 1e-5
    max_batch_size: int = 2 # 推理时预分配缓存的 Batch 大小
    dim_ff_override: int | None = None

    @property
    def dim_ff(self):
        if self.dim_ff_override is not None:
            return self.dim_ff_override
        hidden_dim = int(8 * self.n_embd / 3)
        multiple_of = 256
        hidden_dim = (hidden_dim + multiple_of - 1) // multiple_of * multiple_of
        return hidden_dim

    
    @property
    def n_rep(self):
        # 自动计算倍率，防止手动设置出错
        assert self.n_head % self.n_kv_heads == 0, "n_head 必须能被 n_kv_heads 整除"
        return self.n_head // self.n_kv_heads
    
    @property
    def head_dim(self):
        return self.n_embd // self.n_head


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # 1 modified: replace wpe with Rotry Embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, eps=config.rms_norm_eps),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
        
        # weight sharing scheme in small LLM
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "RES_SCALE_INIT"):
                std *= (2*self.config.n_layer) ** -0.5 
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, start_pos = None):
        B, T= idx.shape #(B, T)
        assert T <= self.config.block_size, f"Cannot forward sequence of length{T},block size is {self.config.block_size}"
          # applied Rope
        tok_emb = self.transformer.wte(idx) #(B, T, embedding)
        x = tok_emb
        # forward process
        for block in self.transformer.h:
            x = block(x, start_pos=start_pos)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # (B, T, vocab_size) -> (B * T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
        

    
    
    def configure_optimizers(self, learning_rate, device, muon_lr=0.02, muon_wd=0.01):
        # 提取所有需要梯度的参数
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        muon_params = []
        adamw_nodecay = []

        seen = set()
        for n, p in param_dict.items():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)

            # 1. 共享的 Embedding/Head 以及所有的 Bias 和 Norm 层
            # 这些参数不适合 Muon，也不应该进行 Weight Decay
            if p is self.lm_head.weight or p.ndim < 2:
                adamw_nodecay.append(p)
            else:
                # 2. 所有的 2D 权重矩阵（Attention 和 MLP 的内部权重）
                muon_params.append(p)

        # 统计信息
        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_nodecay)
        print(f"Optimizer Groups: Muon={n_muon:,} params, AdamW(no_decay)={n_adamw:,} params")

        # 创建 Muon 优化器 (处理 hidden weights)
        opt_muon = Muon(
            muon_params,
            lr=muon_lr,
            momentum=0.95,
            weight_decay=muon_wd,
            nesterov=True,
            ns_steps=5,
        )

        # 创建 AdamW 优化器 (处理 embeddings, norms, biases)
        import inspect
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and ("cuda" in device)

        adamw_kwargs = dict(lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        if fused_available:
            adamw_kwargs["fused"] = use_fused

        opt_adamw = torch.optim.AdamW(
            [{"params": adamw_nodecay, "weight_decay": 0.0}],
            **adamw_kwargs,
        )

        return [opt_muon, opt_adamw]