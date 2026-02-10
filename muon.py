import torch
from torch.optim import Optimizer

@torch.no_grad()
def _newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-10):
    """
    使用 Newton-Schulz 迭代近似矩阵正交化。
    该算子是 Muon 核心，用于寻找最接近梯度的正交矩阵。
    """
    # 关键点 1: 必须使用 float32 进行迭代，否则在 bfloat16 下会因精度不足迅速崩溃
    dtype = G.dtype
    X = G.to(torch.float32)
    
    # 关键点 2: 初始缩放。
    # 确保矩阵的频谱范数处于收敛区间内
    X /= (X.norm() + eps)
    
    m, n = X.shape
    if m >= n:
        S = X.T @ X
        I = torch.eye(n, device=X.device, dtype=torch.float32)
        Z = I
        for _ in range(steps):
            # 迭代公式: Z = 0.5 * Z * (3I - S @ Z @ Z)
            # 这是在近似计算 (X^T X)^(-1/2)
            Z = 0.5 * Z @ (3.0 * I - S @ Z @ Z)
        return (X @ Z).to(dtype)
    else:
        # 处理宽矩阵的情况
        S = X @ X.T
        I = torch.eye(m, device=X.device, dtype=torch.float32)
        Z = I
        for _ in range(steps):
            Z = 0.5 * (3.0 * I - Z @ Z @ S) @ Z
        return (Z @ X).to(dtype)

class Muon(Optimizer):
    """
    Muon 优化器 (Modified Unitary Optimizer)
    
    针对 2D 权重矩阵设计，通过正交化更新取代传统的 AdamW 缩放。
    适用于：Linear 层权重。
    不适用于：Embedding, Norm 层, Bias, 1D 参数。
    """
    def __init__(
        self, 
        params, 
        lr: float = 0.02, 
        momentum: float = 0.95, 
        weight_decay: float = 0.01, 
        nesterov: bool = True,
        ns_steps: int = 5
    ):
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay, 
            nesterov=nesterov,
            ns_steps=ns_steps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Muon 仅支持 2D 权重 (矩阵)
                if p.ndim != 2:
                    raise ValueError(f"Muon 仅支持 2D 参数，当前参数维度为 {p.ndim}")

                g = p.grad
                state = self.state[p]

                # 1. 解耦权重衰减 (Decoupled Weight Decay)
                if group['weight_decay'] != 0:
                    p.mul_(1.0 - group['lr'] * group['weight_decay'])

                # ------------------- 核心修改开始 -------------------
                # 2. 动量更新 (强制使用 FP32)
                if 'momentum_buffer' not in state:
                    # [修改点 1] 初始化 buffer 时指定 dtype=torch.float32
                    state['momentum_buffer'] = torch.zeros_like(g, dtype=torch.float32)
                
                buf = state['momentum_buffer'] # 这是一个 fp32 tensor
                
                # [修改点 2] 将当前梯度临时转为 fp32 参与计算
                g_fp32 = g.to(torch.float32)

                # 在 fp32 精度下进行动量累积：buf = momentum * buf + g
                buf.mul_(group['momentum']).add_(g_fp32)

                # Nesterov 动量处理 (全在 fp32 下计算)
                if group['nesterov']:
                    g_eff = g_fp32.add(buf, alpha=group['momentum'])
                else:
                    g_eff = buf

                # 3. Newton-Schulz 正交化
                # 注意：_newton_schulz_orthogonalize 内部通常也需要 fp32 计算
                # 由于传入的 g_eff 已经是 fp32，这里直接传进去即可
                update = _newton_schulz_orthogonalize(g_eff, steps=group['ns_steps'])

                # 4. 参数更新
                # [修改点 3] 将计算好的 fp32 update 转回参数原始类型 (如 bf16) 再更新
                p.add_(update.to(p.dtype), alpha=-group['lr'])
                # ------------------- 核心修改结束 -------------------

        return loss