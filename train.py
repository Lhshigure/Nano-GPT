import math
import numpy as np
import torch
from Nanogpt_modified import GPT, GPTConfig
from utilities import iterate_examples, render_example, get_most_likely_row
import warnings
import tiktoken

warnings.filterwarnings("ignore", category=FutureWarning)

def generate_sample(model, tokenizer, prompt, max_new_tokens=30, device='cuda'):
    model.eval()
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(tokenizer, "eot_token", None)  # tiktoken 常见
    if eos_id is None:
        eos_id = 50256  # GPT-2 endoftext 兜底

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = tokens[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
            if next_token.item() == eos_id:
                break

    out = tokenizer.decode(tokens[0].tolist())
    model.train()
    return out

def load_tokens(filename):
    npt = np.fromfile(filename, dtype=np.uint16) # 读取原始二进制
    ptt = torch.tensor(npt.astype(np.int64), dtype=torch.long) # 转为 long
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
    
    def reset(self):
        # state init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T)  # targets
        self.current_position += B * T *self.num_processes
        # out of bounds, rest
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x.to(device), y.to(device)
#---------------------------------------------------------------------------------------------------
# simple run:
# python train_gpt2.py
# DDP run:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

import time
import os
# 多卡GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = "cuda" if "cuda" in device else "cpu"
num_return_sequences = 5
max_length = 30

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# 设置 DDP (Distributed Data Parallel)
# torchrun 命令会自动设置环境变量 RANK, LOCAL_RANK, 和 WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # 这是一个 ddp 运行吗？
if ddp:
    # 使用 DDP 模式
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 这个进程将负责日志记录、保存检查点等
else:
    # 普通的单卡运行模式
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # 尝试自动检测设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 #    GPT2(124m):2**19, ~0.5m, in number of tokens
B = 16 # micro_step          GPT2(124m):16
T = 1024  #                  GPT2(124m):1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T *ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

print("I am GPU", ddp_rank)
print("OK!")

train_loader = DataLoaderLite(B = B, T = T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B = B, T = T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high') # mixed precision

#----------------------------------------------------------------------------------------------------------------------------------------------------
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

config = GPTConfig(
    vocab_size=50304, 
    block_size=T, 
    max_batch_size=B, 
    n_layer=12,
    n_head=12,
    n_kv_heads=12,     # QA 设置
    n_embd=768
)
# create model
model = GPT(config)   
model.to(device)
use_compile = True
if use_compile:
    model = torch.compile(model) # model = torch.compile(model) window下不完全支持！

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # all reduce
raw_model = model.module if ddp else model

# GPT-3 learning rate setting, Dataset: edu_fineweb10B
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 712
max_steps = 19073

#----------------------------------------------------------------
# wsd + cosine
def get_lr_wsd_cosine(it, decay_start_pct=0.8):
    # 1. Warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    # 2. Stable
    decay_start_step = int(max_steps * decay_start_pct)
    if it < decay_start_step:
        return max_lr
    
    # 3. Cosine Cooldown (核心改动)
    decay_steps = max_steps - decay_start_step
    it_in_decay = it - decay_start_step
    # 计算余弦系数
    coeff = 0.5 * (1.0 + math.cos(math.pi * it_in_decay / decay_steps))
    return min_lr + coeff * (max_lr - min_lr)


def get_lr_cos(it):
    # 1) linear warmp for warmup_iter steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps = 1e-8)
optimizers = raw_model.configure_optimizers(learning_rate=6e-4, device=device)
opt_muon, opt_adamw = optimizers

#-------------记录训练数据-----------------------------
# --- 新增：断点续训加载逻辑 ---
import csv
log_dir = "log"  # save file
start_step = 0
os.makedirs(log_dir, exist_ok=True)
checkpoint_path = os.path.join(log_dir, "model_latest.pt")
#保存最佳loss model
best_val_loss = float('inf')

if os.path.exists(checkpoint_path):
    if master_process:
        print(f"正在从 {checkpoint_path} 恢复训练...")
    
    # 1. 加载到正确设备 
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 2. 获取状态字典
    state_dict = checkpoint["model"]
    
    # 3. 恢复权重
    raw_model.load_state_dict(state_dict, strict=False)
    
    # 4. 恢复两个优化器的状态
    # 确保在 resume 之前已经初始化好了 opt_muon 和 opt_adamw
    opt_muon.load_state_dict(checkpoint['optimizer_muon'])
    opt_adamw.load_state_dict(checkpoint['optimizer_adamw'])
    
    # 5. 更新起始步数
    start_step = checkpoint['step'] + 1
    
    # 6. 恢复最佳验证损失
    # 增加 val_loss 存在的判断，防止旧 checkpoint 格式报错
    if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
        best_val_loss = checkpoint['val_loss']
    
    if master_process:
        print(f"✅ 断点恢复成功！KV-Cache 已重置为全零，将从 Step {start_step} 继续训练。")
    

csv_log_path = os.path.join(log_dir, "training_stats.csv")
if master_process:

    if start_step == 0:
        with open(csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            # 核心指标：步骤、两种Loss、两种学习率、梯度范数、吞吐量、评测准确率
            writer.writerow([
                "step", "train_loss", "val_loss", "val_ppl", 
                "lr_adamw", "lr_muon", "norm", "dt_ms", 
                "tokens_per_sec", "hella_acc"
            ])


#train loop
import contextlib
#----------------------------------------------------------
for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss:
    if step % 250 ==0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss /val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            # 计算 PPL
            val_ppl = math.exp(val_loss_accum.item())
            print(f"validation loss: {val_loss_accum.item():.4f} | ppl: {val_ppl:.4f}")
    
    # -----------------------------------------------------------------------------
    # HellaSwag 评估逻辑
    if ((step > 0 and step % 500 == 0) or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        
        # iterate_examples("val") 会读取 HellaSwag 的验证集数据
        for i, example in enumerate(iterate_examples("val")):
            
            # 多卡并行 (DDP) 逻辑：每个进程只处理自己那部分数据
            if i % ddp_world_size != ddp_rank:
                continue
                
            # 将原始例子渲染为 tokens、mask 和正确答案 label
            _, tokens, mask, label = render_example(example, tokenizer)
            tokens = tokens.to(device)
            mask = mask.to(device)
            
            # 获取模型预测
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                    
                # get_most_likely_row 会对比 4 个选项中哪一个概率最高
                pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
                
        # 如果是多卡并行，需要将所有进程的统计结果相加
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
            
        # 计算准确率并打印
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")

    
    # -----------------------------------------------------------------------------
    # once a while generate from model (except step 0)
    if step % 500 == 0 or last_step:
        if master_process:
            test_prompt = "The most important thing in life is"
            sample_text = generate_sample(raw_model, tokenizer, test_prompt, device=device)
        
            print(f"--- Step {step} Sample ---")
            print(f"Prompt: {test_prompt}")
            print(f"Generated: {sample_text}")
            
            with open(os.path.join(log_dir, "samples.txt"), "a", encoding="utf-8") as f:
                f.write(f"--- STEP {step} ---\n")
                f.write(f"PROMPT: {test_prompt}\n")
                f.write(f"RESULT: {sample_text}\n\n")
    # training loop
    model.train()
    # 对两个优化器都要清零
    max_lr_muon = 0.02   #Muon 峰值学习率
    for opt in optimizers:
        opt.zero_grad(set_to_none=True) 

    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        ctx = model.no_sync() if ddp and micro_step < grad_accum_steps - 1 else contextlib.nullcontext()
        with ctx:
            logits, loss = model(x, y)
        
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr_wsd_cosine(step)

    lr_multiplier = lr / max_lr if max_lr > 0 else 0.0  #max AdamW_lr = max_lr
    muon_lr = max_lr_muon * lr_multiplier

    # AdamW 学习率
    for param_group in opt_adamw.param_groups:
        param_group['lr'] = lr

    # Muon 学习率
    for param_group in opt_muon.param_groups:
        param_group['lr'] = muon_lr

    for opt in optimizers:
        opt.step()

    # 同步 GPU，确保时间统计准确
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) # time difference in miliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step} | loss:{loss_accum.item()} | lr:{lr:.6f} | norm:{norm:.4f} | dt:{dt*1000:2f}ms | tokens/sec:{tokens_per_sec}")

        # 保存间隔
        if step > 0 and (step % 1000 == 0 or last_step):
            if master_process:
                # 构造 Checkpoint 字典
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item() if 'val_loss_accum' in locals() else None,
                    'optimizer_muon': opt_muon.state_dict(),
                    'optimizer_adamw': opt_adamw.state_dict(),
                }
                
                # 1. 始终保存为 latest，用于下次断点续训（自动覆盖旧的 latest）
                latest_path = os.path.join(log_dir, "model_latest.pt")
                torch.save(checkpoint, latest_path)
                print(f"💾 已更新最新进度至 {latest_path}")

        
        # 准备本步要记录的数据
        # 准备评估数据：只有在特定步数才填入数值，否则留空 ""
        is_val_step = (step % 250 == 0 or last_step)
        is_hella_step = ((step > 0 and step % 500 == 0) or last_step)
        
        v_loss = val_loss_accum.item() if is_val_step else ""
        v_ppl = math.exp(val_loss_accum.item()) if is_val_step else ""
        h_acc = acc_norm if (is_hella_step and 'acc_norm' in locals()) else ""

        with open(csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step, 
                loss_accum.item(), 
                v_loss, 
                v_ppl,           
                lr, 
                muon_lr, 
                norm.item(), 
                dt * 1000,       
                tokens_per_sec, 
                h_acc            
            ])

if ddp:
    destroy_process_group()