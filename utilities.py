import json
import os
import torch
import torch.nn.functional as F

def iterate_examples(split):
    # 指向存放文件的子文件夹
    filename = os.path.join("hellaswag", f"hellaswag_{split}.jsonl") 
    if not os.path.exists(filename):
        raise FileNotFoundError(f"找不到文件: {filename}，请确认文件是否在 hellaswag 文件夹内")
    
    with open(filename, "r") as f:
        for line in f:
            yield json.loads(line)

# 2. 修改函数签名，允许传入外部 tokenizer
def render_example(example, tokenizer):
    ctx = example["ctx"]
    endings = example["endings"]
    label = example["label"] 
    
  
    ctx_tokens = tokenizer.encode(ctx)
    toks = []
    mask = []
    
    for end in endings:
        # 编码选项部分
        end_tokens = tokenizer.encode(" " + end)
        
        full_tokens = ctx_tokens + end_tokens
        toks.append(torch.tensor(full_tokens))
        
        # 建立 mask：背景部分为 0，选项部分为 1
        m = [0] * len(ctx_tokens) + [1] * len(end_tokens)
        mask.append(torch.tensor(m))
    
    # 填充逻辑保持不变
    toks_padded = torch.nn.utils.rnn.pad_sequence(toks, batch_first=True, padding_value=0)
    mask_padded = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)
    
    return example["ind"], toks_padded, mask_padded, label

# get_most_likely_row 逻辑与分词器无关，无需修改
def get_most_likely_row(tokens, mask, logits):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()
    
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_tokens = shift_tokens.view(-1)
    
    losses = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
    losses = losses.view(shift_tokens.size())
    
    masked_losses = losses * shift_mask
    sum_loss = masked_losses.sum(dim=1)
    count = shift_mask.sum(dim=1)
    avg_loss = sum_loss / count
    
    pred = torch.argmin(avg_loss)
    return pred.item()