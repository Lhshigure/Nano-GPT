import os
import glob
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing as mp

# --- 1. 配置参数 ---
# 存储处理后 Token 的目录
local_dir = "edu_fineweb10B"
# 存储原始下载的 Parquet 文件的目录
raw_data_dir = "./raw_data/sample-10BT"
# 每个分片的 Token 数量 (100M)
shard_size = int(1e8) 

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# 初始化分词器
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] 

def tokenize(doc):
    """单个文档的分词函数，用于并行调用"""
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    # 确保在 uint16 范围内
    tokens_np = np.clip(tokens_np, 0, 2**16 - 1)
    return tokens_np.astype(np.uint16)

def write_shard(filename, tokens_np):
    """写入二进制文件"""
    tokens_np.tofile(filename)
    print(f"\n✅ Saved {filename}")

def main():
    # 扫描本地文件
    data_files = sorted(glob.glob(os.path.join(raw_data_dir, "*.parquet")))
    if not data_files:
        print(f"❌ 错误：在 {raw_data_dir} 下没找到文件，请检查下载路径。")
        return

    print(f"🚀 找到 {len(data_files)} 个本地文件，准备使用 {mp.cpu_count()} 个核心并行处理...")

    # 使用本地 parquet 加载
    fw = load_dataset("parquet", data_files=data_files, split="train", streaming=True)

    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    
    # 使用进程池加速分词
    # 我们分批次处理文档，防止内存溢出
    with mp.Pool(mp.cpu_count()) as pool:
        # 使用 imap 保持顺序并流式处理
        for tokens in tqdm(pool.imap(tokenize, fw, chunksize=16), desc="Processing Tokens", unit="docs"):
            
            # 如果当前分片放得下
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
            else:
                # 分片已满，写入文件
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                
                # 填满当前分片
                remainder = shard_size - token_count
                all_tokens_np[token_count : shard_size] = tokens[:remainder]
                write_shard(filename, all_tokens_np)
                
                # 准备下一个分片
                shard_index += 1
                # 将剩余 token 放入新 buffer
                leftover = tokens[remainder:]
                if len(leftover) > 0:
                    all_tokens_np[0 : len(leftover)] = leftover
                    token_count = len(leftover)
                else:
                    token_count = 0

    # 写入最后一个不满的分片
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_shard(filename, all_tokens_np[:token_count])

    print("🎉 所有数据处理完成！可以开始 pretrain 了。")

if __name__ == "__main__":
    main()