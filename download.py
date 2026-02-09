import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    if os.path.exists(filename):
        return
    # 增加超时时间，防止大文件连接中断
    response = requests.get(url, stream=True, timeout=100)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as f, tqdm(
        desc=os.path.basename(filename), total=total_size,
        unit='iB', unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024*1024):
            bar.update(f.write(data))

def main():
    repo_id = "HuggingFaceFW/fineweb-edu"
    # 核心修正：路径为 sample/10BT，去掉了之前的 data/
    base_url = f"https://hf-mirror.com/datasets/{repo_id}/resolve/main/sample/10BT"
    save_dir = "./raw_data/sample-10BT"
    os.makedirs(save_dir, exist_ok=True)

    print("正在连接镜像站下载 FineWeb-Edu 10BT 样本...")
    
    # 该子集共有 14 个文件 (000 到 013)
    files = [f"{i:03d}_00000.parquet" for i in range(14)]

    for file_name in files:
        download_url = f"{base_url}/{file_name}"
        local_path = os.path.join(save_dir, file_name)
        try:
            print(f"\n开始下载: {file_name}")
            download_file(download_url, local_path)
        except Exception as e:
            print(f"❌ 下载 {file_name} 失败: {e}")
            # 如果第一个文件就 404，说明路径还有问题，立即停止
            if "404" in str(e):
                break

if __name__ == "__main__":
    main()