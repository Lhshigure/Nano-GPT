import pandas as pd
import matplotlib.pyplot as plt

def generate_dashboard(csv_path='log/training_stats.csv'):
    # --- OpenAI 官方基准数据 ---
    openai_val_loss_gpt2 = 3.2924
    openai_hella_gpt2 = 0.294463
    openai_hella_gpt3 = 0.337

    # 设置绘图风格 (更现代的简洁风格)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 确保列为数值类型
    cols_to_fix = ['train_loss', 'val_loss', 'hella_acc']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 忽略前 1000 step
    df = df[df['step'] >= 1000].copy()
    
    # 计算 EMA
    df['train_loss_ema'] = df['train_loss'].ewm(span=200).mean()
    
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # --- 左图：Loss 曲线对比 ---
    # 训练 Loss：浅蓝色原线 + 深蓝色 EMA 加粗线
    ax1.plot(df['step'], df['train_loss'], color='#3498db', alpha=0.15, label='Train Loss (Raw)')
    ax1.plot(df['step'], df['train_loss_ema'], color='#2980b9', linewidth=2.5, label='Train Loss (EMA)')
    
    # 验证 Loss：纯鲜红实线 (去掉了点)
    val_df = df.dropna(subset=['val_loss'])
    ax1.plot(val_df['step'], val_df['val_loss'], color='#e74c3c', linewidth=2.5, label='My Val Loss')
    
    # 基准线：黑色虚线
    ax1.axhline(y=openai_val_loss_gpt2, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.8, 
                label=f'OpenAI GPT-2 Baseline ({openai_val_loss_gpt2})')
    
    ax1.set_title('Cross Entropy Loss: Convergence Analysis', fontsize=15, fontweight='bold', pad=20)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(frameon=True, fontsize=11)

    # --- 右图：HellaSwag 准确率对比 ---
    hella_df = df.dropna(subset=['hella_acc'])
    # 绿色实线 (也去掉了点，保持风格统一)
    ax2.plot(hella_df['step'], hella_df['hella_acc'], color='#27ae60', linewidth=3, label='My HellaSwag Acc')
    
    # OpenAI 基准线
    ax2.axhline(y=openai_hella_gpt2, color='#2980b9', linestyle='--', linewidth=1.5, label=f'GPT-2 Baseline ({openai_hella_gpt2:.4f})')
    ax2.axhline(y=openai_hella_gpt3, color='#c0392b', linestyle='--', linewidth=1.5, label=f'GPT-3 Baseline ({openai_hella_gpt3:.4f})')
    ax2.axhline(y=0.25, color='#95a5a6', linestyle=':', label='Random Chance (0.25)')
    
    ax2.set_title('Zero-shot Reasoning: HellaSwag Accuracy', fontsize=15, fontweight='bold', pad=20)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(frameon=True, fontsize=11)

    # 自动调整布局，防止标题重叠
    plt.tight_layout()
    plt.savefig('training_dashboard.png', dpi=200, bbox_inches='tight')
    print("✨ 专业版图表已保存为：training_dashboard_pro.png")

# 运行
generate_dashboard()