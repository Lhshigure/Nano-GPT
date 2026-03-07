# Modernizing GPT-2 Scale Models Under a 10B-Token Training Budget

## Abstract

This report investigates whether modern architectural and optimization strategies improve performance at the GPT-2 (124M) scale under a strict 10B-token single-pass training budget. Starting from a GPT-2–style configuration, we introduce Rotary Positional Embeddings (RoPE), SwiGLU activation, RMSNorm, Query-Key normalization, and a hybrid Muon + AdamW optimization scheme, while keeping the parameter count fixed.

All models are trained under the same 10B-token single-pass setting. Under this constraint, the modernized model achieves lower validation loss and similar HellaSwag performance (reported as completion-style **acc_norm**) compared to commonly reported GPT-2 references, without increasing model size or training tokens. We also run a controlled learning-rate schedule ablation (Warmup + 60% stable + 40% cosine vs. Warmup + pure cosine) under identical settings.

Beyond final metrics, we analyze training dynamics and observe that linguistic fluency emerges earlier than improvements on HellaSwag. The schedule ablation further shows that better perplexity does not necessarily translate into better HellaSwag accuracy at this scale. Overall, these results suggest that modern architecture, structured optimization, and learning-rate scheduling can improve GPT-2–scale training outcomes under limited compute.


## Introduction

Since GPT-2, many techniques have become standard in large language models, including RoPE, SwiGLU, RMSNorm, and improved training recipes. Under a limited training budget, it is valuable to measure how much these modern choices help at the GPT-2 scale.

In this project, we start from a GPT-2–scale configuration (124M; 12 layers, 12 heads, 768 hidden) and build a modernized model with the same parameter count. We introduce RoPE, RMSNorm, SwiGLU, and QK-norm. We also adopt a hybrid optimizer: Muon is applied to 2D weight matrices, and AdamW is used for non-matrix parameters.

All models are trained under a fixed compute budget: FineWeb-Edu 10B tokens in a strict single pass (19,073 steps), on 4× RTX 5090 GPUs.

To isolate the effect of learning-rate scheduling under the same 10B-token constraint, we compare two schedules while keeping all other settings identical:

- **Warmup–Plateau–Cosine (WPC):** warmup, then a constant high-learning-rate plateau (60% of steps), followed by cosine decay (40%).
- **Warmup–Cosine (Cosine):** warmup, then cosine decay for the remaining steps.

Under this setup, WPC achieves the best validation perplexity, while Cosine reaches a slightly higher peak HellaSwag accuracy. Final results are shown below (HellaSwag is reported as completion-style length-normalized accuracy, **acc_norm**).

| Model / Schedule | Val Loss | Val PPL | HellaSwag acc_norm (peak) | HellaSwag acc_norm (final) |
|---|---:|---:|---:|---:|
| Modern-NanoGPT (WPC) | **3.1290** | **22.85** | 29.52% | 29.43% |
| Modern-NanoGPT (Cosine) | 3.1505 | 23.35 | **29.56%** | **29.46%** |

*HellaSwag is evaluated with a completion-style likelihood ranking script: each candidate ending is scored by the length-normalized negative log-likelihood over completion tokens (acc_norm), rather than prompting the model to output a discrete option (A/B/C/D). Reported GPT-2 numbers from prior open-source sources are included only as rough references and may not be directly comparable due to dataset and evaluation differences.*

Overall, these results suggest that modern architectural and optimization choices improve language modeling performance at the GPT-2 scale under a strict 10B-token budget. The schedule ablation further indicates a trade-off: keeping a high learning rate longer improves final perplexity, while pure cosine yields slightly earlier and marginally higher HellaSwag peaks.



**Contributions**

- Implemented a modernized GPT-2–scale training stack (RoPE/RMSNorm/SwiGLU/QK-Norm) while keeping ~124M params.

- Integrated a hybrid optimizer (Muon for 2D matrices + AdamW for others) into a NanoGPT-style codebase.

- Designed and ran a controlled learning-rate schedule ablation (plateau+cosine vs pure cosine) under the same 10B-token single-pass budget.

- Analyzed training dynamics: fluency emerges early, while HellaSwag improvements lag and correlate more with late-stage refinement.


## Experimental Setup

### 3.1 Model Architecture

We adopt a GPT-2 small–scale transformer architecture (124M parameters) as the baseline configuration, consisting of 12 layers, 12 attention heads, and a hidden dimension of 768. The context length follows the standard GPT-2 setting.

To modernize the architecture while keeping the parameter count comparable, we introduce the following modifications:

* **RoPE replaces absolute positional embeddings.**
* **RMSNorm replaces LayerNorm.**
* **SwiGLU replaces GELU in the feed-forward network.**
* **QK-norm is applied to stabilize attention scores.**
* **We use standard multi-head attention (no GQA; n_kv_heads = n_head).**

Tokenization uses GPT-2 BPE via tiktoken. The total number of parameters remains close to that of the original GPT-2 (124 million), ensuring fairness in architectural comparisons.

We also implement KV-cache for efficient inference during generation (this does not change the training objective or loss).

### 3.2 Optimization

We adopt a hybrid optimization strategy tailored to transformer parameters. Large 2D weight matrices (e.g., attention projections and MLP weights) often benefit from structure-aware updates. Using AdamW for these matrices can lead to imbalanced, ill-conditioned updates because AdamW is element-wise and does not control the update geometry at the matrix level. To address this, we apply Muon (Keller Jordan et al., 2024), which approximately orthogonalizes the update direction using a Newton–Schulz iteration. Given a gradient matrix **G**, Muon iteratively transforms it as:

$$
X_{k+1} = \frac{1}{2} X_k (3I - X_k^\top X_k), \quad \text{with } X_0 = G.
$$

In practice, a small fixed number of iterations (e.g., 5) is sufficient. This provides a cheap approximation to orthogonalizing the matrix update without an SVD, helping keep updates balanced across directions and improving training stability. For remaining parameters (e.g., biases and RMSNorm scales), we use AdamW, which is stable and well-understood for non-matrix-structured tensors.These settings are shared by both learning-rate schedules.

### 3.3 Learning Rate Schedule

We run a controlled learning-rate schedule ablation under the same 10B-token single-pass budget. Unless noted otherwise, all settings (architecture, optimizer split, batch size, data, and total steps) are kept identical, and only the learning-rate schedule differs.

Let the total training steps be $T = 19{,}073$, with $t \in [0, T)$. Both schedules use the same linear warmup for the first 712 steps. After warmup, we compare:

**(1) Warmup + 60% stable + 40% cosine decay (WSD+Cosine).**  
After warmup, we keep the learning rate at $\text{max\_lr}$ until step $t_\text{stable} = 11{,}444 \approx 0.6T$, then apply cosine decay from $\text{max\_lr}$ to $\text{min\_lr}$ for the remaining steps:

- Warmup: $t \in [0, 712)$ (linear increase to $\text{max\_lr}$)
- Stable: $t \in [712, 11{,}444)$ (constant at $\text{max\_lr}$)
- Cosine: $t \in [11{,}444, T)$ (cosine decay to $\text{min\_lr}$)

This schedule is designed to preserve a longer high-learning-rate phase under a short, fixed token budget, and use the final portion for refinement.

**(2) Warmup + pure cosine decay (Cosine).**  
After warmup, we immediately apply cosine decay from $\text{max\_lr}$ to $\text{min\_lr}$ for the rest of training:

- Warmup: $t \in [0, 712)$
- Cosine: $t \in [712, T)$

This schedule decays earlier, which can improve early stability and convergence, but may reduce the time spent exploring at a high learning rate under a fixed budget.

We use the same $\text{max\_lr}$ and $\text{min\_lr}$ values for both schedules (per-parameter-group values are unchanged); only the scheduling shape differs.


### 3.4 Dataset

All models are trained on the FineWeb-Edu 10B-token subset under a single-pass training budget. The dataset provides sufficiently large-scale web text for language modeling while remaining compatible with limited compute resources. Data is tokenized using GPT-2 BPE.These settings are shared by both learning-rate schedules.

### 3.5 Evaluation Metrics: HellaSwag

To evaluate semantic continuation beyond simple token-level prediction, we employ the HellaSwag benchmark (Zellers et al., 2019). The evaluation is conducted in a zero-shot setting, ensuring that the model relies solely on knowledge acquired during pre-training without task-specific fine-tuning.

**Evaluation style.** We use completion-style likelihood ranking rather than prompting the model to output a discrete option (A/B/C/D). Small models are often sensitive to prompt formatting and may not reliably follow an “output-the-letter” instruction; likelihood ranking avoids this format dependency and matches the training objective of a decoder-only language model. Concretely, we compute the token-level negative log-likelihood on the completion region only and choose the candidate with the lowest length-normalized loss (acc_norm). Note that different evaluation styles (e.g., Eleuther harness multiple-choice formatting vs. completion-style scoring) can yield different absolute numbers, so we report completion-style acc_norm consistently.

The HellaSwag task presents a significant challenge for language models as it focuses on *adversarial filtering*, where machine-generated wrong endings are designed to be plausible to shallow models but obvious to humans.

For each evaluation instance, the model is presented with a context and four potential completion candidates. The evaluation mechanism is based on log-likelihood comparison:

For a given context $C$ and a set of candidate endings ${E_1, E_2, E_3, E_4}$, the model calculates the conditional cross-entropy loss for each completion.

The model's prediction is determined by selecting the ending with the lowest average loss:

$$
\hat{E} = \arg\min_{E_i} \text{Loss}(E_i \mid C)
$$

We specifically report **accuracy_norm (acc_norm)** as our primary metric. Unlike raw accuracy, acc_norm applies length normalization, preventing the model from being biased toward shorter, high-probability endings.

Concretely, if we define the (length-normalized) score as average negative log-likelihood over completion tokens:

$$
\text{Score}(E_i) =
\frac{1}{|E_i|}
\sum_{t \in E_i}
-\log P(t \mid C, t_{<t}),
$$

then the prediction is $\hat{E} = \arg\min_{E_i} \text{Score}(E_i)$.

This approach provides a robust measurement of the model’s ability to capture semantic coherence and logical continuity rather than preferring short, frequent phrases.

## Experiments and Results

This section analyzes the training dynamics and final performance of Modern-NanoGPT under a fixed 10B-token single-pass budget. We compare two learning-rate schedules (WSD+Cosine vs. pure Cosine) under identical settings, and examine both quantitative metrics and qualitative generation behavior over training.
We examine both quantitative metrics and qualitative generation behavior over training.

### 4.1 Training Dynamics and Sample Evolution

### 4.1 Training Dynamics and Sample Evolution

Training progresses smoothly over 19,073 steps without instability. Figure 1 compares the learning curves under the two schedules. In the mid phase, the pure Cosine schedule slightly improves earlier, while WSD+Cosine continues to improve later and finishes with lower validation loss. HellaSwag acc_norm follows a similar upward trend for both runs: Cosine reaches a marginally higher peak, but the final accuracies are very close.

Qualitative generation samples show a clear transition from incoherent token-level output in early training to syntactically correct and semantically meaningful continuations in later stages. In the first few thousand steps, outputs contain fragmented tokens and unstable sentence structure. By around 5k–10k steps, sentences become grammatically correct with coherent themes. In the final stage of training (15k–19k steps), generations demonstrate stable paragraph-level structure and improved logical continuity.

Overall, these observations align with the steady reduction in validation loss and indicate consistent convergence under both learning-rate schedules.

### 4.2 Final Evaluation

Under the 10B-token single-pass setting, we report final validation metrics and completion-style HellaSwag acc_norm for both learning-rate schedules:

| Schedule | Val Loss (final) | Val PPL (final) | HellaSwag acc_norm (peak) | HellaSwag acc_norm (final) |
|---|---:|---:|---:|---:|
| WSD+Cosine | **3.1290** | **22.85** | 29.52% | 29.43% |
| Cosine | 3.1505 | 23.35 | **29.56%** | **29.46%** |

Overall, WSD+Cosine achieves the best final perplexity, while pure Cosine reaches a slightly higher peak HellaSwag accuracy and a very similar final value. Compared to commonly reported GPT-2 (124M) references, our modernized model attains lower validation loss and comparable HellaSwag accuracy under the same parameter scale and a strict 10B-token budget. However, these GPT-2 reference numbers should be treated as rough context rather than a strict apples-to-apples comparison, since dataset, preprocessing, and evaluation pipelines can differ across reports.


### 4.3 Empirical Insights

![Training Dynamics](./training_dashboard.png)
*Figure 1: Training dynamics under a fixed 10B-token single-pass budget, comparing two learning-rate schedules (WSD+Cosine vs. pure Cosine). Left: loss curves. Right: HellaSwag acc_norm (completion-style).*

This project studies how modern architecture and training choices affect a GPT-2–scale model (124M) under a strict 10B-token single-pass budget. Based on training curves, samples, and final metrics over 19,073 steps, we summarize the following points.

**1. Strong results at fixed model size (with imperfect GPT-2 comparability).**  
With the same parameter scale as GPT-2 (124M), the modernized model reaches a final validation loss of **3.1290** (PPL **22.85**) under the best schedule. These numbers are better than commonly reported GPT-2 references, but they are not a strict apples-to-apples comparison because data and evaluation setups may differ.

**2. The LR schedule mainly changes *when* gains happen (and what wins at the end).**  
With all other settings fixed, **pure Cosine** improves a bit earlier in the middle stage and reaches a slightly higher **peak** HellaSwag (**29.56%** at step **18,500**). **WSD+Cosine** improves more in the late stage and ends with better validation loss (final **3.1290** vs. **3.1505** for Cosine). The validation-loss crossover happens late (around **~17.5k** steps).

**3. Better perplexity does not always mean better HellaSwag.**  
WSD+Cosine gives clearly better final perplexity, but the **final HellaSwag accuracy is almost the same** (29.43% vs. 29.46%). This shows that lower loss does not always translate into higher HellaSwag accuracy at this scale.

**4. Muon + AdamW training is stable under both schedules.**  
We apply Muon to large 2D weight matrices and AdamW to other parameters. Both runs are stable for the full 19k steps. We see a few one-step spikes in training loss, but validation loss stays smooth, so these spikes are likely batch noise rather than optimizer issues.

**5. Fluency appears earlier than HellaSwag gains.**  
Samples become grammatically correct relatively early, while HellaSwag improves more slowly and keeps rising later. This suggests that basic fluency is learned before stronger semantic choice ability in this budget setting.

Overall, these results suggest that modern design choices help at the GPT-2 scale under limited compute, and that the learning-rate schedule mostly affects the timing of improvements and the final trade-off between perplexity and downstream accuracy.



## Analysis and Discussion

**5.1 Why modern design helps at GPT-2 scale**

The goal of this project is to test whether modern architecture and training choices help at the GPT-2 (124M) scale under a strict 10B-token single-pass budget.

At the same model size, Modern-NanoGPT reaches lower validation loss and similar HellaSwag acc_norm compared to commonly reported GPT-2 references. These reference numbers are only rough context (data and evaluation can differ), but the results are consistent with gains coming from design choices rather than scaling.

A possible reason is that modern components (RoPE, RMSNorm, SwiGLU, and QK-norm) make training more stable and improve how the model uses its capacity. The hybrid optimizer (Muon for large 2D matrices, AdamW for other parameters) may also keep updates better behaved. Together, these choices can make better use of limited tokens without increasing model size or data.

**5.2 Perplexity vs downstream performance (schedule ablation evidence)**

Validation perplexity drops steadily during training, but HellaSwag acc_norm does not always improve at the same rate. The learning-rate schedule ablation makes this clear.

Pure Cosine improves a bit earlier in the mid phase, while WSD+Cosine improves more in the late phase and finishes with lower validation loss / perplexity. However, the final HellaSwag accuracy is almost the same across schedules (and the peak can even be slightly higher with pure Cosine). This suggests that lower perplexity does not always translate into higher HellaSwag accuracy at this scale.

One explanation is that perplexity mainly measures next-token prediction, while HellaSwag is a choice task that needs broader semantic consistency. Small changes that improve token prediction can have little effect on this kind of semantic ranking.


**5.3 Fluency vs semantic discrimination during training**

Qualitative samples show that basic fluency appears relatively early: sentences become grammatical and coherent in the mid stage. In contrast, HellaSwag improves more slowly and continues to rise later.

This suggests that learning common syntax and frequent patterns is easier than learning stronger semantic choice ability. Under a fixed token budget, the late stage seems to matter more for this kind of downstream behavior.

**5.4 Stability and generalization under a fixed budget**

Training is stable for the full 19,073 steps under both learning-rate schedules. The train–val gap stays small, which suggests limited overfitting in this setting.

Although we do not isolate each component with full ablations, the overall stability supports that modern normalization, attention scaling, and the Muon+AdamW optimizer work well under constrained compute.

Taken together, these observations support the main message of this report: at the GPT-2 scale, careful architecture, optimization, and learning-rate scheduling can improve training results under strict token and hardware limits.


## Limitations and Future Work

While this work shows better performance at the GPT-2 (124M) scale under a strict 10B-token single-pass budget, several limitations remain.

**6.1 Longer training budgets**

All experiments use a fixed 10B-token single pass. Although train and validation losses stay close near the end, the model may not be fully saturated.

Future work will train for larger token budgets (e.g., 30B–50B tokens, multi-pass) to test whether the same design choices keep helping at longer horizons.

**6.2 More controlled ablations (what is still missing)**

This study changes multiple factors at once (architecture + optimizer). We also ran a learning-rate schedule ablation (WSD+Cosine vs. pure Cosine). The schedule ablation suggests a trade-off: WSD+Cosine gives better final perplexity, while pure Cosine can reach slightly higher peak HellaSwag earlier. However, we did not isolate the effect of each architectural component or the optimizer choice.

Future experiments should include:

* Muon + AdamW vs. AdamW-only (with the same schedule)
* Step-by-step architectural ablations (RoPE, RMSNorm, SwiGLU, QK-norm)
* More schedule variants (e.g., different plateau length, different min_lr, different warmup)

These ablations are needed to identify which choices matter most for convergence and downstream metrics under limited compute.


**6.3 Broader evaluation**

Downstream evaluation currently focuses on HellaSwag. Future work will add more benchmarks to test whether the gains generalize (e.g., other zero-shot multiple-choice tasks and simple QA-style evaluations).

**6.4 Training stability**

Training is stable overall, but occasional one-step loss spikes are observed. Future analysis will check possible causes such as data batching/sharding, numerical precision, or rare hard batches.

Understanding these spikes may help further improve robustness under constrained budgets.


## Conclusion

This project studies whether modern architecture and training choices can improve a GPT-2–scale model (124M) under a strict 10B-token single-pass budget.

Modern-NanoGPT (RoPE, RMSNorm, SwiGLU, QK-norm, and a Muon+AdamW optimizer split) reaches lower validation loss and comparable HellaSwag acc_norm compared to commonly reported GPT-2 references, without increasing model size or training tokens (noting these references are not a strict apples-to-apples comparison).

We also run a controlled learning-rate schedule ablation under identical settings. WSD+Cosine achieves the best final perplexity, while pure Cosine reaches a slightly higher peak HellaSwag and improves a bit earlier in the mid phase, showing a small trade-off between final loss and the timing of downstream gains.

Finally, we find that lower perplexity does not always translate into higher HellaSwag accuracy: samples become fluent early, while HellaSwag improves more gradually. Overall, careful architecture, optimization, and scheduling choices can improve GPT-2–scale training under limited compute.

## References

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).
*Language Models are Unsupervised Multitask Learners.* OpenAI Technical Report.

Su, J., Lu, Y., Pan, S., Wen, B., & Liu, Y. (2021).
*RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.

Shazeer, N. (2020).
*GLU Variants Improve Transformer.* arXiv:2002.05202.

Zhang, B., & Sennrich, R. (2019).
*Root Mean Square Layer Normalization.* arXiv:1910.07467.

Wortsman, M., Hayase, P., Zhai, X., Jaitly, N., & Schmidt, L. (2023).
*Stable and Efficient Training of Large Language Models with QK-Norm.* arXiv:2302.05442.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., & Choi, Y. (2019).
*HellaSwag: Can a Machine Really Finish Your Sentence?* Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL).

Jordan, K. (2024).
*Muon: An Optimizer for Matrix-Valued Parameters.* GitHub Repository.
https://github.com/KellerJordan/Muon

Karpathy, A. (2023).
*nanoGPT.* GitHub Repository.
https://github.com/karpathy/nanoGPT

Penedo, G., Malartic, Q., Hesslow, D., Cojocaru, B., Cappelli, A., Alobeidli, H., et al. (2024).
*The FineWeb Datasets: Decanting the Web for the Next Generation of Language Models.* arXiv:2406.17539.
