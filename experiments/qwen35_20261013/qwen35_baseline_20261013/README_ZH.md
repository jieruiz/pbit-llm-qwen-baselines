# Qwen3.5-0.8B 远程部署与原始基线

记录日期：2026-10-13，北京时间。更新：原始模型的完整 WT2、五项基础任务及生成/性能检查均已完成。后续替换实验见相邻目录 qwen35_pbit_20261013。

## 已确认的结果

| 检查或指标 | 结果 |
|---|---|
| 官方权重加载 | 通过；没有缺失或形状不匹配的参数 |
| 短文本前向、有限数值检查 | 通过 |
| 分块输出头计分与直接 logits 计分 | 通过一致性检查 |
| PIQA 评估接口小样本检查 | 通过；小样本成绩不作为正式成绩 |
| 递归缓存与完整前向的末位置 logits | 余弦相似度 0.999957；BF16/不同内核路径下不是逐位相等 |
| 完整 WikiText-2 test PPL | **15.6610978048** |
| WT2 总 token／实际计分 token | 297193／297192 |
| WT2 滑窗数 | 290 |
| PIQA 全量零样本 | acc **68.0631%**；acc_norm 69.1513%；1838 题 |
| HellaSwag 全量零样本 | acc_norm **49.6216%**；10042 题 |
| WinoGrande 全量零样本 | acc **58.7214%**；1267 题 |
| ARC-E 全量零样本 | acc **65.7828%**；2376 题 |
| ARC-C 全量零样本 | acc_norm **37.2867%**；1172 题 |
| 生成样例及 2K/8K 延迟、峰值显存 | 完成；2K/8K decode 约 20.91/20.20 ms/token；峰值 allocated 约 1.80/2.26 GiB |

WT2 上下文 2048，步长 1024，无额外 BOS；每个可预测 token 恰好计分一次。窗口开始时重置完整注意力缓存和 DeltaNet 状态，与此前的滑窗协议一致。输出头按 64 个位置分块，仅控制显存，不改变预测目标或计分范围。语料文件 SHA256 为 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`。

这里的基线是用户指定的 **后训练版本 Qwen/Qwen3.5-0.8B**，不是 Base。只评估文本，不调用视觉编码器。所有 24 层保留官方连续计算，包括 18 层 Gated DeltaNet、6 层完整门控注意力和 24 层 SwiGLU FFN，没有加入 p-bit、随机 FFN、随机 Q/K、PV 采样或 Ising 替换。

新旧模型分词器不同，新模型 WT2 计分 297192 个 token，旧模型计分 299077 个 token；而且旧实验为 Qwen2.5-0.5B Base。因此 **15.6611 与旧模型 11.6527 不能直接用来判断新旧模型质量优劣**。后续替换应以新模型自身的准确基线为参照。任务准确率可以在相同数据和提示协议下对照，并注明训练阶段。

## 远程位置、版本与任务

- 工作目录：`/home/Hongjie_Zeng/qwen35_baseline_20261013/`
- 模型：`models/Qwen3.5-0.8B/`
- 环境：`venv/bin/python`
- 正式结果：`results/wikitext2.json` 和 `results/<任务名>.json`
- 原始日志：`results/logs/`
- 模型版本和每个下载文件的 SHA256：`deployment_manifest.json`
- 实际依赖：`requirements.lock.txt`
- 部署时模型及评估框架源码：`sources/`

官方仓库版本锁定为 `2fc06364715b967f1860aea9cf38778875588b17`。下载使用 Hugging Face 镜像端点，保存官方仓库 revision 和实际文件哈希。全部文件合计 1769980465 字节，未在本地重复下载大权重。

环境采用独立 venv，并通过 system-site-packages 复用旧环境的 PyTorch/CUDA；新 Transformers、lm-eval、PEFT 及 FLA 安装在新 venv 中，没有修改旧实验环境。主要实际版本：PyTorch 2.7.1+cu128、Transformers 5.19.0、lm_eval 0.4.13、datasets 4.8.5、PEFT 0.21.2、flash-linear-attention/fla-core 0.5.2。部署 manifest 记录的是安装 FLA 前的初始版本，最终环境以 requirements.lock.txt 为准。

使用 BF16 权重及模型原生的 FP32 内部计算；完整注意力采用 SDPA，DeltaNet 采用 FLA 内核。短卷积保留 Transformers 的准确 PyTorch 参考实现，没有安装 causal-conv1d。这可能影响速度，不代表最优推理框架性能。

| Slurm 任务 | 用途 | 状态 |
|---|---|---|
| 1225 | 下载、哈希和源码保存；不占 GPU | 完成 |
| 1226 | 初次兼容性检查 | 新版任务配置接口不兼容，已修复 |
| 1228 | 修复后的检查，含 FLA 内核 | 完成 |
| 1227_0 | 完整 WT2 | 完成 |
| 1227_1 | HellaSwag | 完成 |
| 1227_2 | PIQA | 完成 |
| 1227_3 | WinoGrande | 完成 |
| 1227_4 | ARC-E | 完成 |
| 1227_5 | ARC-C | 完成 |
| 1227_6 | 生成、2K/8K 性能 | 完成 |

1227 阵列限制最多两个 GPU 任务同时运行，全部由 Slurm 分配，每个任务一张 5090，batch=1。正式评估依赖 1228 检查成功。最长的 HellaSwag 实际耗时约 25 分钟，全部任务已完成。

## 五项任务的统一协议

| 任务 | 数据划分 | 全量题数 | 汇总优先指标 |
|---|---|---:|---|
| HellaSwag | validation | 10042 | acc_norm |
| PIQA | validation | 1838 | acc |
| WinoGrande XL | validation | 1267 | acc |
| ARC-E | test | 2376 | acc |
| ARC-C | test | 1172 | acc_norm |

保留 harness 同时输出的 acc、acc_norm 和标准误等完整指标。全部为零样本，不使用聊天模板计算候选对数似然；聊天模板仅用于生成样例，并显式关闭 thinking。复用已校验的离线 parquet 数据，逐文件核对旧 manifest 的 SHA256，没有再下载一套重复数据。当前评估工具升级到 0.4.13，后续跨模型任务对比仍需注明工具版本；最严格的算子替换对照将在这一套冻结环境内完成。

性能任务记录一张 GPU、batch=1、2K/8K 提示长度、64 个逐 token 解码的延迟和峰值显存，并保存三条确定性生成样例。它只做少量预热和一次每长度测量，可能包含形状首次初始化，不作为稳定吞吐、整机功率或能耗结论。

## 复查与继续运行

连接命令沿用本机既有 SSH 配置，不在文件中复制凭据：

```powershell
ssh -F C:\Users\86134\.ssh\lab-server-ai.ssh_config lab-server-ai
```

登录后查看当前任务和结果：

```bash
/opt/slurm/21.08.8/bin/squeue -u Hongjie_Zeng
cd /home/Hongjie_Zeng/qwen35_baseline_20261013
venv/bin/python collect_status.py
```

若后续某项失败，阅读对应日志后提交 baseline.sbatch 的相应阵列下标；脚本会跳过已有完成结果，不覆盖正式结果。所有 GPU 执行都要求 SLURM_JOB_ID，不能直接在登录会话启动。

旧模型的本次正式结果汇总位于本目录 `PREVIOUS_RESULTS_ZH.md`，原始只读快照位于 `previous_experiments_snapshot.json`。它保留所有已完成种子，不将未完成对照组写成最终结果。

参考：[官方模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。
