# 四个 FFN 的训练预算与中间输出蒸馏对照

预先确定于 2026-09-29，基于 main `f41bbbb`。本轮独立于师弟的十层优化：固定替换零基层 10、11、12、13，不新增 decoder 替换层，不改变 P-DNN 深度、宽度、编码、温度或推理计算。

## 条件

| 名称 | 联合更新步数 | 辅助损失权重 |
| --- | ---: | ---: |
| reference_1000 | 1,000 | 0 |
| long_3000 | 3,000 | 0 |
| feature_3000 | 3,000 | 0.1 |

四个局部初始化各按原协议训练：seed 0、2000 mean-field + 2000 四路径 STE、896→4864→896、入口温度 0.25、隐藏温度 1.0、tanh ±1。三组共享同一套初始化 checkpoint；联合训练 seed 0、有效 batch 1024 token、学习率 1e-5、50 步 warm-up、各自预算内 cosine decay，KL/CE 权重 0.8/0.2。因此 3000 步与 1000 步比较包含相应学习率日程的变化，不是从 1000 步 checkpoint 续训。

新增目标为 `L = 0.8 KL + 0.2 CE + λ * mean_l NMSE(FFN_student_l, FFN_teacher_l)`。
匹配同一段文本、同一 token 位置上的 FFN 连续输出（每个 student 内完成路径平均之后）。Teacher 和 student 分别接收各自网络产生的隐藏状态；不向 student 前向注入 teacher 状态。按每层 teacher 输出均方值归一化，teacher 张量和统计量均停止梯度。这里匹配的是 FFN 残差更新，而非包含恒等残差的完整 decoder block 输出。

推理与原模型替换流程完全相同；辅助 loss、teacher 和捕获 hook 不进入部署 checkpoint。

## 验证与选择

- 留出训练文本末尾 65,536 token，不参加局部或联合优化；验证均匀取 64 个 256-token 窗口，实际计分 16,320 token。
- 所有候选 checkpoint 使用固定验证采样种子 12345；验证保护训练 RNG。每 500 步验证一次，按验证 CE/PPL 最低选择，另保存明确的 final checkpoint。
- 三组条件与辅助权重在跑 test 前确定；不据 test 改权重或挑训练 checkpoint。
- 最终对每组测完整 WikiText-2 test，window 2048、stride 1024；N=4 推理 seed 0/1/2，另测 N=0 和 N=16 seed 0。N=0 是 mean-field 代理，不是严格无限采样极限。
- 三个推理种子只描述同一训练 checkpoint 的随机性。本轮每个条件只有一个联合训练种子；结论属于探索性结果，不宣称多次训练稳定性或下游任务泛化。
- 尚不扩大到六层；只有有质量收益的改动才值得下一轮进一步重复、扩层及下游评测。

## 运行

在已分配的 GPU 节点执行 `test_feature_distillation.py`，再运行 `run_joint_feature_ablation.py`，传入 `--model`、`--train-text`、`--test-text`、`--baseline` 和新的 `--output-dir`。原 BF16 baseline 必须在同一环境重测，要求原模型 revision、模型和文本哈希一致。

运行器在正式训练前完成一次短局部训练、带辅助 loss 的联合训练和整模型评估。任一步失败即停止。旧的联合训练脚本保持原样，新入口为 `train_joint_feature_distillation.py`。
