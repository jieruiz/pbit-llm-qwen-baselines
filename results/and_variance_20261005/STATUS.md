# 四路径 AND 输出方差约束：执行状态

2026-10-05：已提交单GPU Slurm初始化作业1073，个人目录
`/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005`。

- 已完成：五项解析方差/梯度/RNG/归一化测试；完整BF16 PPL核验11.652735。
- 进行中：按原脚本重训全部20个局部FFN，然后1000步联合训练选择共同初始化。
- 待执行：单层lambda筛选、六组2000步正式训练、26次完整测试、制品哈希核验。
- 起点为本次同方法新权重，不是师弟原checkpoint的字节复现。
- 固定K=4、N=4；成功门槛相对同预算对照PPL改善至少2%，详见
  [预定协议](../../experiments/pdnn_ffn/AND_VARIANCE_PROTOCOL_ZH.md)。

这里的预检通过不代表正式质量目标已达到。结果产生后会继续更新并保留负结果。
