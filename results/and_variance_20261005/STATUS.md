# 固定 K=4、N=4 方差约束：已完成

重训初始化作业1073；正式作业1075；补充采样核验作业1076；CPU制品归档作业1077，均COMPLETED、退出码0。
六组正式训练各2000步，26次完整测试已完成；两个源码快照及全部原始结果SHA256核验，局部实际N4采样核验已记录。

- A：三组训练seed的N4平均PPL 13.386293。
- B：同预算N4平均PPL 13.390170，lambda=0.1。
- 相对改善 -0.029%，配对胜出 1/3；预定目标未达到。
- 所有组使用同方法重训的共同初始化；不是师弟原checkpoint的字节复现。三个seed仅重复继续训练阶段。
- [原始结果报告](RESULTS_ZH.md)、[比较数据](comparison.json)、[核验记录](VERIFICATION_ZH.md)、[实际N4采样核验](empirical_variance.json)。
- 140个权重的远端归档位于 `/home/Weican_Chen/projects/pbit-qwen-and-variance-20261005/best-checkpoints.zip`，身份见best-checkpoints-manifest.json；权重不上传Git。

A日志variance_penalty=0表示该项未启用、未计算，不表示A的物理输出方差为零。
