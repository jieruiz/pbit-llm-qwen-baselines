
## 执行与制品核验

初始化GPU作业1073；正式GPU作业1075，应用退出码0，RUN_COMPLETE存在。
正式执行源码提交`13be8a871cd714e430db00ae36d141abb16f65e2`；两个源码zip和全部162个原始结果文件均核验SHA256和字节数。
在Windows独立重算全部汇总指标，离散配置与判断严格一致，浮点数使用1e-12相对/绝对容差；原始comparison.json与RESULTS_ZH.md均保留服务器字节。
原BF16基线完整复现；共同初始化、每个正式模型best/final及其他checkpoint身份见result-manifest.json。权重不进入Git。
