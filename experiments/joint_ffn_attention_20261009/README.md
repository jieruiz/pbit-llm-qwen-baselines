# 全24层固定尺度随机FFN＋Attention：三个联合版本

本分支发布2026-10-09完成的三种联合配置、产生结果的源码、离线校准和完整对照结果。
使用Qwen2.5-0.5B **Base**；没有训练，没有额外学生权重。
全部24层、prefill与decode都启用对应替换，BF16模块接口和残差保持不变。

| 配置 | Attention内容 | 三seed后缀PPL（均值±总体标准差） |
|---|---|---:|
| [`pv_only`](configs/pv_only.json) | 准确Q/K、准确Softmax、512次IID PV采样 | 12.490136 ± 0.050821 |
| [`qk_ising`](configs/qk_ising.json) | 随机Q/K投影T256、Ising 512次采样、V选择累加 | 13.215760 ± 0.081924 |
| [`qk_softmax_pv`](configs/qk_softmax_pv.json) | 随机Q/K投影T256、准确Softmax、512次IID PV采样 | 13.237724 ± 0.026454 |

同协议原模型PPL=10.381261，当前FFN＋准确手写attention=12.342713±0.030084。
这里的随机Q/K是固定权重q_proj/k_proj投影，**QK点积仍确定性**，V/O投影保持原始。
在当前流长下，新增损失主要来自随机Q/K这一组；建议先使用`pv_only`作为集成基线。
不是完整WT2滑窗PPL：32段×128个后缀计分token、context2048、batch4，共4096计分。
不能直接与以前完整语料的11.652735或13.873066比较。

详细结论、置信区间、完整对照及15条续写见[结果报告](results/RESULTS_ZH.md)。
结构化结果见[summary.json](results/summary.json)，全部22项test见[results/test](results/test)。

## 实现与范围

三个版本共用当前固定尺度FFN：W8，12位幅度加符号，group_max固定二次幂激活尺度，
输入4096次、每256次更新、25%预热、输出3072次；保留理想sigmoid和宽中间估计。
它不是旧训练AND版，也不是在线计算最大值的动态尺度版。
Q/K随机投影使用W10、9位幅度加符号输入、固定二次幂尺度。
Ising为强互斥极限连续时间热浴：最大score参考，sigmoid转移速率，burn16、spacing4、512次等时观测；
没有用准确Softmax概率驱动Ising。没有验证有限耦合、驱动电路或传播延迟。
IID和Ising都用dense counts@V模拟选择累加，GPU代码不是稀疏加速内核。
FFN/投影使用计数律压缩模拟随机流，不是RTL或有限位宽累加器模拟，不能据此声称硬件节能。

## 单卡复现三种配置

测试环境：Python3.12、PyTorch2.7.1+cu128、Transformers4.45.2，RTX5090 32GB。
在本目录运行，先安装PyTorch CUDA12.8构建，再安装依赖：

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
mkdir -p models data
ln -s /absolute/path/to/Qwen2.5-0.5B models/Qwen2.5-0.5B
ln -s /absolute/path/to/wikitext-2 data/wikitext-2
python run_variant.py pv_only --seed 0 --gpu 0
python run_variant.py qk_ising --seed 0 --gpu 0
python run_variant.py qk_softmax_pv --seed 0 --gpu 0
```

模型目录应包含原始Base权重、config及tokenizer；数据目录包含`wiki.test.raw`，
重跑验证还需`wiki.train.raw`。Windows可复制目录代替符号链接。
模型与数据由使用者按各自许可证另行获取，未提交进本分支。
每个配置分别运行seed0/1/2以对应发布结果；`--dry-run`仅显示命令，不加载GPU。
默认写入`outputs/<variant>_s<seed>.json`，不覆盖发布的`results/`。
流长、batch和调用顺序影响随机序列，比较时请保留配置。

## 审计与归档

无需模型、数据或GPU即可核对归档：

```bash
python report.py
```

应通过281条检查，包括源代码/校准哈希、12项验证、22项test、调用范围、计分位置、缓存和15条续写。
3项GPU采样单元测试的原始日志在`results/logs/units.log`；可用`python test_joint.py`重新运行。
原始数值实现、校准文件逐字节保留，`results/source_sha256.json`及`calibration_sha256.json`用于验证。
发布新增的配置和包装脚本未重新做GPU评测；它们直接调用原先已测试的evaluate.py。
`run.py`/`extra.py`是原多卡调度器，因保留历史协议而不改写；run.py要求results目录不存在，
完整重跑应复制源码与校准到干净目录、准备assets，再分别启动两脚本；通常用run_variant.py即可。
为避免覆盖原始结果，不要删除此发布目录的results来重跑。
公开记录仅将服务器目录与Python路径替换为`<RUN_ROOT>`/`<PYTHON>`，
没有更改数值结果、文本、seed、token哈希或冻结源码；详见[发布清单](PUBLICATION.json)。
