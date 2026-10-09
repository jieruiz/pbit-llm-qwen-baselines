# 当前FFN＋QK前筛块：准确PV与采样PV

2026-10-09按用户要求启动并完成。模型Qwen2.5-0.5B Base，没有训练。

## 结果

26项正式评测、4项GPU单元测试、4项短模型检查完成，94条源码/协议/结果审计通过。
下表随机配置均为3seed均值±总体标准差；原模型为确定性单次。

| 配置 | 2K后缀PPL | 8K后缀PPL |
|---|---:|---:|
| 原模型 | 10.3813 | 9.4190 |
| 当前FFN＋不筛块、准确PV | 12.3129±0.0598 | 11.1199±0.1568 |
| 当前FFN＋不筛块、PV采样 | 12.3976±0.1073 | 11.2562±0.1047 |
| 当前FFN＋筛块、准确PV | 12.8287±0.0763 | 11.2204±0.1712 |
| 当前FFN＋筛块、PV采样 | 12.9763±0.0235 | 11.2383±0.1159 |

两种筛选均准确Softmax，2K实际保留约27.9%，8K约29.5%；本批处理padding后QK比例约37.7%/36.0%。
相对筛块准确PV，叠加PV采样的均值增幅约2K1.15%、8K0.16%；仅为本次均值比较，不是显著性结论。
8K筛块准确/采样PV相对同FFN准确attention约增加0.90%/1.06%；2K约增加4.19%/5.39%。
当前证据支持长上下文进一步研究叠加，不能推广到全部任务；FFN本身相对原模型仍有明显损失。
GPU参考解码耗时没有改善：2K不筛准确PV约226s，筛块准确/采样约238/242s；8K约115/118/122s。
筛选、索引、padding、采样和FFN仿真开销不能从候选比例中省略。
详细原始表见[完整报告](results/RESULTS_ZH.md)，结构化结果见[summary.json](results/summary.json)。

## 配置与协议

- `screen_exact`：四摘要p-bit筛块，候选内准确QK/Softmax/PV。
- `screen_sampled`：同一筛选器，候选内准确QK/Softmax，512次IID V采样取平均。
- 对照：`all_exact`、`all_sampled`不筛块；原模型SDPA。
- 当前固定尺度FFN覆盖24层prefill和decode：W8，幅度12位+符号，输入4096/输出3072，interval256/burn25%。
- Attention筛选覆盖24层decode；prefill保持原始SDPA。Q/K/V/O投影与BF16残差准确。
- 64-token块、4个16-token子摘要取最大评分；当前评分均值/标准差归一化，旧moment.pt校准，目标30%。
- 首块/最近两块强制保留；温度固定于校准值，不进行多轮退火。筛选与PV、各层RNG独立。
- 真正先收集候选K再QK，短行有padding，分别统计逻辑与实际padding算术范围。
- sampled PV直接gather512个V并平均；不先算准确PV，也不是dense counts@V。

两档上下文2K/8K，分别32/16段、每段128步、batch4；随机配置三个seed。
共26项正式测试；此前12.49等指标来自prefill也采样的另一协议，不能直接作为本轮基线。
先运行4项GPU单元测试与4种配置的短模型检查，成功后自动并行测试，最后自动生成报告。
原始校准来自训练集；本轮不以test调参，不重新训练。

## 单卡复现与归档审计

本目录和前一实验的prefill范围不同，不能直接混用基线。原实验源码及校准逐字节保留，
新增run_variant.py只包装已测试的evaluate.py，不改变算法，也没有另做GPU重测。

在本目录安装Python3.12环境，准备Qwen2.5-0.5B Base模型和WikiText-2 raw：

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
mkdir -p models data
ln -s /absolute/path/to/Qwen2.5-0.5B models/Qwen2.5-0.5B
ln -s /absolute/path/to/wikitext-2 data/wikitext-2
python run_variant.py screen_exact --context 8192 --seed 0 --gpu 0
python run_variant.py screen_sampled --context 8192 --seed 0 --gpu 0
```

context可选2048/8192，seed可选0/1/2，对应[configs](configs)中的四个配置。
Windows可复制模型/数据目录替代符号链接。模型需完整权重、config及tokenizer；数据需要wiki.test.raw。
模型与数据按各自许可证另行获取；发布包包含85KB的筛选校准moment.pt及FFN尺度，无额外学生权重。
默认结果写入outputs/，不会覆盖results/；--dry-run显示命令，无需GPU。

无需模型、数据或GPU可执行`python report.py`，应通过94条历史审计；完整结果在results/。
4项GPU单元测试可用`python test_router.py`重跑；历史日志在results/logs/units.log。
公开记录将服务器路径替换为<RUN_ROOT>/<PYTHON>，数值结果、seed、token哈希和冻结代码均保持。
发布来源及文件哈希见[PUBLICATION.json](PUBLICATION.json)。

run.py/start.py是原多卡调度器，要求干净目录且results不存在；完整重跑应另复制源码、校准并准备assets。
不要在本发布目录删除results重跑。通常用上面的单卡入口即可。
使用GPU1–6中空闲设备，0/7保留；每任务一张卡。已经存在results时拒绝覆盖。
GPU数值模拟不代表实际p-bit硬件面积、速度或能耗。KV缓存容量不因少读候选而自动减少。
