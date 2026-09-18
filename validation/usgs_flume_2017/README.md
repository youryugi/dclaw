# USGS 2017 水槽 D-Claw 验证

本目录把 2017-05-24 和 2017-05-25 的动态 LiDAR、1 kHz 传感器记录与 D-Claw 两相孔压模型放到同一坐标与指标体系中。当前成果是可复现的未校准基线和小型敏感性试验，不代表模型已通过工程验证。

## 已实现的观测量

| 数据 | 用途 |
|---|---|
| 流深传感器 | 2.5、31.7、65.4、80 m 的首次/持续到达时间、峰值和时序误差 |
| 孔压传感器 | 31.7、65.4、80 m 的孔压时序和峰值 |
| 60 Hz 线扫 LiDAR | 床面纵剖面、运动起点、相干前缘位置和最大传播距离 |
| 试验体积与上游深度 | 重建 LiDAR 视场外的闸后初始料堆 |

## 环境与运行

Clawpack 运行树位于 `/home/yang/github/clawpack-runtime`。从本目录执行：

```bash
source environment.sh
python -m unittest discover -s tests -v

FLUME_RUN_DATE=2017-05-25 make -C model input data output
python compare_model.py --date 2017-05-25
```

参数敏感性运行不修改基线配置，例如：

```bash
source environment.sh
FLUME_RUN_DATE=2017-05-25 DCLAW_PHI_DEG=42 \
  make -C model input data OUTDIR=_output_2017-05-25_phi42 output
python compare_model.py --date 2017-05-25 --label phi42 \
  --outdir model/_output_2017-05-25_phi42
python summarize_results.py
```

作者论文约束方案使用论文材料参数，并用 2017 实测孔压与闸门角度覆盖相应旧试验条件：

```bash
source environment.sh
FLUME_RUN_DATE=2017-05-25 DCLAW_PARAMETER_SET=paper2014 \
  make -C model input data OUTDIR=_output_2017-05-25_paper2014 output
python compare_model.py --date 2017-05-25 --label paper2014 \
  --outdir model/_output_2017-05-25_paper2014
```

支持 `DCLAW_PHI_DEG`、`DCLAW_PINIT_RATIO` 和 `DCLAW_KREF_M2` 三个环境变量覆盖。大体积 LiDAR 原始文件只需预处理一次；中间数据、图和模型输出均位于被 Git 忽略的 `generated/` 或 `_output_*/`。

## 当前假设与限制

- 计算采用 0.25 m 网格、2 m 槽宽和瞬时移除闸门；真实闸门开启历时尚未建模。
- 动态 LiDAR 是狭窄线扫，局部遮挡会限制前缘和初始料堆识别；初始料堆因此由体积和上游法向深度重建。
- 当前初始网格质量相对目标体积偏差约 0.9%（05-25）和 3.1%（05-24），后续网格收敛试验需量化其影响。
- 首次到达阈值为 0.02 m，LiDAR 相干前缘阈值为 0.03 m；阈值敏感性尚未纳入参数校准。
- LiDAR 在约 65–70 m 后存在遮挡或有效覆盖限制；前缘 RMSE 只统计观测前缘不超过 65 m 的时段，不能解释为最终堆积距离误差。
- `paper2014` 来自更早的 10 m³ SGM 聚合试验，是作者先验而非 2017 逐次实测参数；映射依据见 `results/paper_parameter_mapping.md`。
- 当前结果不宜用于预测；应先完成闸门运动、网格收敛和留出试验验证。
