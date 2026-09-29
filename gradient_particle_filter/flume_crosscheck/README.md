# 水槽算例交叉对比：dclaw_jax vs Fortran D-Claw（2010 SGM 闸门释放）

目的：在做粒子滤波之前，确认 JAX 模型在**相同输入**下能复现 Fortran D-Claw。
这里是模型对模型的对比，不是与实测数据对比（Fortran 与实测本身差很多，见
`validation/usgs_flume_2010_paper2014/results/summary.md`）。

## 运行

```
source validation/usgs_flume_2010_paper2014/environment.sh
python fortran_setup.py                                  # 按论文默认配置生成输入到 fortran_run/（不改 model/）
cd fortran_run && OMP_NUM_THREADS=32 ../../../validation/usgs_flume_2010_paper2014/model/xdclaw && cd ..   # ~10 min
python run_jax.py                                        # ~22 s（GPU）；CPU ~5 min
python compare.py                                        # crosscheck.png + 汇总（JAX_RUN=... 选文件）
python summarize.py "label=file.npz" ...                 # 多个 JAX 运行一行一行对比 Fortran
```
`run_jax.py` 的环境变量只影响本次运行，不改包：`JAX_DX`（网格）、`JAX_DRY_TOL`（薄层阈值，经 `../flume_solver.py`）、
`JAX_RECON`（`audusse` / `chen_noelle`）、`JAX_UTAPER`、`JAX_OUT`。

## 输入一致性

地形、初始料堆、固体体积分数全部读自 Fortran 自己的 `.tt3`；初始 h 与 Fortran `fort.q0000` 的最大差 **4e-16 m**，
体积均为 10.1492 m³。JAX 默认用均匀 0.0625 m 网格（= Fortran 3 级 AMR 最细层），横向均匀故取 2 个 1 m 格。

## 结果（包已修正：去掉 α 下限；重构默认 Audusse）

| 运行 | 前锋 t=1 / 5 / 10 / 35 s (m) | 到达 x=2 / 32 / 66 / 90 m (s) | 测点流深 RMS 差 (m) | 35 s 时 32/66 m 残留 | 35 s 质量变化 |
|---|---|---|---|---|---|
| **Fortran**（0.0625 m AMR） | 6.5 / 54.9 / 110.1 / 113.3 | 0.42 / 3.39 / 5.69 / 7.35 | — | 0 / 0 | −7.3% |
| 修正前（0.0625） | 3.8 / 41.2 / 91.8 / 95.5 | 0.69 / 4.13 / 7.09 / 9.65 | 0.049 / 0.032 / 0.038 / 0.143 | 18 / 16 mm | −8.5% |
| **修正后 默认**（0.0625） | 4.0 / 44.8 / 96.3 / 99.3 | 0.67 / 3.89 / 6.65 / 8.90 | 0.044 / 0.018 / 0.025 / 0.110 | 19 / 16 mm | −8.3% |
| 修正后 默认（0.03125） | 4.2 / 47.4 / 102.0 / 105.7 | 0.64 / 3.79 / 6.32 / 8.27 | 0.027 / 0.012 / 0.016 / 0.051 | 7 / 4 mm | −8.7% |
| 修正后 默认（0.015625） | 4.5 / 48.8 / 104.9 / 108.9 | 0.60 / 3.74 / 6.16 / 7.99 | 0.013 / 0.010 / 0.011 / 0.026 | 4 / 0 mm | −10.4% |
| Chen–Noelle，1 mm（0.0625） | 5.0 / 44.8 / 96.3 / 99.3 | 0.51 / 3.90 / 6.65 / 8.90 | 0.009 / 0.020 / 0.027 / 0.110 | 0 / 0 | −12.8% |
| Chen–Noelle，1 mm（0.015625） | 4.9 / 48.8 / 104.9 / 108.9 | 0.48 / 3.73 / 6.16 / 7.99 | 0.006 / 0.010 / 0.011 / 0.026 | 0 / 0 | −13.7% |
| Chen–Noelle，2 mm（0.0625） | 4.6 / 43.7 / 94.3 / 97.5 | 0.55 / 3.98 / 6.79 / 9.19 | 0.009 / 0.022 / 0.029 / 0.112 | 0 / 0 | −30.3% |

图：`crosscheck_before_fixes.png`、`crosscheck_after_fixes.png`（默认，0.0625 m）。

## 找到并处理的问题

### 1. α 下限（已修正，包 README 第 12 条）
`model.py` 把压缩系数 α 截在 ≥1e-5 Pa⁻¹，Fortran 没有下限。本算例 α 应为 2.0e-6（论文表 2），
剪缩产生的孔压和孔压松弛速率（都 ∝1/α）弱了 5 倍：闸门后 1 s 时 pb/岩土静压 Fortran 0.634、JAX 0.346；
去掉后 0.621。测点流深误差降约 40%。梯度仍精确（全部测试、Montecito `grad_check.py`、`sensitivity.py`）。
d(输出)/d(alpha_c) 从 −1.5 变为 −44.7——之前 alpha_c 几乎失效。Montecito 几乎不变（CSI 0.491 → 0.488）。
`DILATANCY_U_TAPER`（包 README 第 11 条）在本算例无影响。

### 2. 陡坡上薄层驱动力不足（Audusse 重构）——保留为默认，Chen–Noelle 作为选项
Audusse 重构对比床面台阶 db 还薄的层只给 h/(2 db) 的重力驱动力（31° 坡、0.0625 m 格、db=3.8 cm：2 cm 层只有 27%），
薄尾被钉在坡上（32/66 m 处 ~2 cm 残留）、薄前锋偏慢。误差是一阶的，随加密减小（见上表）。

已在包里实现 Chen & Noelle (2017) 重构（`solver.RECONSTRUCTION = "chen_noelle"`；全湿界面与 Audusse 完全相同，
静水、岸线、正性不变；薄层驱动力 1 − h/(2 db)，2 cm 层 73%）。它消除了残留层、把 x=2 m 误差从 0.044 降到 0.009 m，
**但没有设为默认**，原因是和可微性冲突：它让 1–3 mm 的薄膜动起来，反复穿过 `_clip_dry` 的 1–2 mm 过渡区：

| 在 `test_gate` 设置下 d/dφ（AD vs FD eps=1e-3 / 1e-5 / 1e-7） | Audusse | Chen–Noelle 1 mm | Chen–Noelle 2 mm |
|---|---|---|---|
| 3 m 下游体积 | −0.08229 vs −0.08200 / −0.08229 / −0.08229 | −0.0605 vs −0.088 / −0.154 / −1.06 | −0.07980 vs −0.07983 / −0.07980 / −0.07980 |
| x=10 m 峰值流深 | +1.414e-3，三者一致 | −2.46e-3 vs +2.93e-3 / +8.98e-3 / −2.41e-3 | +2.751e-3，三者一致 |
| x=5 m 峰值流深 | 一致 | 一致 | 一致 |

再加上质量：Chen–Noelle 下 `_clip_dry` 在 35 s 内删掉 13–18%（2 mm 阈值时 30%），Fortran 7%、Audusse 8%。
结论：**要梯度就用 Audusse + 加密网格**（0.03125 m 约 44 s/次，0.015625 m 约 140 s/次）；
要同时得到 Chen–Noelle 的精度和精确梯度，需要一种质量守恒且可微的薄层处理，目前没有。

### 3. 质量
两边都丢质量：Fortran 7.3%（15 s 后不变），JAX 默认 8.3%（0.0625 m）。JAX 流动停止后仍缓慢减少，
来自堆积体边缘蠕动时被 `_clip_dry` 削掉的薄片（按算子分解：剪胀、qfix 贡献为 0）；蠕动与第 2 条及未移植的
静摩擦上限（Fortran Riemann 求解器里的 calc_taudir）有关。

### 仍未解决
开闸头 1–2 s（x=2 m 到达：Audusse 0.60–0.67 s，Chen–Noelle 0.48 s，Fortran 0.42 s）：料面一开始就比墙顶高约
0.18 m 的"溢流"阶段，两种 Riemann 求解器处理不同，加密网格改善很慢。

## 对粒子滤波的意义

- 修正 1 之后，孔压和测点流深时程与 Fortran 接近，alpha_c 成为可辨识的参数。
- 梯度辅助同化用默认 Audusse；为压低陡坡误差，建议用 0.03125 m（32/66 m 测点 RMS 0.012/0.016 m）。
- 用 Fortran 生成合成观测、JAX 做同化，会把剩余的 JAX–Fortran 差异带进"模型误差"；
  合成 twin 实验应由 JAX 自己生成观测。
