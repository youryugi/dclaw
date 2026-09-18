#!/usr/bin/env python3
"""Build a compact cross-run validation and sensitivity summary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
COMPARISON = HERE / "generated" / "comparison"
RESULTS = HERE / "results"
GAUGES = ("2.5", "31.7", "65.4", "80")
CASES = (
    ("2017-05-24", "baseline", "φ=40°, 初始孔压比=0.60"),
    ("2017-05-25", "baseline", "φ=40°, 初始孔压比=0.60"),
    ("2017-05-25", "phi42", "φ=42°, 初始孔压比=0.60"),
    ("2017-05-25", "phi44", "φ=44°, 初始孔压比=0.60"),
    ("2017-05-25", "p055", "φ=40°, 初始孔压比=0.55"),
    ("2017-05-24", "phi42", "φ=42°, 初始孔压比=0.60；独立检验"),
    ("2017-05-25", "paper2014_nogate", "论文参数；瞬时开闸"),
    ("2017-05-25", "paper2014", "论文参数；实测有限开闸"),
    ("2017-05-25", "paper2014_k5e8", "论文参数；k=5×10⁻⁸ m²"),
    ("2017-05-25", "paper2014_k5e7", "论文参数；k=5×10⁻⁷ m²"),
    ("2017-05-24", "paper2014", "论文参数；留出试验"),
    ("2017-05-24", "paper2014_k5e8", "k=5×10⁻⁸ m²；留出验证"),
)


def load(date: str, label: str) -> dict:
    return json.loads((COMPARISON / f"{date}_{label}_summary.json").read_text())


def finite_mean(values: list[float]) -> float:
    values = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    return float(np.mean(values[np.isfinite(values)]))


def main() -> None:
    rows = []
    machine = []
    for date, label, parameters in CASES:
        summary = load(date, label)
        arrival_errors = [summary["gauges"][key]["arrival_error_s"] for key in GAUGES]
        coherent_errors = [
            summary["gauges"][key]["coherent_arrival_error_s"] for key in GAUGES
        ]
        depth_rmse = [
            summary["gauges"][key]["depth_metrics_0_20s"]["rmse"] for key in GAUGES
        ]
        reached = int(np.isfinite([np.nan if value is None else value for value in arrival_errors]).sum())
        item = {
            "date": date,
            "label": label,
            "parameters": parameters,
            "gauges_reached": reached,
            "arrival_mae_s": finite_mean(
                [abs(value) if value is not None else np.nan for value in arrival_errors]
            ),
            "coherent_arrival_mae_s": finite_mean(
                [abs(value) if value is not None else np.nan for value in coherent_errors]
            ),
            "mean_depth_rmse_m": finite_mean(depth_rmse),
            "lidar_front_rmse_m": summary["front"]["rmse_m"],
            "lidar_front_bias_m": summary["front"]["bias_m"],
            "maximum_cfl": summary["numerics"]["maximum_cfl"],
        }
        machine.append(item)
        rows.append(
            f"| {date} | {label} | {parameters} | {reached}/4 | "
            f"{item['arrival_mae_s']:.3f} | {item['coherent_arrival_mae_s']:.3f} | "
            f"{item['mean_depth_rmse_m']:.4f} | "
            f"{item['lidar_front_rmse_m']:.3f} | {item['lidar_front_bias_m']:.3f} |"
        )

    report = "\n".join(
        [
            "# D-Claw USGS 2017 水槽验证汇总",
            "",
            "到达 MAE 只统计达到 0.02 m 流深阈值的测点；必须与“达到测点数”一起解释。",
            "",
            "| 试验 | 方案 | 参数 | 达到测点 | 首次到达 MAE (s) | 持续到达 MAE (s) | 平均流深 RMSE (m) | LiDAR 前缘 RMSE，x≤65 m (m) | 前缘偏差 (m) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## 判读",
            "",
            "- 两个基线均稳定完成，但下游首次到达系统性偏早，且多数流深过程的 NSE 小于 0。",
            "- φ=42° 在 05-25 上小幅改善，但在独立的 05-24 试验中只达到 3/4 个测点，不能视为通过验证。",
            "- φ=44° 出现厚流主体滞后与薄前缘仍偏快并存，说明仅调内摩擦角无法同时拟合前缘和流深。",
            "- 论文参数与实测有限开闸显著改善近场到达和峰值；开闸相对瞬时释放使各测点延迟约 0.4–0.6 s。",
            "- 05-25 选出的 k=5×10⁻⁸ m² 在 05-24 留出试验上继续改善到达时间，但持续流体主体的误差大于首次颗粒到达误差，且仍低估下游峰值流深。",
            "- LiDAR 在约 65–70 m 后受遮挡/覆盖限制，前缘指标只统计观测前缘 x≤65 m 的共同时间窗，不能作为最终堆积距离。",
            "- k=5×10⁻⁸ m² 方案只在到达时间目标上优于基线；05-24 的流深和 LiDAR 前缘指标反而变差，因此尚未通过多指标验证。",
            "",
        ]
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "validation_summary.md").write_text(report)
    (COMPARISON / "validation_summary.json").write_text(
        json.dumps(machine, indent=2, ensure_ascii=False) + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
