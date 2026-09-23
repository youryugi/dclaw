"""Parse the SGM rough-bed aggregate dataset (2010 JGR paper supplement).

dataset/usgs_flume_2010/jgrf703-sup-0004-ds03.txt lists three cross
sections (x = 32, 66, 90 m) in succession, each with its own header block
and a tab-delimited table of t, h, SD h, sigma, SD sigma, pbed, SD pbed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DS03_PATH = REPO / "dataset" / "usgs_flume_2010" / "jgrf703-sup-0004-ds03.txt"


def load_sections(path: Path = DS03_PATH) -> dict[float, dict[str, np.ndarray]]:
    lines = path.read_text(errors="replace").splitlines()
    section_starts = [
        i for i, line in enumerate(lines) if line.strip().startswith("Cross section:")
    ]
    sections: dict[float, dict[str, np.ndarray]] = {}
    for start_index, start in enumerate(section_starts):
        position = float(lines[start].split("=")[1].split("m")[0].strip())
        header_index = next(
            i
            for i in range(start, len(lines))
            if lines[i].strip().startswith("t(s)")
        )
        end = (
            section_starts[start_index + 1]
            if start_index + 1 < len(section_starts)
            else len(lines)
        )
        rows = []
        for line in lines[header_index + 1 : end]:
            if not line.strip():
                continue
            fields = [f for f in line.split("\t")]
            values = [float(f) for f in fields if f.strip() != ""]
            if len(values) == 7:
                rows.append(values)
        data = np.asarray(rows, dtype=float)
        sections[position] = {
            "t_s": data[:, 0],
            "h_m": data[:, 1],
            "h_sd_m": data[:, 2],
            "sigma_kpa": data[:, 3],
            "sigma_sd_kpa": data[:, 4],
            "pbed_kpa": data[:, 5],
            "pbed_sd_kpa": data[:, 6],
        }
    return sections


if __name__ == "__main__":
    sections = load_sections()
    for position, table in sections.items():
        print(
            f"x={position:g} m: t in [{table['t_s'].min():.2f}, {table['t_s'].max():.2f}] s,"
            f" n={table['t_s'].size}, peak h={table['h_m'].max():.3f} m"
        )
