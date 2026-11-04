#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 MVTec AD 各类别缺陷掩码的平均面积占比。

对应论文图 2 的横轴数据：缺陷越稀疏的类别，像素级与图像级 AUROC 的差距越大。

用法：
    python scripts/analyze_gap.py --data /path/to/mvtec_ad
    python scripts/analyze_gap.py --data /path/to/mvtec_ad --csv areas.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def collect(data_dir: Path):
    rows = []
    for cdir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        gt = cdir / "ground_truth"
        if not gt.exists():
            continue
        areas = []
        for mask_path in sorted(gt.rglob("*_mask.png")):
            mask = np.array(Image.open(mask_path).convert("L"))
            areas.append(float((mask > 0).mean() * 100.0))
        if not areas:
            continue
        rows.append({
            "class": cdir.name,
            "n_defect": len(areas),
            "mean_area_pct": round(float(np.mean(areas)), 2),
            "median_area_pct": round(float(np.median(areas)), 2),
            "max_area_pct": round(float(np.max(areas)), 2),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="MVTec AD 根目录")
    ap.add_argument("--csv", default=None, help="可选：把结果写入 CSV")
    args = ap.parse_args()

    rows = collect(Path(args.data))
    if not rows:
        raise SystemExit("未找到任何 ground_truth/*_mask.png，请检查 --data 路径")

    header = ["class", "n_defect", "mean_area_pct", "median_area_pct", "max_area_pct"]
    print("".join(f"{h:>16}" for h in header))
    for r in rows:
        print("".join(f"{str(r[h]):>16}" for h in header))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n已写入 {args.csv}")


if __name__ == "__main__":
    main()
