from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


Point = tuple[float, float]


@dataclass
class SplitSummary:
    split: str
    images: int
    annfiles: int
    annfiles_scanned: int
    missing_images: int
    missing_annfiles: int
    objects: int
    classes: dict[str, int]
    invalid_lines: int
    bowtie_quads: int
    image_count_histogram: dict[str, int]
    area_stats: dict[str, float]
    side_ratio_stats: dict[str, float]
    resolution_sample_stats: dict[str, float]


def _iter_files(dir_path: Path, suffixes: tuple[str, ...]) -> Iterable[Path]:
    if not dir_path.exists():
        return []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in suffixes:
            yield p


def _polygon_area(points: list[Point]) -> float:
    if len(points) < 3:
        return 0.0
    s = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _ccw(a: Point, b: Point, c: Point) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    return _ccw(p1, p3, p4) != _ccw(p2, p3, p4) and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)


def _is_bowtie_quad(points: list[Point]) -> bool:
    if len(points) != 4:
        return False
    p1, p2, p3, p4 = points
    return _segments_intersect(p1, p2, p3, p4) or _segments_intersect(p2, p3, p4, p1)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    values_sorted = sorted(values)
    n = len(values_sorted)

    def q(p: float) -> float:
        if n == 1:
            return values_sorted[0]
        idx = int(round(p * (n - 1)))
        return values_sorted[max(0, min(n - 1, idx))]

    mean = sum(values_sorted) / n
    return {
        "count": float(n),
        "min": float(values_sorted[0]),
        "p25": float(q(0.25)),
        "median": float(q(0.50)),
        "p75": float(q(0.75)),
        "max": float(values_sorted[-1]),
        "mean": float(mean),
    }


def _draw_boxes(image_path: Path, ann_path: Path, out_path: Path) -> None:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    draw = ImageDraw.Draw(img)
    lines = ann_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        coords = list(map(float, parts[:8]))
        label = parts[8]
        pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
        draw.line(pts + [pts[0]], width=2, fill=(255, 64, 64))
        draw.text((pts[0][0] + 2, pts[0][1] + 2), label, fill=(255, 255, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def audit_split(
    data_root: Path,
    split: str,
    rng: random.Random,
    vis_samples: int,
    resolution_samples: int,
    ann_samples: int,
) -> SplitSummary:
    images_dir = data_root / split / "images"
    ann_dir = data_root / split / "annfiles"

    image_files = list(_iter_files(images_dir, (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")))
    ann_files = list(_iter_files(ann_dir, (".txt",)))

    image_map = {p.stem: p for p in image_files}
    ann_map = {p.stem: p for p in ann_files}

    missing_images = sorted(set(ann_map.keys()) - set(image_map.keys()))
    missing_annfiles = sorted(set(image_map.keys()) - set(ann_map.keys()))

    classes: dict[str, int] = {}
    invalid_lines = 0
    bowtie = 0
    per_image_counts: list[int] = []
    areas: list[float] = []
    side_ratios: list[float] = []

    # Scan annotation files (sample by default; set ann_samples=0 to scan all)
    ann_stems = list(ann_map.keys())
    if ann_samples > 0 and ann_samples < len(ann_stems):
        ann_stems = rng.sample(ann_stems, ann_samples)

    for stem in ann_stems:
        ann_path = ann_map[stem]
        lines = ann_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 10:
                invalid_lines += 1
                continue
            try:
                coords = list(map(float, parts[:8]))
            except ValueError:
                invalid_lines += 1
                continue

            label = parts[8]
            classes[label] = classes.get(label, 0) + 1
            count += 1

            pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            if _is_bowtie_quad(pts):
                bowtie += 1

            area = abs(_polygon_area(pts))
            if area > 0:
                areas.append(area)

            side_lengths = [
                math.dist(pts[0], pts[1]),
                math.dist(pts[1], pts[2]),
                math.dist(pts[2], pts[3]),
                math.dist(pts[3], pts[0]),
            ]
            mn = min(side_lengths)
            mx = max(side_lengths)
            if mn > 1e-6:
                side_ratios.append(mx / mn)

        per_image_counts.append(count)

    # Histogram for instances per image (coarse buckets)
    buckets = {"0": 0, "1": 0, "2-5": 0, "6-10": 0, "11-30": 0, "31+": 0}
    for c in per_image_counts:
        if c == 0:
            buckets["0"] += 1
        elif c == 1:
            buckets["1"] += 1
        elif 2 <= c <= 5:
            buckets["2-5"] += 1
        elif 6 <= c <= 10:
            buckets["6-10"] += 1
        elif 11 <= c <= 30:
            buckets["11-30"] += 1
        else:
            buckets["31+"] += 1

    # Resolution stats (sample)
    res_w: list[int] = []
    res_h: list[int] = []
    if resolution_samples > 0 and image_files:
        sample = image_files if resolution_samples >= len(image_files) else rng.sample(image_files, resolution_samples)
        for p in sample:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                res_w.append(w)
                res_h.append(h)
            except Exception:
                continue

    res_stats = {
        "sample_count": float(min(resolution_samples, len(image_files))) if resolution_samples > 0 else 0.0,
        "w_mean": float(sum(res_w) / len(res_w)) if res_w else 0.0,
        "h_mean": float(sum(res_h) / len(res_h)) if res_h else 0.0,
        "w_min": float(min(res_w)) if res_w else 0.0,
        "h_min": float(min(res_h)) if res_h else 0.0,
        "w_max": float(max(res_w)) if res_w else 0.0,
        "h_max": float(max(res_h)) if res_h else 0.0,
    }

    # Visualization (sample)
    if vis_samples > 0:
        candidates = [s for s in ann_map.keys() if s in image_map]
        if candidates:
            sample = candidates if vis_samples >= len(candidates) else rng.sample(candidates, vis_samples)
            for stem in sample:
                img_path = image_map[stem]
                ann_path = ann_map[stem]
                out_path = data_root / "debug_vis" / split / f"{stem}.png"
                try:
                    _draw_boxes(img_path, ann_path, out_path)
                except Exception:
                    continue

    return SplitSummary(
        split=split,
        images=len(image_files),
        annfiles=len(ann_files),
        annfiles_scanned=len(ann_stems),
        missing_images=len(missing_images),
        missing_annfiles=len(missing_annfiles),
        objects=sum(per_image_counts),
        classes=dict(sorted(classes.items(), key=lambda kv: (-kv[1], kv[0]))),
        invalid_lines=invalid_lines,
        bowtie_quads=bowtie,
        image_count_histogram=buckets,
        area_stats=_stats(areas),
        side_ratio_stats=_stats(side_ratios),
        resolution_sample_stats=res_stats,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("."), help="RSAR 数据根目录（默认当前目录）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--vis-samples", type=int, default=200, help="每个 split 可视化抽样数量（0 关闭）")
    parser.add_argument("--resolution-samples", type=int, default=2000, help="每个 split 抽样读取分辨率数量（0 关闭）")
    parser.add_argument(
        "--ann-samples",
        type=int,
        default=5000,
        help="每个 split 抽样解析标注文件数量（0 表示全量解析，可能很慢）",
    )
    parser.add_argument("--report", type=Path, default=Path("outputs/data_audit/rsar_audit.json"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    summaries: list[SplitSummary] = []
    for split in args.splits:
        summaries.append(
            audit_split(
                args.data_root,
                split,
                rng,
                args.vis_samples,
                args.resolution_samples,
                args.ann_samples,
            )
        )

    report = {"data_root": str(args.data_root.resolve()), "summaries": [asdict(s) for s in summaries]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
