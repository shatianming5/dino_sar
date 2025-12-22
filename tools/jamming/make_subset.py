from __future__ import annotations

import argparse
import os
import os.path as osp
from pathlib import Path


def _find_image_path(images_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=str, default="outputs/datasets")
    args = ap.parse_args()

    split = args.split
    n = int(args.n)
    out_root = Path(args.out)

    ann_dir = Path(split) / "annfiles"
    img_dir = Path(split) / "images"
    if not ann_dir.exists() or not img_dir.exists():
        raise SystemExit(f"Missing split dirs: {ann_dir} or {img_dir}")

    out_dir = out_root / f"rsar_{split}_{n}"
    out_ann = out_dir / "annfiles"
    out_img = out_dir / "images"
    out_ann.mkdir(parents=True, exist_ok=True)
    out_img.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(p for p in ann_dir.glob("*.txt"))
    if not ann_files:
        raise SystemExit(f"No annfiles under {ann_dir}")

    picked = ann_files[:n]
    (out_dir / "_list.txt").write_text("\n".join(p.name for p in picked) + "\n", encoding="utf-8")

    for ann_path in picked:
        stem = ann_path.stem
        img_path = _find_image_path(img_dir, stem)
        if img_path is None:
            continue
        os.symlink(osp.abspath(ann_path), out_ann / ann_path.name) if not (out_ann / ann_path.name).exists() else None
        os.symlink(osp.abspath(img_path), out_img / img_path.name) if not (out_img / img_path.name).exists() else None

    print(str(out_dir))


if __name__ == "__main__":
    main()

