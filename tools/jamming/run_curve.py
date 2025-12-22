from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.stdout


def _parse_map(stdout: str) -> float | None:
    # mmrotate prints: {'mAP': 0.123}
    m = re.search(r"\{'mAP':\s*([0-9.eE+-]+)\}", stdout)
    return float(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--subset-dir", default="outputs/datasets/rsar_val_200")
    ap.add_argument("--sigmas", default="0,0.02,0.05,0.1,0.2")
    ap.add_argument("--out", default="outputs/robustness")
    args = ap.parse_args()

    subset = Path(args.subset_dir)
    ann = subset / "annfiles"
    img = subset / "images"
    if not ann.exists() or not img.exists():
        raise SystemExit(f"Missing subset dataset under: {subset} (run tools/jamming/make_subset.py first)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip() != ""]
    records: list[dict] = []

    for sigma in sigmas:
        work_dir = out_dir / f"eval_sigma_{sigma:g}"
        cmd = [
            "mim",
            "test",
            "mmrotate",
            args.config,
            "--checkpoint",
            args.ckpt,
            "--gpus",
            str(args.gpus),
            "--eval",
            "mAP",
            "--work-dir",
            str(work_dir),
            "--cfg-options",
            f"data.test.ann_file={ann}",
            f"data.test.img_prefix={img}",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{Path.cwd()}:{env.get('PYTHONPATH','')}"
        env["DINO_SAR_JAM_SIGMA"] = str(sigma)
        out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env).stdout
        mAP = _parse_map(out)
        if mAP is None:
            raise RuntimeError("Failed to parse mAP from output")
        records.append({"sigma": sigma, "mAP": mAP})
        print(f"sigma={sigma:g} mAP={mAP:.4f}")

    (out_dir / "curve.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(str(out_dir / "curve.json"))


if __name__ == "__main__":
    main()
