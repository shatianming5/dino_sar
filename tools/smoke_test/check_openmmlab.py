from __future__ import annotations

import importlib
import sys


def _import_and_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "unknown")


def main() -> int:
    print(f"python: {sys.version.split()[0]}")

    try:
        import torch  # noqa: F401
    except Exception as exc:
        print(f"[FAIL] import torch: {exc}")
        return 1

    import torch

    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    for name in ["mmcv", "mmdet", "mmrotate"]:
        try:
            ver = _import_and_version(name)
            print(f"{name}: {ver}")
        except Exception as exc:
            print(f"[FAIL] import {name}: {exc}")
            return 1

    try:
        from mmcv import ops as mmcv_ops  # noqa: F401
    except Exception as exc:
        print(f"[FAIL] import mmcv.ops: {exc}")
        return 1

    try:
        from mmcv.ops import box_iou_rotated

        a = torch.tensor([[0.0, 0.0, 10.0, 5.0, 0.0]], device="cpu")
        b = torch.tensor([[0.0, 0.0, 10.0, 5.0, 0.0]], device="cpu")
        iou = box_iou_rotated(a, b)
        print(f"mmcv.ops.box_iou_rotated OK: {iou.item():.4f}")
    except Exception as exc:
        print(f"[FAIL] mmcv.ops rotated op check: {exc}")
        return 1

    print("[OK] MMRotate stack looks usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
