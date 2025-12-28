#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path


_TEST_METRIC_RE = re.compile(
    r"""Epoch\(test\).*
        dota/mAP:\s*(?P<map>[0-9.]+).*
        dota/AP50:\s*(?P<ap50>[0-9.]+).*
        dota/AP75:\s*(?P<ap75>[0-9.]+)
    """,
    re.VERBOSE,
)


def _parse_last_test_metrics(log_path: Path) -> tuple[int, str, str, str]:
    last = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            m = _TEST_METRIC_RE.search(line)
            if m:
                last = (idx, m.group("map"), m.group("ap50"), m.group("ap75"))
    if last is None:
        raise SystemExit(f"Could not find test metrics in log: {log_path}")
    return last


_IOU_THR_RE = re.compile(r"iou_thr:\s*(?P<thr>[0-9.]+)")


def _parse_iou_thr_class_ap(log_path: Path,
                            iou_thr: float) -> dict[str, tuple[int, str]]:
    current_thr = None
    table: dict[str, tuple[int, str]] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            m = _IOU_THR_RE.search(line)
            if m:
                current_thr = float(m.group("thr"))
                continue

            if current_thr is None or abs(current_thr - iou_thr) > 1e-9:
                continue

            s = line.strip()
            if not (s.startswith("|") and s.endswith("|")):
                continue
            if s.startswith("| class"):
                continue
            if s.startswith("| mAP"):
                break

            cols = [c.strip() for c in s.split("|")[1:-1]]
            if len(cols) < 2:
                continue
            cls = cols[0]
            ap = cols[-1]
            if cls:
                table[cls] = (idx, ap)
    return table


def _rel_ref(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def _upsert_record(readme_path: Path, header: str, record_line: str,
                   log_ref: str) -> bool:
    lines = readme_path.read_text(encoding="utf-8").splitlines(keepends=True)
    try:
        header_idx = next(i for i, l in enumerate(lines) if header in l)
    except StopIteration:
        raise SystemExit(f'Header not found in {readme_path}: "{header}"')

    if record_line in "".join(lines):
        return False

    # If this log_ref already exists, update that line in-place to avoid
    # duplicating records for the same run.
    log_ref_token = f"`{log_ref}`"
    for i, line in enumerate(lines):
        if log_ref_token in line:
            lines[i] = record_line + "\n"
            readme_path.write_text("".join(lines), encoding="utf-8")
            return True

    insert_at = header_idx + 1
    lines.insert(insert_at, record_line + "\n")
    readme_path.write_text("".join(lines), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record RSAR MMRotate1.x test metrics into dino_sar/README.md"
    )
    parser.add_argument(
        "--log",
        required=True,
        type=Path,
        help="RSAR MMRotate1.x tools/test.py log file (*.log)",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Target README.md (default: repo root README.md)",
    )
    parser.add_argument(
        "--header",
        default="**RSAR（MMRotate1.x / RSAR 主工程）单次训练评估**",
        help="Markdown header line to insert under",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional note appended to the record line (e.g. run tag)",
    )
    parser.add_argument(
        "--ap75-classes",
        default="tank,bridge,harbor",
        help="Comma-separated class names to record AP75 for",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    readme_path = args.readme or (repo_root / "README.md")

    log_path = args.log.expanduser()
    line_no, map_v, ap50_v, ap75_v = _parse_last_test_metrics(log_path)

    ap75_table = _parse_iou_thr_class_ap(log_path, iou_thr=0.75)
    ap75_classes = [c.strip() for c in args.ap75_classes.split(",") if c.strip()]
    ap75_parts = []
    for cls in ap75_classes:
        if cls not in ap75_table:
            continue
        _, ap = ap75_table[cls]
        ap75_parts.append(f"{cls}={ap}")
    ap75_cls = f"（AP75分类: {', '.join(ap75_parts)}）" if ap75_parts else ""

    rasr_root = repo_root.parent
    log_ref = f"{_rel_ref(log_path, rasr_root)}:{line_no}"

    note = f" {args.note.strip()}" if args.note.strip() else ""
    record_line = (
        f"- Test（测试集）（我已用 tools/test.py 跑完）：dota/mAP={map_v}，AP50={ap50_v}，"
        f"AP75={ap75_v}{ap75_cls}（见 `{log_ref}`）{note}"
    )

    changed = _upsert_record(readme_path, args.header, record_line, log_ref)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(
        f"changed={str(changed).lower()} map={map_v} ap50={ap50_v} ap75={ap75_v} "
        f"log_ref={log_ref} time={now}"
    )


if __name__ == "__main__":
    main()
