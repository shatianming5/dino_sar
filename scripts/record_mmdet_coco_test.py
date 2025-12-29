#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path


_BBOX_COPypaste_RE = re.compile(
    r"""bbox_mAP_copypaste:\s*
        (?P<map>[0-9.]+)\s+
        (?P<ap50>[0-9.]+)\s+
        (?P<ap75>[0-9.]+)
    """,
    re.VERBOSE,
)


def _parse_last_bbox_metrics(log_path: Path) -> tuple[int, str, str, str]:
    last = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            m = _BBOX_COPypaste_RE.search(line)
            if m:
                last = (idx, m.group("map"), m.group("ap50"), m.group("ap75"))
    if last is None:
        raise SystemExit(f"Could not find bbox_mAP_copypaste in log: {log_path}")
    return last


def _parse_classwise_map75_table(log_path: Path) -> dict[str, tuple[int, str]]:
    # CocoMetric(classwise=True) prints an AsciiTable with headers:
    # category | mAP | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l
    last_table: dict[str, tuple[int, str]] = {}
    in_table = False
    cat_col = None
    ap75_col = None
    current: dict[str, tuple[int, str]] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            s = line.strip()

            if s.startswith("|") and "category" in s and "mAP_75" in s:
                headers = [c.strip() for c in s.split("|")[1:-1]]
                try:
                    cat_col = headers.index("category")
                    ap75_col = headers.index("mAP_75")
                except ValueError:
                    cat_col = None
                    ap75_col = None
                    continue
                current = {}
                in_table = True
                continue

            if not in_table:
                continue

            if s.startswith("+"):
                continue

            if not (s.startswith("|") and s.endswith("|")):
                if current:
                    last_table = current
                in_table = False
                continue

            cols = [c.strip() for c in s.split("|")[1:-1]]
            if cat_col is None or ap75_col is None:
                continue
            if len(cols) <= max(cat_col, ap75_col):
                continue

            cls = cols[cat_col]
            ap75 = cols[ap75_col]
            if cls and cls != "category":
                current[cls] = (idx, ap75)

    if current:
        last_table = current
    return last_table


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
        description="Record mmdet CocoMetric bbox test metrics into dino_sar/README.md"
    )
    parser.add_argument(
        "--log",
        required=True,
        type=Path,
        help="MSFA/mmdet tools/test.py log file (*.log)",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Target README.md (default: repo root README.md)",
    )
    parser.add_argument(
        "--header",
        default="**SARDet-100K（MSFA / mmdet）单次训练评估**",
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
        help="Comma-separated class names to record AP75 for (from classwise table)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    readme_path = args.readme or (repo_root / "README.md")

    log_path = args.log.expanduser()
    line_no, map_v, ap50_v, ap75_v = _parse_last_bbox_metrics(log_path)

    ap75_table = _parse_classwise_map75_table(log_path)
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
        f"- Test（测试集）（我已用 tools/test.py 跑完）：coco/bbox_mAP={map_v}，AP50={ap50_v}，"
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

