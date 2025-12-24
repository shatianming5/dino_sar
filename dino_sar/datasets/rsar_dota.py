from __future__ import annotations

import glob
import hashlib
import os
import os.path as osp
import pickle
from typing import Any

import numpy as np

from mmrotate.core import obb2poly_np, poly2obb_np
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset


@ROTATED_DATASETS.register_module()
class RSARDOTADataset(DOTADataset):
    """RSAR in DOTA polygon format (OBB), with mixed image suffixes.

    Differences vs upstream DOTADataset:
    - Supports all RSAR classes (6 classes by default)
    - Resolves image filename by scanning `img_prefix` for common suffixes
      (RSAR contains both .jpg and .png).
    """

    # RSAR official labels (6 classes).
    # NOTE: Keep names in lower-case to match annotation text.
    CLASSES = ("aircraft", "bridge", "car", "harbor", "ship", "tank")
    PALETTE = [
        (255, 64, 64),    # aircraft
        (64, 255, 64),    # bridge
        (64, 64, 255),    # car
        (255, 255, 64),   # harbor
        (255, 64, 255),   # ship
        (64, 255, 255),   # tank
    ]

    def __init__(
        self,
        *args: Any,
        cache_dir: str | None = None,
        cache_refresh: bool = False,
        **kwargs: Any,
    ):
        self.cache_dir = cache_dir
        self.cache_refresh = cache_refresh
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_folder: str):
        img_prefix = self.img_prefix
        if isinstance(img_prefix, dict):
            img_prefix = img_prefix.get("img", "")

        cache_path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_key = "|".join(
                [
                    osp.abspath(ann_folder),
                    osp.abspath(img_prefix),
                    str(getattr(self, "version", "")),
                    str(getattr(self, "difficulty", "")),
                    str(getattr(self, "filter_empty_gt", "")),
                    ",".join(self.CLASSES),
                ]
            )
            cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:16]
            cache_path = osp.join(self.cache_dir, f"rsar_dota_{cache_hash}.pkl")

        if cache_path and osp.exists(cache_path) and not self.cache_refresh:
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
                data_infos = payload["data_infos"] if isinstance(payload, dict) else payload
                self.img_ids = (
                    payload.get("img_ids")  # type: ignore[union-attr]
                    if isinstance(payload, dict) and payload.get("img_ids") is not None
                    else [osp.splitext(info["filename"])[0] for info in data_infos]
                )
                try:
                    from mmcv.utils import print_log

                    print_log(f"[RSARDOTADataset] Loaded cache: {cache_path}", logger="root")
                except Exception:
                    pass
                return data_infos
            except Exception:
                pass

        ann_files = sorted(glob.glob(osp.join(ann_folder, "*.txt")))

        img_files = []
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            img_files.extend(glob.glob(osp.join(img_prefix, f"*{ext}")))

        img_map = {osp.splitext(osp.basename(p))[0]: osp.basename(p) for p in img_files}

        cls_map = {c: i for i, c in enumerate(self.CLASSES)}
        data_infos = []

        for ann_file in ann_files:
            data_info: dict = {}
            img_id = osp.splitext(osp.basename(ann_file))[0]
            data_info["filename"] = img_map.get(img_id, img_id + ".png")
            data_info["ann"] = {}

            gt_bboxes = []
            gt_labels = []
            gt_polygons = []

            if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                continue

            with open(ann_file, encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    bbox_info = raw.split()
                    if len(bbox_info) < 6:
                        continue

                    # RSAR annotations are commonly in DOTA style:
                    #   x1 y1 x2 y2 x3 y3 x4 y4 class [difficulty]
                    # But some conversions may store OBB directly:
                    #   xc yc w h angle class [difficulty]
                    poly = None
                    if len(bbox_info) >= 9:
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        cls_name = bbox_info[8].lower()
                        difficulty = int(bbox_info[9]) if len(bbox_info) > 9 else 0
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except Exception:
                            continue
                    else:
                        try:
                            x, y, w, h, a = map(float, bbox_info[:5])
                        except Exception:
                            continue
                        cls_name = bbox_info[5].lower()
                        difficulty = int(bbox_info[6]) if len(bbox_info) > 6 else 0
                        try:
                            poly = obb2poly_np(
                                np.array([[x, y, w, h, a]], dtype=np.float32), version=self.version
                            )
                            poly = np.asarray(poly, dtype=np.float32).reshape(-1)
                            if poly.shape[0] != 8:
                                poly = None
                        except Exception:
                            poly = None
                    if difficulty > self.difficulty:
                        continue

                    if cls_name not in cls_map:
                        continue

                    gt_bboxes.append([x, y, w, h, a])
                    gt_labels.append(cls_map[cls_name])
                    if poly is not None:
                        gt_polygons.append(np.array(poly, dtype=np.float32))

            if gt_bboxes:
                data_info["ann"]["bboxes"] = np.array(gt_bboxes, dtype=np.float32)
                data_info["ann"]["labels"] = np.array(gt_labels, dtype=np.int64)
                data_info["ann"]["polygons"] = np.array(gt_polygons, dtype=np.float32) if gt_polygons else np.zeros((0, 8), dtype=np.float32)
            else:
                data_info["ann"]["bboxes"] = np.zeros((0, 5), dtype=np.float32)
                data_info["ann"]["labels"] = np.array([], dtype=np.int64)
                data_info["ann"]["polygons"] = np.zeros((0, 8), dtype=np.float32)

            data_info["ann"]["bboxes_ignore"] = np.zeros((0, 5), dtype=np.float32)
            data_info["ann"]["labels_ignore"] = np.array([], dtype=np.int64)
            data_info["ann"]["polygons_ignore"] = np.zeros((0, 8), dtype=np.float32)

            data_infos.append(data_info)

        self.img_ids = [osp.splitext(info["filename"])[0] for info in data_infos]

        if cache_path:
            try:
                payload = {"data_infos": data_infos, "img_ids": self.img_ids}
                tmp_path = f"{cache_path}.{os.getpid()}.tmp"
                with open(tmp_path, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, cache_path)
                try:
                    from mmcv.utils import print_log

                    print_log(f"[RSARDOTADataset] Saved cache: {cache_path}", logger="root")
                except Exception:
                    pass
            except Exception:
                pass

        return data_infos
