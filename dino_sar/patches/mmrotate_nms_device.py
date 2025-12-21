from __future__ import annotations

import importlib
from typing import Any, Tuple

import torch


def _fixed_multiclass_nms_rotated(
    multi_bboxes: torch.Tensor,
    multi_scores: torch.Tensor,
    score_thr: float,
    nms: Any,
    max_num: int = -1,
    score_factors: torch.Tensor | None = None,
    return_inds: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fix device mismatch in mmrotate<=0.3.4 `multiclass_nms_rotated`.

    mmrotate creates `labels` on CPU, but `inds` is on the same device as
    `scores`, causing `labels[inds]` to throw when scores are on CUDA.
    """
    from mmcv.ops import nms_rotated

    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    valid_mask = scores > score_thr
    if score_factors is not None:
        score_factors = score_factors.view(-1, 1).expand(multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        return dets, labels

    max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes.clone()
    bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
    _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if return_inds:
        return torch.cat([bboxes, scores[:, None]], 1), labels, keep
    return torch.cat([bboxes, scores[:, None]], 1), labels


def apply() -> None:
    from mmrotate.core import post_processing
    from mmrotate.core.post_processing import bbox_nms_rotated
    import mmrotate.core as core

    # Patch canonical locations.
    bbox_nms_rotated.multiclass_nms_rotated = _fixed_multiclass_nms_rotated  # type: ignore[attr-defined]
    post_processing.multiclass_nms_rotated = _fixed_multiclass_nms_rotated  # type: ignore[attr-defined]
    core.multiclass_nms_rotated = _fixed_multiclass_nms_rotated  # type: ignore[attr-defined]

    # Patch modules that imported the symbol directly (bind at import time).
    for module_name in [
        "mmrotate.models.dense_heads.rotated_anchor_head",
        "mmrotate.models.dense_heads.csl_rotated_retina_head",
        "mmrotate.models.dense_heads.rotated_reppoints_head",
        "mmrotate.models.dense_heads.sam_reppoints_head",
        "mmrotate.models.dense_heads.oriented_reppoints_head",
        "mmrotate.models.dense_heads.csl_rotated_fcos_head",
        "mmrotate.models.dense_heads.rotated_fcos_head",
        "mmrotate.models.roi_heads.bbox_heads.rotated_bbox_head",
        "mmrotate.models.roi_heads.bbox_heads.gv_bbox_head",
    ]:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(mod, "multiclass_nms_rotated"):
            setattr(mod, "multiclass_nms_rotated", _fixed_multiclass_nms_rotated)


apply()
