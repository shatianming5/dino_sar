# RSAR 6-class mAP 偏低：根因定位与下一步建议

## 结论（先给一句话）

当前 10k iters 的 RSAR 6-class mAP@0.5 偏低，核心原因是 **按“图片级”统计的长尾极端**：`aircraft/car/tank/harbor` 在 `train/` 中只出现在约 **0.8%~1.7%** 的图片里，10k iter（batch=1）训练期望只见到 **~80~170 张**“含稀有类”的训练图，导致稀有类几乎学不起来。

## 证据 1：训练集图片级类别覆盖极低

统计方式：对每个 split，统计“包含该类别的 annfile 数”（每张图一个 `*.txt`）。

```bash
for split in train val test; do
  echo "== $split ==";
  for cls in aircraft bridge car harbor ship tank; do
    rg -l -w "$cls" $split/annfiles | wc -l
  done
done
```

结果（`train/` 一共 78837 张）：

| class | images containing class |
|---|---:|
| ship | 61113 |
| bridge | 15110 |
| harbor | 1327 |
| car | 837 |
| aircraft | 795 |
| tank | 659 |

换算比例（train）：

- `aircraft`：`795 / 78837 ≈ 1.01%`
- `car`：`837 / 78837 ≈ 1.06%`
- `tank`：`659 / 78837 ≈ 0.84%`
- `harbor`：`1327 / 78837 ≈ 1.68%`

## 证据 2：10k iter 对稀有类曝光过少（数量级不够）

以 `batch=1` 近似，10k iter 期望看到的“含稀有类图片”数量：

- `aircraft`：`10000 * 1.01% ≈ 101 张`
- `tank`：`10000 * 0.84% ≈ 84 张`
- `harbor`：`10000 * 1.68% ≈ 168 张`

这个曝光量下，出现“稀有类 AP=0 或接近 0”是符合预期的。

## 证据 3：把训练拉长到 100k，mAP 直接翻倍以上（链路正确、确实是曝光不足）

Baseline R50-FPN（100k iters，test mAP@0.5）多 seed：

- 记录：`docs/repro_runs/2025-12-25_6class_scorethr0p05_multiseed_baseline_r50_100k.md`
- 结果：`0.2296 ± 0.0062`（对比 10k 的 `0.1008 ± 0.0203`）

## 下一步建议（按收益/成本排序）

1. **继续推进长训练（>=100k）**：尤其是 LoRA/ConvNeXt 等分支，先做 seed0，再补齐 seed1/2。
2. **加大“图片级”过采样力度**：`ClassBalancedDataset` 的 `oversample_thr=0.1` 对 1% 类别只会带来 ~3x 重复，可能仍不够；可考虑更激进阈值（如 0.3/0.5）或专门针对 `tank/aircraft/car/harbor` 的采样策略。
3. **针对 tank 专项排查**：100k baseline 里 `tank` 仍可能是 0 AP（样本极少且目标很小/易混淆）。建议做：
   - 统计 `tank` 的尺寸分布（w/h/area）与 anchor 覆盖；
   - 提高输入分辨率或增大小目标召回的 anchor 设置。

