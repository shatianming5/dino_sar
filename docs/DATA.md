# 数据说明（RSAR / DOTA-OBB）

**GitHub 仓库（每个阶段结束都要 push）：** https://github.com/shatianming5/dino_sar  
**强制工作流：** 每完成一个阶段（Step）→ `git commit` → `git push origin main`

---

## 1) 目录结构（本地）

本项目默认数据就在仓库根目录下（但 **不提交到 GitHub**）：

```text
./train/
  images/    # 图片（.jpg/.png）
  annfiles/  # 标注（.txt）
./val/
  images/
  annfiles/
./test/
  images/
  annfiles/
```

标注文件为 DOTA 风格，每行：

```text
x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
```

例如：`train/annfiles/0000501.txt`

> 说明：上述 8 点 polygon 本质上就是 OBB（旋转矩形的 4 个顶点）。MMRotate 训练时会将其转换为内部使用的 5 参数表示 `(xc, yc, w, h, angle)`。

## 1b) 类别（RSAR 6 类）

本仓库默认按 RSAR 官方 6 类训练/评估（类别名需与标注文本一致，均为小写）：

- `aircraft`
- `bridge`
- `car`
- `harbor`
- `ship`
- `tank`

---

## 2) 数据不进 Git（必须遵守）

仓库已在 `.gitignore` 中忽略 `train/`、`val/`、`test/`，防止误提交大文件与数据集版权风险。

---

## 3) 数据体检（必须先做）

运行体检脚本：

```bash
python tools/data_audit/rsar_dota_audit.py --data-root . --vis-samples 200
```

默认会对每个 split 抽样解析一部分标注文件（`--ann-samples`，默认 5000）以保证速度；如果你想做全量解析可用：

```bash
python tools/data_audit/rsar_dota_audit.py --data-root . --ann-samples 0 --vis-samples 200
```

输出内容：
- 文件完整性：图片与标注是否一一对应、缺失列表
- 标注合法性：每行字段数、数值可解析、是否出现自交“蝴蝶结框”
- 统计：每张图实例数、类别计数、面积与长宽比（近似）分布、图像分辨率（抽样）
- 可视化：随机抽样画框到 `debug_vis/`（该目录已忽略，不提交）
