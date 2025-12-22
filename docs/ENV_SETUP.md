# 环境搭建（主线 MMRotate + 兜底 dinov3_stack）

**GitHub 仓库（每个阶段结束都要 push）：** https://github.com/shatianming5/dino_sar  
**强制工作流：** 每完成一个阶段（Step）→ `git commit` → `git push origin main`

---

## 1) 主线环境：MMRotate（DOTA/OBB）

> 说明：目前 **PyPI 上的 `mmrotate` 最新版是 `0.3.x`**，属于 OpenMMLab 旧栈，依赖 **`mmcv-full` + `mmdet<3.0.0`**。  
> 为避免版本互相踩踏，本项目主线环境以 **`mmrotate==0.3.4`** 为准。

### 1.1 创建 conda 环境

在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate dino_sar
```

### 1.2 安装 OpenMMLab 依赖

> 说明：
> - `mmcv-full` 必须用 `openmim` 装（自动匹配 torch/cuda，对应带 C++/CUDA ops）
> - 为了拿到稳定的 `mmcv-full` 预编译 wheel，这里先固定 PyTorch 到 `cu118`（CUDA 11.8）

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

pip install -U openmim
mim install -U "mmcv-full==1.7.2"
pip install -U "mmdet>=2.25.1,<3.0.0" "mmrotate==0.3.4"
```

### 1.3 冒烟检查（必须通过）

```bash
python tools/smoke_test/check_openmmlab.py
```

---

## 2) 兜底环境：dinov3_stack（快速验证）

### 2.1 准备（不提交到 Git）

- 克隆 `dinov3_stack`（建议放到仓库外或 `third_party/`，但不要提交）
- 克隆 `dinov3` 官方仓库并准备权重文件
- 在本机创建 `.env`（不提交），写入（示例）：

```env
DINOv3_REPO="/abs/path/to/dinov3"
DINOv3_WEIGHTS="/abs/path/to/dinov3/weights"
```

### 2.2 验收

进入 `dinov3_stack` 目录后：

```bash
python train_detection.py --help
```

---

## 3) Step5：DINOv3（集成到 MMRotate）

> 说明：
> - 主线环境固定在 `torch==2.0.1` + `mmrotate==0.3.4`（旧栈），DINOv3 官方 `hubconf.py` 会触发与更新版 torch 相关的 import
> - 同时，DINOv3 官方权重 URL（`dl.fbaipublicfiles.com`）在部分网络环境会出现 **403 Forbidden**
>
> 所以这里提供两种接入方式：**推荐（timm 权重）** / **可选（官方 dinov3 仓库 + 本地权重）**。

### 3.1 推荐：timm 权重（无需克隆 dinov3）

```bash
pip install "timm>=1.0.17"
```

配置使用：

- `configs/dinov3_frozen/retinanet_dinov3_timm_convnext_small_fpn_rsar_le90_frozen_2kiter.py`
- backbone：`Dinov3TimmConvNeXt`（从 HuggingFace `timm/*` 下载 `pytorch_model.bin`，不需要 token）

### 3.2 可选：官方 dinov3 仓库 + 本地权重

```bash
git clone https://github.com/facebookresearch/dinov3 third_party/dinov3
export DINOV3_REPO_DIR="$(pwd)/third_party/dinov3"
```

- `Dinov3ConvNeXt` 通过“直接加载 `dinov3/models/convnext.py`”绕开 `hubconf.py`
- 由于官方 URL 可能 403，建议把权重文件下载到本地后，在配置里把 `model.backbone.weights` 指向本地 `.pth`
