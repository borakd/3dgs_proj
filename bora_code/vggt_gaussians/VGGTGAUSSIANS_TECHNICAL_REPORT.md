# VGGTGaussians Technical Report
## Comprehensive Code Analysis and Component Significance

**Date:** December 2024  
**Purpose:** Detailed explanation of every code component in the VGGTGaussians implementation

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Philosophy](#architecture-philosophy)
3. [File-by-File Analysis](#file-by-file-analysis)
   - [main_cursor.py](#main_cursorpy)
   - [train_cursor.py](#train_cursorpy)
   - [inference_cursor.py](#inference_cursorpy)
   - [check_outputs_vggt_gaussians.py](#check_outputs_vggt_gaussianspy)
   - [vggt_gaussians.yaml](#vggt_gaussiansyaml)
4. [Key Design Decisions](#key-design-decisions)
5. [Data Flow](#data-flow)
6. [Training Pipeline](#training-pipeline)
7. [Inference Pipeline](#inference-pipeline)
8. [Visualization Pipeline](#visualization-pipeline)

---

## Overview

VGGTGaussians is a 3D Gaussian Splatting model that extends the Vision-Guided Geometry Transformer (VGGT) backbone with a trainable Gaussian parameter prediction head. The model follows the Splatt3R paradigm: a frozen backbone (VGGT) extracts rich 3D scene representations, while a trainable DPT (Dense Prediction Transformer) head predicts Gaussian parameters (means, scales, rotations, spherical harmonics, opacities) for novel view synthesis.

**Key Innovation:** Unlike Splatt3R which uses MASt3R, VGGTGaussians uses VGGT as the frozen backbone, providing potentially richer geometric understanding through VGGT's unified transformer architecture.

---

## Architecture Philosophy

### Core Principle: Frozen Backbone + Trainable Head

The entire design is built around a fundamental constraint:
- **VGGT backbone**: Completely frozen (`requires_grad_(False)`, `eval()` mode)
- **Gaussian DPT head**: Fully trainable, receives aggregated tokens from VGGT
- **PixelSplat decoder**: Used for rendering but not trainable (deterministic CUDA rasterizer)

This design allows:
1. **Efficient training**: Only ~15M parameters (Gaussian head) are trainable vs. ~1B (VGGT)
2. **Rich features**: VGGT's pretrained weights provide strong geometric priors
3. **Modularity**: Gaussian head can be swapped/improved without retraining VGGT

---

## File-by-File Analysis

### main_cursor.py

**Purpose:** Core model implementation (`VGGTGaussians` Lightning module)

#### Class: `VGGTGaussians(L.LightningModule)`

**Inheritance:** PyTorch Lightning's `LightningModule` provides:
- Automatic training loop integration
- Checkpoint saving/loading
- Hyperparameter logging
- Multi-GPU support

#### `__init__(self, config)`

**Lines 58-203:** Model initialization

**Config Normalization (Lines 62-104):**
```python
if isinstance(config, omegaconf.DictConfig):
    # Convert OmegaConf to Config dataclass
```
**Significance:** Handles multiple config formats (OmegaConf YAML, dataclass, dict) ensuring compatibility with both Splatt3R's config system and direct Python usage. This flexibility is crucial for:
- Loading from YAML files (training)
- Direct instantiation (testing)
- Checkpoint loading (inference)

**Resolution Handling (Lines 71-80):**
```python
resolution_raw = config_dict.get('data', {}).get('resolution', 518)
if isinstance(resolution_raw, (int, float)):
    render_res = (int(resolution_raw), int(resolution_raw))
```
**Significance:** VGGT requires image dimensions divisible by 14 (patch size). 518 is chosen because:
- 518 ÷ 14 = 37 (exact division)
- 512 ÷ 14 = 36.57... (not divisible)
- Ensures no padding artifacts in patch extraction

**VGGT Loading (Lines 106-114):**
```python
self.vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
self.vggt.requires_grad_(False)
self.vggt.eval()
```
**Significance:**
- `from_pretrained`: Loads 1B-parameter model from HuggingFace
- `requires_grad_(False)`: Disables gradient computation (memory + speed)
- `eval()`: Sets BatchNorm/Dropout to inference mode AND ensures `images` are stored in predictions (VGGT's `eval()` mode stores input images for visualization)

**Gaussian Head Definition (Lines 122-143):**
```python
gaussian_num_channels = 3 + 3 + 4 + 3 * sh_degree + 1
# 3 (offsets) + 3 (scales) + 4 (rotations) + 3*sh_degree (SH) + 1 (opacity)
self.gaussian_dpt = DPTHead(
    dim_in=2 * embed_dim,  # 2 views concatenated
    output_dim = gaussian_num_channels + 1,  # +1 for confidence channel
)
```
**Significance:**
- `dim_in=2 * embed_dim`: DPT head receives concatenated tokens from both views (VGGT aggregator outputs `[B, S, embed_dim]` per view, concatenated to `[B, S, 2*embed_dim]`)
- `output_dim = gaussian_num_channels + 1`: Extra channel for implicit confidence (used by DPT head internally, not in final predictions)
- `gaussian_num_channels`: Total channels needed for all Gaussian parameters

**Weight Initialization (Lines 145-164):**
```python
splits_and_inits = [
    (3, 0.001, 0.001),      # Offsets: small scale, small bias
    (3, 0.00003, -7.0),     # Scales: tiny scale, large negative bias (log-space)
    (4, 1.0, 0.0),          # Rotations: normal scale, zero bias
    (3 * sh_degree, 1.0, 0.0),  # SH: normal scale, zero bias
    (1, 1.0, -2.0)          # Opacity: normal scale, negative bias (sigmoid space)
]
```
**Significance:**
- **Offsets (0.001)**: Small initial offsets (Gaussians start near VGGT's world points)
- **Scales (-7.0 bias)**: Large negative bias → after `exp()` in `reg_dense_scales`, initial scales are tiny (`exp(-7) ≈ 0.001`). This prevents degenerate Gaussians (too large = blurry, too small = point-like)
- **Rotations (0.0 bias)**: Zero bias → quaternions initialized near identity
- **SH (0.0 bias)**: Zero bias → color starts neutral (will be seeded with RGB2SH of input)
- **Opacity (-2.0 bias)**: Negative bias → after sigmoid, initial opacity is low (`sigmoid(-2) ≈ 0.12`), preventing overconfident predictions

**Decoder Initialization (Lines 166-170):**
```python
self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
    background_color=[0.0, 0.0, 0.0]
)
```
**Significance:** PixelSplat's CUDA rasterizer for rendering Gaussians. Black background matches training data (ScanNet++ has black regions).

**LPIPS Criterion (Line 178):**
```python
self.lpips_criterion = lpips.LPIPS('vgg')
```
**Significance:** Perceptual loss using VGG features. Captures high-level structure better than MSE alone.

**Hyperparameter Saving (Lines 180-202):**
```python
hparams_dict = {
    'image_size': self.config.image_size,
    # ... convert Config to plain dict
}
self.save_hyperparameters(hparams_dict)
```
**Significance:** Lightning saves hyperparameters in checkpoint for later reconstruction. Converting dataclass to dict avoids OmegaConf serialization issues.

#### `forward(self, views)`

**Lines 205-406:** Main forward pass

**Input Handling (Lines 208-211):**
```python
if len(views.shape) == 4: views = views.unsqueeze(0)
B_imgs, S_imgs, _, H_img, W_img = views.shape
```
**Significance:** Handles both `[S, C, H, W]` (single batch) and `[B, S, C, H, W]` (batched) inputs.

**Frozen VGGT Inference (Lines 213-223):**
```python
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        frozen_vggt_preds = self.vggt(views)
        aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(views)
```
**Significance:**
- `torch.no_grad()`: Disables gradient tracking (VGGT is frozen, no need for gradients)
- `autocast(dtype=bfloat16)`: Mixed precision for memory efficiency (1B params → ~2GB in bfloat16 vs ~4GB in float32)
- `self.vggt(views)`: Gets `world_points`, `depth`, `pose_enc`, `images` (stored in eval mode)
- `self.vggt.aggregator(views)`: Gets aggregated tokens `[B, S, embed_dim]` and patch indices for DPT head

**Gaussian Head Prediction (Lines 225-233):**
```python
# UNFrozen Gaussian parameter prediction
with torch.cuda.amp.autocast(dtype=dtype):
    gaussian_params_preds, _ = self.gaussian_dpt(
        aggregated_tokens_list,
        views,
        patch_start_idx
    )
```
**Significance:**
- **NOT wrapped in `torch.no_grad()`**: Gradients must flow through Gaussian head during training
- `autocast`: Still uses mixed precision for memory efficiency
- DPT head outputs raw logits for Gaussian parameters

**Shape Handling Logic (Lines 235-336):**

This is the most complex part of `forward()`. It handles shape mismatches between VGGT outputs and DPT head outputs.

**Problem:** DPT head may output `[B, S, C, H, W]` or `[B, S, H, W, C]`, and spatial dimensions may not match `world_points` dimensions.

**Solution (Lines 246-305):**
```python
if len(gaussian_params_preds.shape) == 5:
    B, S, dim1, dim2, dim3 = gaussian_params_preds.shape
    # Determine which dimension is channels vs spatial
    if dim1 == gaussian_num_channels:
        # Shape is [B, S, C, H, W]
        C, H, W = dim1, dim2, dim3
    elif dim3 == gaussian_num_channels:
        # Shape is [B, S, H, W, C]
        H, W, C = dim1, dim2, dim3
```
**Significance:** Dynamically infers tensor layout by comparing dimensions to expected channel count. This handles:
- Different DPT head implementations
- Non-square inputs (e.g., 518x294 from DROID dataset)
- Edge cases where channel count exceeds expected (slices to first `gaussian_num_channels`)

**Spatial Dimension Correction (Lines 307-335):**
```python
if H != H_vggt or W != W_vggt:
    # Interpolate to match world_points dimensions
    gaussian_params_flat = F.interpolate(
        gaussian_params_flat,
        size=(H_vggt, W_vggt),
        mode='bilinear',
        align_corners=False
    )
```
**Significance:** Ensures Gaussian parameters align spatially with `world_points`. This is critical because:
- `means = world_points + offset` (line 363) requires matching spatial dimensions
- Rendering expects aligned `means` and `covariances`
- Interpolation preserves spatial relationships

**Parameter Splitting (Lines 337-343):**
```python
splits = [3, 3, 4, 3 * self.sh_degree, 1]
offset, scales, rotations, sh, opacities = torch.split(fmap, splits, dim=-1)
```
**Significance:** Splits concatenated channels into individual parameter tensors. Order matches initialization order.

**Post-Processing (Lines 345-350):**
```python
offset = reg_dense_offsets(offset)
scales = reg_dense_scales(scales)
rotations = reg_dense_rotation(rotations)
sh = reg_dense_sh(sh)
opacities = reg_dense_opacities(opacities)
```
**Significance:**
- `reg_dense_offsets`: Clamps to reasonable range (e.g., ±0.1m)
- `reg_dense_scales`: Applies `exp()` to convert log-space to linear (ensures positive scales)
- `reg_dense_rotation`: Normalizes quaternions to unit length
- `reg_dense_sh`: Clamps SH coefficients
- `reg_dense_opacities`: Applies sigmoid to get [0, 1] opacity

**Means Computation (Lines 361-365):**
```python
if self.use_offsets:
    means = world_points + offset
else:
    means = world_points
```
**Significance:** Gaussian centers are either:
- VGGT's world points (if `use_offsets=False`)
- VGGT's world points + learned offset (if `use_offsets=True`)

Offsets allow fine-tuning 3D positions beyond VGGT's predictions.

**SH Seeding (Lines 367-371):**
```python
base_sh = sh_utils.RGB2SH(einops.rearrange(views, "b s c h w -> b s h w c"))
sh = sh.clone()  # Avoid in-place modification error
sh[..., 0] = sh[..., 0] + base_sh
```
**Significance:**
- Seeds DC (degree-0) SH term with input image colors
- Provides meaningful initial color (prevents gray monotone)
- `clone()`: Required because `sh` is a view from `torch.split()`, and in-place modification during backprop causes errors

**Covariance Construction (Line 374):**
```python
covariances = utils.geometry.build_covariance(scales, rotations)
```
**Significance:** Builds 3x3 covariance matrices from scales (diagonal) and rotations (rotation matrix). Required for CUDA rasterizer.

**Camera Recovery (Lines 376-390):**
```python
camera_extrinsics, camera_intrinsics = pose_encoding_to_extri_intri(
    pose_enc, image_size_hw=(H_img, W_img)
)
```
**Significance:** Converts VGGT's pose encoding to standard camera matrices. Extrinsics are padded to 4x4 homogeneous form for rendering.

**Return Values (Lines 392-406):**
```python
return frozen_vggt_preds, gaussian_head_preds
```
**Significance:**
- `frozen_vggt_preds`: VGGT outputs (for visualization/debugging)
- `gaussian_head_preds`: Dictionary with all Gaussian parameters (for rendering/training)

#### `_render_from_gaussians(self, gaussian_preds, image_shape_hw)`

**Lines 411-463:** Renders images from Gaussian parameters

**Input Extraction (Lines 424-429):**
```python
means = gaussian_preds['means']               # [B, S, H, W, 3]
covariances = gaussian_preds['covariances']   # [B, S, H, W, 3, 3]
sh = gaussian_preds['sh']                     # [B, S, H, W, 3, sh_degree]
opacities = gaussian_preds['opacities']       # [B, S, H, W, 1]
extrinsics = gaussian_preds['camera_extrinsics']  # [B, S, 4, 4]
intrinsics = gaussian_preds['camera_intrinsics']  # [B, S, 3, 3]
```
**Significance:** Extracts all required parameters for CUDA rasterization.

**Float32 Conversion (Lines 438-444):**
```python
means_f32 = means.float()
covariances_f32 = covariances.float()
# ... convert all to float32
```
**Significance:** CUDA rasterizer (`render_cuda`) expects `float32`, but model uses `bfloat16` for memory efficiency. Conversion is necessary.

**Rendering (Lines 446-457):**
```python
color = pixelsplat_decoder.render_cuda(
    rearrange(extrinsics_f32, "b v i j -> (b v) i j"),
    # ... rearrange all inputs to (B*V, ...) format
)
```
**Significance:**
- `render_cuda` expects flattened batch-view dimension `(B*V, ...)`
- `einops.rearrange`: Efficient tensor reshaping
- Rasterizes Gaussians into 2D images using differentiable CUDA kernel

**Memory Cleanup (Lines 460-461):**
```python
del means_f32, covariances_f32, ...
```
**Significance:** Explicitly frees intermediate tensors to prevent OOM during training.

#### `_compute_losses(self, batch, rendered_color, views_tensor, ...)`

**Lines 465-558:** Computes training losses

**Target Selection (Lines 485-498):**
```python
if 'context' in batch and len(batch['context']) >= S_rendered:
    target_color = torch.stack([ctx['original_img'] for ctx in batch['context'][:S_rendered]], dim=1)
elif 'target' in batch:
    target_color = torch.stack([target_view['original_img'] for target_view in batch['target'][:S_rendered]], dim=1)
else:
    target_color = views_tensor[:, :S_rendered]
```
**Significance:** Selects target images for loss computation. Priority:
1. Context views (same views used for prediction)
2. Target views (novel views, if available)
3. Input views (fallback)

**Mask Application (Lines 508-519):**
```python
if apply_mask and mask is not None:
    B_mask, V_mask, H_mask, W_mask = mask.shape
    if V_mask == S_rendered and H_mask == H_rendered and W_mask == W_rendered:
        target_color = target_color * mask[..., None, :, :]
        predicted_color = predicted_color * mask[..., None, :, :]
    else:
        apply_mask = False  # Shape mismatch, disable masking
```
**Significance:** Applies loss mask (from `loss_mask.calculate_loss_mask`) to focus loss on valid regions (e.g., non-occluded, non-background). If mask shape doesn't match (e.g., mask for target views but training on context views), masking is disabled.

**MSE Loss (Lines 527-532):**
```python
rgb_l2_loss = (predicted_color - target_color) ** 2
if average_over_mask and mask is not None:
    mse_loss = (rgb_l2_loss * mask[:, None, ...]).sum() / mask.sum()
else:
    mse_loss = rgb_l2_loss.mean()
```
**Significance:** Pixel-wise L2 loss. If `average_over_mask=True`, averages only over masked pixels (prevents background from dominating loss).

**LPIPS Loss (Lines 534-539):**
```python
lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
```
**Significance:** Perceptual loss using VGG features. Captures high-level structure (edges, textures) better than MSE.

**Total Loss (Lines 541-546):**
```python
loss = mse_weight * mse_loss + lpips_weight * lpips_loss
```
**Significance:** Weighted combination. Default: `mse_weight=1.0`, `lpips_weight=0.25` (LPIPS is typically smaller magnitude).

**SSIM (Lines 548-556):**
```python
if calculate_ssim:
    ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=False)
```
**Significance:** Structural Similarity Index (metric, not used in loss). Higher = better quality.

#### `training_step(self, batch, batch_idx)`

**Lines 577-631:** Training step

**Image Normalization (Lines 586-592):**
```python
views_normalized = torch.stack([ctx['img'] for ctx in batch['context'][:self.config.num_views]], dim=1)
views = (views_normalized + 1.0) / 2.0  # Convert from [-1, 1] to [0, 1]
```
**Significance:** ScanNet++ dataset uses `ImgNorm` (normalizes to `[-1, 1]`), but VGGT expects `[0, 1]`. Conversion is critical.

**Forward Pass (Lines 595-598):**
```python
vggt_preds, gaussian_preds = self(views)
del vggt_preds  # Free memory
```
**Significance:** 
- Calls `forward()` which handles frozen VGGT + trainable Gaussian head
- Deletes `vggt_preds` immediately (not needed for training, saves memory)

**Rendering (Lines 600-603):**
```python
rendered_color = self._render_from_gaussians(gaussian_preds, image_shape_hw=(H, W))
torch.cuda.empty_cache()
```
**Significance:** Renders predicted Gaussians into images. Cache clearing prevents OOM.

**Loss Computation (Lines 605-624):**
```python
mask = loss_mask.calculate_loss_mask(batch) if apply_mask_config else None
loss, mse, lpips_val = self._compute_losses(batch, rendered_color, views, mask=mask, ...)
```
**Significance:** Computes loss with optional masking. Mask focuses loss on valid regions.

**Memory Management (Lines 626-628):**
```python
if batch_idx % 10 == 0:
    torch.cuda.empty_cache()
```
**Significance:** Periodic cache clearing every 10 batches prevents gradual memory accumulation.

**Logging (Line 630):**
```python
self.log_metrics('train', loss, mse, lpips_val)
```
**Significance:** Logs metrics to Lightning loggers (CSV, WandB).

#### `validation_step` and `test_step`

**Lines 633-702:** Similar to `training_step`, but:
- `validation_step`: Logs with `on_step=False` (only at end of epoch)
- `test_step`: Includes SSIM computation and benchmarker timing

#### `configure_optimizers(self)`

**Lines 704-731:** Optimizer configuration

**VGGT Freezing (Line 709):**
```python
self.vggt.requires_grad_(False)
```
**Significance:** Ensures VGGT stays frozen (defensive check, already frozen in `__init__`).

**Optimizer (Lines 712-714):**
```python
params = [p for p in self.gaussian_dpt.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=opt_lr)
```
**Significance:** Only Gaussian head parameters are optimized. Adam optimizer with default `β1=0.9`, `β2=0.999`.

**LR Scheduler (Lines 717-722):**
```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[opt_epochs // 2],
    gamma=0.1
)
```
**Significance:** Reduces learning rate by 10x at half-way point. Helps fine-tuning in later epochs.

#### `Config` Dataclass

**Lines 733-778:** Configuration dataclass

**Fields:**
- `image_size`: Input image size (518 for VGGT)
- `batch_size`: Batch size (typically 1 due to memory)
- `num_views`: Number of input views (2)
- `opt`: Optimization parameters (lr, epochs, gradient_clip_val)
- `loss`: Loss parameters (weights, masking flags)
- `sh_degree`: Spherical harmonics degree (0 or 1)
- `render_res`: Rendering resolution
- `use_offsets`: Whether to use learned offsets

**`__post_init__` (Lines 762-778):**
```python
if self.opt is None:
    self.opt = {'lr': 0.0001, 'epochs': 20, 'gradient_clip_val': 0.5}
```
**Significance:** Sets defaults if nested dicts are not provided. Ensures backward compatibility.

---

### train_cursor.py

**Purpose:** Training orchestration script (mirrors Splatt3R's `run_experiment`)

#### `run_experiment_vggt_gaussians(config)`

**Lines 35-241:** Main training function

**Seed Setting (Lines 42-44):**
```python
seed = getattr(config, 'seed', 0)
L.seed_everything(seed, workers=True)
```
**Significance:** Ensures reproducibility. `workers=True` sets seed for data loader workers.

**Logger Setup (Lines 46-75):**
```python
csv_logger = L.pytorch.loggers.CSVLogger(save_dir=save_dir, name=name)
loggers.append(csv_logger)
```
**Significance:** CSV logger saves metrics to `{save_dir}/{name}/metrics.csv`. WandB is disabled by default to avoid interactive prompts.

**WandB Disabling (Lines 69-75):**
```python
os.environ['WANDB_MODE'] = 'disabled'
```
**Significance:** Prevents WandB from prompting for account setup during training.

**Model Instantiation (Lines 97-109):**
```python
model = VGGTGaussians(config)
if hasattr(config, 'use_pretrained') and config.use_pretrained:
    # Load pretrained checkpoint
```
**Significance:** Creates model with config. Optional pretrained checkpoint loading for fine-tuning.

**Dataset Loading (Lines 111-186):**
```python
train_dataset = scannetpp.get_scannet_dataset(
    data_root,
    'train',
    resolution,
    num_epochs_per_epoch=getattr(config, 'data', {}).get('epochs_per_train_epoch', 1),
)
```
**Significance:**
- Loads ScanNet++ training dataset
- `resolution`: Tuple `(H, W)` (518, 518) for VGGT
- `num_epochs_per_train_epoch`: How many times to sample from each scene per epoch (100 = diverse sampling)

**Validation Dataset (Lines 155-176):**
```python
try:
    val_dataset = scannetpp.get_scannet_test_dataset(...)
except Exception as e:
    print(f"Warning: Could not load validation dataset: {e}")
    data_loader_val = None
```
**Significance:** Validation is optional. If test set JSON is missing, falls back to training dataset structure.

**DataLoader Creation (Lines 147-153):**
```python
data_loader_train = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
)
```
**Significance:**
- `shuffle=True`: Randomizes order each epoch
- `num_workers=4`: Parallel data loading
- `pin_memory=True`: Faster GPU transfer

**Checkpoint Callback (Lines 207-216):**
```python
checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    dirpath=os.path.join(save_dir, name, "checkpoints"),
    filename="vggt_gaussians-{epoch:02d}-{train_loss:.2f}",
    monitor="train/loss",
    save_top_k=3,
    save_last=True,
)
```
**Significance:**
- Saves top 3 checkpoints by training loss
- Also saves last checkpoint (`save_last=True`)
- Filename includes epoch and loss for easy identification

**Trainer Setup (Lines 218-231):**
```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=devices,
    max_epochs=config.opt.get('epochs', 20),
    gradient_clip_val=config.opt.get('gradient_clip_val', 0.5),
    check_val_every_n_epoch=1 if data_loader_val is not None else None,
)
```
**Significance:**
- `accelerator="gpu"`: Uses GPU
- `gradient_clip_val=0.5`: Clips gradients to prevent explosion
- `check_val_every_n_epoch=None`: Skips validation if no validation dataset

**Training (Line 235):**
```python
trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)
```
**Significance:** Starts training loop. Lightning handles:
- Epoch iteration
- Batch iteration
- Loss computation
- Optimizer steps
- LR scheduling
- Checkpoint saving
- Metric logging

---

### inference_cursor.py

**Purpose:** Load trained checkpoint and run inference on image pairs

#### `main()`

**Lines 39-380:** Main inference function

**Checkpoint Loading (Lines 47-83):**
```python
checkpoint = torch.load(args.checkpoint, map_location=args.device)
if 'hyper_parameters' in checkpoint:
    hparams = checkpoint['hyper_parameters']
    config = Config(...)  # Reconstruct from hyperparameters
model = VGGTGaussians.load_from_checkpoint(args.checkpoint, config=config, ...)
```
**Significance:**
- Loads checkpoint file
- Extracts hyperparameters to reconstruct `Config`
- Uses Lightning's `load_from_checkpoint` to restore model state

**Verification Logic (Lines 88-272):**

This is a comprehensive set of checks to ensure trained weights were actually loaded (not random initialization).

**Check 1: Gaussian Head Keys (Lines 93-96):**
```python
gaussian_keys_in_ckpt = [k for k in checkpoint_state.keys() if 'gaussian_dpt' in k]
if len(gaussian_keys_in_ckpt) == 0:
    raise RuntimeError("No Gaussian head weights found!")
```
**Significance:** Ensures checkpoint contains Gaussian head weights.

**Check 2: Weight Matching (Lines 115-140):**
```python
for key in gaussian_keys_in_ckpt:
    if torch.allclose(ckpt_val, model_val, atol=1e-4):
        matched_keys += 1
    else:
        mismatched_keys += 1
```
**Significance:** Verifies loaded model weights match checkpoint weights (confirms loading worked).

**Check 3: Trained vs. Untrained (Lines 142-174):**
```python
fresh_model = VGGTGaussians(config)  # Untrained
if torch.allclose(trained_val, untrained_val, atol=1e-4):
    raise RuntimeError("Trained weights match untrained initialization!")
```
**Significance:** **Critical check**: If trained weights match untrained initialization, checkpoint wasn't loaded properly. Raises error to prevent false confidence.

**Check 4: Training Metadata (Lines 176-190):**
```python
if 'epoch' in checkpoint:
    epoch = checkpoint['epoch']
    if epoch == 0:
        print("WARNING: Checkpoint is from epoch 0")
```
**Significance:** Checks if training actually occurred (epoch > 0, global_step > 0).

**Check 5: Loss from Filename (Lines 198-206):**
```python
loss_match = re.search(r'train_loss=([\d.]+)', checkpoint_name)
if loss_match:
    train_loss = float(loss_match.group(1))
    if train_loss == 0.0:
        print("WARNING: Training loss is 0.0 - suspicious!")
```
**Significance:** Extracts training loss from checkpoint filename. Warns if loss is 0.0 (suspicious).

**Check 6: Optimizer State (Lines 208-212):**
```python
if 'optimizer_states' in checkpoint or 'lr_schedulers' in checkpoint:
    print("✓ Optimizer/scheduler state found")
```
**Significance:** Confirms training occurred (optimizer state is only saved during training).

**Check 7: Initialization Pattern (Lines 214-238):**
```python
scales_bias = final_conv_bias[3:6].mean().item()
opacity_bias = final_conv_bias[-1].item()
if abs(scales_bias - (-7.0)) < 0.01 and abs(opacity_bias - (-2.0)) < 0.01:
    print("WARNING: Biases match initialization pattern exactly!")
```
**Significance:** Checks if biases match initialization values. If they do, weights likely weren't trained.

**Final Verification Summary (Lines 244-271):**
```python
if verification_passed:
    print("✓ VERIFICATION PASSED: Trained weights confirmed loaded")
else:
    raise RuntimeError("Verification failed!")
```
**Significance:** Provides clear pass/fail status. Raises error if verification fails.

**Episode Processing (Lines 274-371):**
```python
for episode_dir in episodes_list:
    episode_idx = int(episode_dir.split('_')[-1])
    # Load image pairs
    images = load_and_preprocess_images(image_paths, mode="crop")
    # Run inference
    with torch.no_grad():
        vggt_preds, gaussian_preds = model(images)
    # Save to .npz
    np.savez(output_path, vggt_preds=..., gaussian_preds=...)
```
**Significance:**
- Iterates through episode directories
- Loads and preprocesses image pairs (resizes to 518px width, maintains aspect ratio)
- Runs inference (no gradients)
- Saves predictions to `.npz` files for later visualization

---

### check_outputs_vggt_gaussians.py

**Purpose:** Load `.npz` outputs and create visualizations

#### Main Script

**Lines 10-144:** Visualization orchestration

**Episode Parsing (Lines 27-36):**
```python
if args.do_episodes:
    if '-' in args.do_episodes:
        start, end = map(int, args.do_episodes.split('-'))
        episodes_to_process = set(range(start, end + 1))
```
**Significance:** Supports single episode (`0`), range (`1-5`), or comma-separated (`1,3,5`).

**Step Selection (Lines 88-93):**
```python
if episode_steps_to_process is None:
    num_steps = len(vggt_preds)
    episode_steps_to_process = sorted(set(np.linspace(0, num_steps - 1, 3, dtype=int).tolist()))
```
**Significance:** If no steps specified, samples 3 evenly-spaced steps from episode (reduces visualization time).

**Visualization Call (Lines 136-143):**
```python
create_visualizations_vggt_gaussians(
    vggt_preds,
    gaussian_preds,
    episode_idx,
    vggt_gaussians_visuals_dir,
    steps_to_process=episode_steps_to_process,
    render_gaussians=args.render_gaussians
)
```
**Significance:** Calls visualization utility (in `visualization_utils.py`) to create:
- VGGT visualizations (depth, world points, images)
- Gaussian visualizations (means, scales, rotations, SH, opacities, offsets)
- Rendered Gaussian images (if `--render_gaussians`)
- PLY files (for Gaussian Splatting viewers)

---

### vggt_gaussians.yaml

**Purpose:** Configuration file for training

**Key Parameters:**

```yaml
data:
  root: '/DATA/scannetpp_sample'
  batch_size: 1
  resolution: 518
  epochs_per_train_epoch: 100

opt:
  epochs: 20
  lr: 0.0001
  gradient_clip_val: 0.5

loss:
  mse_loss_weight: 1.0
  lpips_loss_weight: 0.25
  apply_mask: True
  average_over_mask: True
```

**Significance:**
- `resolution: 518`: VGGT-compatible (divisible by 14)
- `batch_size: 1`: Conservative (can increase if GPU memory allows)
- `epochs_per_train_epoch: 100`: Samples 100 view pairs per scene per epoch (diverse training)
- `lr: 0.0001`: Learning rate for Gaussian head (VGGT frozen)
- `mse_loss_weight: 1.0, lpips_loss_weight: 0.25`: Loss weighting (LPIPS typically smaller magnitude)

---

## Key Design Decisions

### 1. Frozen VGGT Backbone

**Decision:** Keep VGGT completely frozen (`requires_grad_(False)`, `eval()`)

**Rationale:**
- VGGT is 1B parameters → training would require massive GPU memory
- VGGT is pretrained on large-scale data → provides strong geometric priors
- Only Gaussian head needs training → efficient fine-tuning

**Implementation:**
- `torch.no_grad()` around VGGT calls in `forward()`
- `self.vggt.requires_grad_(False)` in `__init__` and `configure_optimizers()`
- Only `self.gaussian_dpt.parameters()` in optimizer

### 2. Shape Handling in Forward

**Decision:** Complex shape inference and interpolation logic

**Rationale:**
- DPT head may output different tensor layouts
- Non-square inputs (e.g., 518x294) require careful handling
- Spatial alignment with `world_points` is critical

**Implementation:**
- Dynamic shape inference (lines 246-305)
- Interpolation to match `world_points` dimensions (lines 307-335)
- Validation checks to prevent dimension confusion

### 3. Image Normalization

**Decision:** Convert from `[-1, 1]` (ScanNet++) to `[0, 1]` (VGGT)

**Rationale:**
- ScanNet++ uses `ImgNorm` (normalizes to `[-1, 1]`)
- VGGT expects `[0, 1]` range
- Conversion is simple: `(x + 1) / 2`

**Implementation:**
- In `training_step`, `validation_step`, `test_step`: `views = (views_normalized + 1.0) / 2.0`

### 4. Mixed Precision Training

**Decision:** Use `bfloat16` for VGGT, `float32` for rendering

**Rationale:**
- VGGT is 1B parameters → `bfloat16` saves ~50% memory
- CUDA rasterizer requires `float32` → conversion before rendering
- Minimal accuracy loss with `bfloat16`

**Implementation:**
- `autocast(dtype=bfloat16)` around VGGT and Gaussian head
- Explicit `float()` conversion before `render_cuda` call

### 5. Memory Management

**Decision:** Aggressive memory cleanup

**Rationale:**
- Training on single GPU (RTX 3090) with 1B-parameter backbone
- Rendering is memory-intensive
- Prevents OOM errors

**Implementation:**
- `del vggt_preds` after use
- `del means_f32, ...` after rendering
- `torch.cuda.empty_cache()` every 10 batches
- Periodic cache clearing in training loop

### 6. Loss Masking

**Decision:** Optional loss masking with fallback

**Rationale:**
- Mask focuses loss on valid regions (non-occluded, non-background)
- Mask shape may not match rendered views (e.g., mask for target views, training on context views)
- Fallback to no masking if shape mismatch

**Implementation:**
- `loss_mask.calculate_loss_mask(batch)` if enabled
- Shape validation before applying mask
- Disable masking if shapes don't match

### 7. Verification in Inference

**Decision:** Comprehensive weight verification

**Rationale:**
- User concern: "results too good" → might be using random weights
- Need to ensure checkpoint was loaded correctly
- Prevent false confidence in untrained model

**Implementation:**
- 7 verification checks (see `inference_cursor.py` lines 88-272)
- Compares trained vs. untrained weights
- Raises error if weights match initialization

---

## Data Flow

### Training Flow

```
1. DataLoader
   └─> batch['context']: List of dicts with 'img' ([-1, 1]) and 'original_img' ([0, 1])

2. Image Normalization
   └─> views = (batch['context'][i]['img'] + 1.0) / 2.0  # Convert to [0, 1]

3. Forward Pass
   ├─> VGGT (frozen, no_grad)
   │   ├─> self.vggt(views) → frozen_vggt_preds
   │   │   ├─> world_points: [B, S, H, W, 3]
   │   │   ├─> depth: [B, S, H, W]
   │   │   ├─> pose_enc: [B, S, ...]
   │   │   └─> images: [B, S, C, H, W] (stored in eval mode)
   │   └─> self.vggt.aggregator(views) → aggregated_tokens_list, patch_start_idx
   │       └─> aggregated_tokens_list: [B, S, embed_dim]
   │
   └─> Gaussian Head (trainable, gradients enabled)
       └─> self.gaussian_dpt(aggregated_tokens_list, views, patch_start_idx)
           └─> gaussian_params_preds: [B, S, C, H, W] or [B, S, H, W, C]
               ├─> Shape inference and interpolation
               ├─> Split into: offset, scales, rotations, sh, opacities
               ├─> Post-process: reg_dense_* functions
               ├─> means = world_points + offset
               ├─> SH seeding: sh[..., 0] += RGB2SH(views)
               ├─> covariances = build_covariance(scales, rotations)
               └─> camera_extrinsics, camera_intrinsics from pose_enc

4. Rendering
   └─> _render_from_gaussians(gaussian_preds, image_shape_hw)
       ├─> Extract: means, covariances, sh, opacities, extrinsics, intrinsics
       ├─> Convert to float32
       ├─> Rearrange to (B*V, ...) format
       └─> render_cuda(...) → rendered_color: [B, S, C, H, W]

5. Loss Computation
   └─> _compute_losses(batch, rendered_color, views, mask)
       ├─> Select target_color from batch['context'] or batch['target']
       ├─> Apply mask (if enabled and shapes match)
       ├─> MSE loss: (rendered_color - target_color) ** 2
       ├─> LPIPS loss: lpips_criterion(target_color, rendered_color)
       └─> Total loss: mse_weight * mse_loss + lpips_weight * lpips_loss

6. Backward Pass
   └─> loss.backward()  # Gradients flow only through Gaussian head
       └─> optimizer.step()  # Update Gaussian head parameters only
```

### Inference Flow

```
1. Checkpoint Loading
   └─> torch.load(checkpoint) → extract hyper_parameters
       └─> Reconstruct Config from hyper_parameters
           └─> VGGTGaussians.load_from_checkpoint(checkpoint, config=config)

2. Verification
   └─> 7 verification checks (see inference_cursor.py)
       └─> Ensure trained weights were loaded

3. Episode Processing
   └─> For each episode:
       ├─> Load image pairs: load_and_preprocess_images(image_paths, mode="crop")
       │   └─> Resize to 518px width, maintain aspect ratio, ensure divisible by 14
       ├─> Run inference: model(images) with torch.no_grad()
       │   └─> Returns: vggt_preds, gaussian_preds
       └─> Save to .npz: np.savez(output_path, vggt_preds=..., gaussian_preds=...)

4. Visualization (check_outputs_vggt_gaussians.py)
   └─> Load .npz files
       └─> create_visualizations_vggt_gaussians(...)
           ├─> VGGT visualizations (depth, world points, images)
           ├─> Gaussian visualizations (means, scales, rotations, SH, opacities, offsets)
           ├─> Rendered Gaussian images (if --render_gaussians)
           └─> PLY files (for Gaussian Splatting viewers)
```

---

## Training Pipeline

### Step-by-Step Training Process

1. **Initialization**
   - Load VGGT from HuggingFace (`facebook/VGGT-1B`)
   - Freeze VGGT (`requires_grad_(False)`, `eval()`)
   - Initialize Gaussian DPT head with Xavier initialization
   - Set up optimizer (Adam) for Gaussian head only
   - Set up LR scheduler (MultiStepLR, reduce at epoch 10)

2. **Data Loading**
   - Load ScanNet++ dataset
   - Sample view pairs from scenes
   - Apply `ImgNorm` (normalizes to `[-1, 1]`)
   - Batch into DataLoader

3. **Training Loop** (per batch)
   - Normalize images: `(img + 1) / 2` → `[0, 1]`
   - Forward pass:
     - VGGT (frozen, no_grad) → `world_points`, `aggregated_tokens`
     - Gaussian head (trainable) → Gaussian parameters
     - Post-process parameters
     - Render Gaussians → `rendered_color`
   - Compute loss:
     - MSE: `(rendered_color - target_color) ** 2`
     - LPIPS: `lpips_criterion(target_color, rendered_color)`
     - Apply mask (if enabled)
     - Total: `mse_weight * mse + lpips_weight * lpips`
   - Backward pass:
     - `loss.backward()` (gradients only through Gaussian head)
     - `optimizer.step()` (update Gaussian head only)
   - Log metrics (MSE, LPIPS, PSNR)

4. **Validation** (end of epoch)
   - Similar to training, but:
     - No gradient computation
     - Log with `on_step=False` (only at end of epoch)
     - No parameter updates

5. **Checkpoint Saving**
   - Save top 3 checkpoints by training loss
   - Save last checkpoint
   - Filename: `vggt_gaussians-{epoch:02d}-{train_loss:.2f}.ckpt`

---

## Inference Pipeline

### Step-by-Step Inference Process

1. **Checkpoint Loading**
   - Load checkpoint file
   - Extract `hyper_parameters` from checkpoint
   - Reconstruct `Config` from hyperparameters
   - Load model: `VGGTGaussians.load_from_checkpoint(checkpoint, config=config)`

2. **Verification**
   - Check Gaussian head keys exist in checkpoint
   - Verify loaded weights match checkpoint weights
   - Compare trained weights vs. untrained initialization (raise error if match)
   - Check training metadata (epoch, global_step, optimizer state)
   - Check initialization pattern (biases)

3. **Episode Processing**
   - For each episode directory:
     - Find image pairs (`exterior_image_1_left`, `exterior_image_2_left`)
     - For each image pair:
       - Load and preprocess: `load_and_preprocess_images(image_paths, mode="crop")`
         - Resize to 518px width
         - Maintain aspect ratio
         - Ensure divisible by 14
       - Run inference: `model(images)` with `torch.no_grad()`
       - Convert to NumPy: `vggt_preds_numpy`, `gaussian_preds_numpy`
       - Append to episode lists
     - Save episode: `np.savez(output_path, vggt_preds=..., gaussian_preds=...)`

4. **Visualization** (separate script)
   - Load `.npz` files
   - For each episode and step:
     - Visualize VGGT outputs (depth, world points, images)
     - Visualize Gaussian outputs (means, scales, rotations, SH, opacities, offsets)
     - Render Gaussians to 2D images (if `--render_gaussians`)
     - Save PLY file (for Gaussian Splatting viewers)

---

## Visualization Pipeline

### Visualization Components

1. **VGGT Visualizations** (`_visualize_vggt_single_step`)
   - Depth maps: Colormap visualization
   - World points: 3D point cloud visualization
   - Input images: Original images from VGGT predictions

2. **Gaussian Visualizations** (`_visualize_vggt_gaussians_single_step`)
   - Means: 3D positions (colored by depth)
   - Scales: Log-space scales (colored by magnitude)
   - Rotations: Quaternion visualization
   - SH: Spherical harmonics DC component (RGB visualization)
   - Opacities: Opacity maps
   - Offsets: Learned offsets (diverging colormap, centered at 0)

3. **Rendered Images** (`_render_vggt_gaussians_single_step`)
   - Renders predicted Gaussians into 2D images using CUDA rasterizer
   - Shows what the model "sees" after training

4. **PLY Files** (`save_vggt_gaussians_ply_from_npz`)
   - Exports Gaussian parameters to PLY format
   - Compatible with Gaussian Splatting viewers (e.g., mip-splatting demo)
   - Fields: `x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3`

---

## Conclusion

VGGTGaussians is a carefully designed system that:

1. **Leverages pretrained VGGT** for rich geometric understanding
2. **Trains only a small Gaussian head** for efficient fine-tuning
3. **Handles complex shape mismatches** between VGGT and DPT head outputs
4. **Manages memory efficiently** through mixed precision and aggressive cleanup
5. **Provides comprehensive verification** to ensure trained weights are used
6. **Integrates seamlessly** with Splatt3R's visualization and evaluation pipelines

The codebase demonstrates attention to detail in:
- Config compatibility (OmegaConf, dataclass, dict)
- Shape handling (dynamic inference, interpolation)
- Memory management (explicit cleanup, cache clearing)
- Verification (comprehensive weight checks)
- Visualization (comprehensive output generation)

This makes VGGTGaussians a robust, production-ready implementation of Gaussian Splatting with a frozen VGGT backbone.

---

**End of Report**

