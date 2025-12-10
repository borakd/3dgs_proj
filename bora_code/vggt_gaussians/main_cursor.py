"""
Re-implementing Splatt3R but with VGGT backbone. Freeze the VGGT, train the Gaussian head,
then freeze everything for inference.
"""

import os
import sys
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import einops
import lpips
import numpy as np
from dataclasses import dataclass
import argparse

# Get absolute path to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Now add other paths
sys.path.append(os.path.join(project_root, 'src/mast3r_src'))  # Fixed: should be mast3r_src, not mast3r_src/mast3r
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src/vggt'))

import utils.geometry
import utils.sh_utils as sh_utils
import utils.loss_mask as loss_mask
import utils.compute_ssim as compute_ssim

# Import Gaussian head from modified MASt3R implementation
from mast3r.catmlp_dpt_head import (
    reg_dense_offsets,
    reg_dense_scales,
    reg_dense_rotation,
    reg_dense_sh,
    reg_dense_opacities,
    gaussian_postprocess,
    GaussianHead
    )

# Import Pixelsplat
sys.path.append(os.path.join(project_root, 'src/pixelsplat_src'))
import pixelsplat_src.benchmarker as benchmarker
import pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder

# Import VGGT
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.heads.dpt_head import DPTHead
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class VGGTGaussians(L.LightningModule):

    def __init__(self, config):

        super().__init__()
        
        # Normalize config: handle OmegaConf or dataclass
        import omegaconf
        if isinstance(config, omegaconf.DictConfig):
            # Convert OmegaConf to Config dataclass
            config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
            # Extract nested structures
            opt_dict = config_dict.get('opt', {})
            loss_dict = config_dict.get('loss', {})
            # Handle resolution: can be int or list/tuple
            resolution_raw = config_dict.get('data', {}).get('resolution', 518)
            if isinstance(resolution_raw, (int, float)):
                # Single value: use for both dimensions
                render_res = (int(resolution_raw), int(resolution_raw))
            elif isinstance(resolution_raw, (list, tuple)):
                # List/tuple: convert to tuple
                render_res = tuple(resolution_raw)
            else:
                # Default fallback
                render_res = (518, 518)
            
            # Create normalized config
            normalized_config = Config(
                image_size=config_dict.get('image_size', 518),
                batch_size=config_dict.get('data', {}).get('batch_size', 1),
                num_views=2,  # Default
                opt=opt_dict if opt_dict else None,
                loss=loss_dict if loss_dict else None,
                sh_degree=config_dict.get('sh_degree', 1),
                render_res=render_res,
                use_offsets=config_dict.get('use_offsets', True),
            )
            self.config = normalized_config
        elif isinstance(config, Config):
            # Already a Config dataclass
            self.config = config
        else:
            # Try to create Config from dict or other object
            if isinstance(config, dict):
                self.config = Config(**config)
            else:
                # Assume it's a Config-like object, convert to dict
                config_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith('_')}
                self.config = Config(**config_dict)
        
        # Load pretrained VGGT and freeze it completely
        # VGGT does not have separate encoder/decoder architecture, rather it is one deep transformer
        print("Loading VGGT base model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device: {device}")
        self.vggt = VGGT.from_pretrained("facebook/VGGT-1B")    # removed ".to(device)", let Lightning handle it automatically for DDP
        self.vggt.requires_grad_(False)
        # TODO: eval mode needed?
        self.vggt.eval()

        # Debug print
        # print(f"Aggregator instance attributes: {vars(self.vggt.aggregator)}")
        print(f"input token shape: {self.vggt.aggregator.camera_token.shape}")
        embed_dim = self.vggt.aggregator.camera_token.shape[-1]
        print(f"embed dim (last dim of input): {embed_dim}")

        # Define the Gaussian head
        # Computing the output dim:
        # 3D mean offsets (3) +
        # Scales (3) +
        # Rotations (4) +
        # Spherical Harmonics (3 * sh_degree) +
        # Opacity (1)
        sh_degree = self.config.sh_degree
        gaussian_num_channels = 3 + 3 + 4 + 3 * sh_degree + 1
        self.gaussian_num_channels = gaussian_num_channels  # Store for use in forward
        print(f"sh degree: {sh_degree}")
        print(f"gaussian num channels: {gaussian_num_channels}")
        #       degree 0 = 1 coeeficient per color (3 total)
        #       degree 1 = 4 coefficients per color (12 total)
        #       with degree 1 SH output dim becomes 23
        self.gaussian_dpt = DPTHead(
            dim_in=2 * embed_dim,
            output_dim = gaussian_num_channels + 1, # channels = 14, output dim = 15
            activation='linear',
            conf_activation='expp1',
            feature_only=False
        )   # .to(device)

        final_conv_layer = self.gaussian_dpt.scratch.output_conv2[-1]
        # print(f"final_conv_layer: {final_conv_layer}")
        splits_and_inits = [
            (3, 0.001, 0.001),  # 3D mean offsets
            (3, 0.00003, -7.0),  # Scales
            (4, 1.0, 0.0),  # Rotations
            (3 * sh_degree, 1.0, 0.0),  # Spherical Harmonics
            (1, 1.0, -2.0)  # Opacity
        ]
        start_channels = 0
        for out_channel, s, b in splits_and_inits:
            torch.nn.init.xavier_uniform_(
                final_conv_layer.weight[start_channels:start_channels+out_channel, :, :, :],
                s
            )
            torch.nn.init.constant_(
                final_conv_layer.bias[start_channels:start_channels+out_channel],
                b
            )
            start_channels += out_channel
        
        # The decoder which we use to render the predicted Gaussians into
        # images, lightly modified from PixelSplat
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        self.benchmarker = benchmarker.Benchmarker()

        self.sh_degree = sh_degree
        self.use_offsets = self.config.use_offsets

        # Perceptual loss (used during training)
        self.lpips_criterion = lpips.LPIPS('vgg')

        # Save hyperparameters: convert Config dataclass to plain dict for Lightning
        # This avoids OmegaConf serialization issues with dataclasses
        # We save the normalized config (self.config) as a plain dict
        try:
            if isinstance(self.config, Config):
                # Convert Config dataclass to plain dict
                hparams_dict = {
                    'image_size': self.config.image_size,
                    'batch_size': self.config.batch_size,
                    'num_views': self.config.num_views,
                    'sh_degree': self.config.sh_degree,
                    'render_res': list(self.config.render_res),  # Convert tuple to list for YAML serialization
                    'use_offsets': self.config.use_offsets,
                    'opt': dict(self.config.opt) if isinstance(self.config.opt, dict) else {},
                    'loss': dict(self.config.loss) if isinstance(self.config.loss, dict) else {},
                }
                self.save_hyperparameters(hparams_dict)
            else:
                # If config is not a Config dataclass, try to save without arguments (auto-detect)
                self.save_hyperparameters()
        except Exception as e:
            # If hyperparameter saving fails, log warning but continue (not critical for training)
            print(f"Warning: Could not save hyperparameters: {e}. Continuing without hyperparameter logging.")


    def forward(self, views):
        # print("DEBUG: forward called")

        # vggt(views) tries to unpack 5 values but get 4, so add another dim
        if len(views.shape) == 4: views = views.unsqueeze(0)

        B_imgs, S_imgs, _, H_img, W_img = views.shape

        # Frozen VGGT inference
        with torch.no_grad():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                # print("Making frozen vggt predictions")
                frozen_vggt_preds = self.vggt(views)

                # Gaussian head is a DPT head and requires certain input tokens
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(views)


        # UNFrozen Gaussian parameter prediction
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.cuda.amp.autocast(dtype=dtype):
            # print("Making gaussian head predictions")
            gaussian_params_preds, _ = self.gaussian_dpt(
                aggregated_tokens_list,
                views,
                patch_start_idx
            )
            
            # Get world_points early to use as reference for spatial dimensions
            world_points = frozen_vggt_preds['world_points']
            B_vggt, S_vggt, H_vggt, W_vggt, _ = world_points.shape
            
            # The DPT head returns preds with shape [B, S, C, H, W] or [B, S, H, W, C]
            # Check the actual shape and handle accordingly
            gaussian_num_channels = self.gaussian_num_channels
            # print(f"DEBUG: gaussian_params_preds shape: {gaussian_params_preds.shape}, expected channels: {gaussian_num_channels}")
            # print(f"DEBUG: world_points spatial dims: H={H_vggt}, W={W_vggt}")
            
            # Handle different possible shapes from DPT head
            if len(gaussian_params_preds.shape) == 5:
                B, S, dim1, dim2, dim3 = gaussian_params_preds.shape
                # print(f"DEBUG: dim1={dim1}, dim2={dim2}, dim3={dim3}")
                
                # Determine which dimension is channels vs spatial dimensions
                # Channels should be gaussian_num_channels (14), spatial dims should be similar (e.g., 518, 518)
                # If dim1 matches expected channels, shape is [B, S, C, H, W]
                if dim1 == gaussian_num_channels:
                    # Shape is [B, S, C, H, W] - correct format
                    C, H, W = dim1, dim2, dim3
                    gaussian_params_flat = gaussian_params_preds.view(B * S, C, H, W)
                # If dim3 matches expected channels, shape is [B, S, H, W, C]
                elif dim3 == gaussian_num_channels:
                    # Shape is [B, S, H, W, C] - need to permute
                    H, W, C = dim1, dim2, dim3
                    gaussian_params_flat = gaussian_params_preds.view(B * S, H, W, C).permute(0, 3, 1, 2)
                else:
                    # Neither dim1 nor dim3 match expected channels
                    # Check if dim1 is much larger (likely channels) and dim2, dim3 are similar (likely spatial)
                    # or if dim3 is much larger (likely channels) and dim1, dim2 are similar (likely spatial)
                    if dim1 > gaussian_num_channels and dim2 == dim3:
                        # Shape is [B, S, C, H, W] where C > gaussian_num_channels and H == W
                        # Take only the first gaussian_num_channels channels
                        C = gaussian_num_channels
                        H, W = dim2, dim3  # These are the spatial dimensions
                        gaussian_params_flat = gaussian_params_preds[:, :, :gaussian_num_channels, :, :].view(B * S, C, H, W)
                    elif dim3 > gaussian_num_channels and dim1 == dim2:
                        # Shape is [B, S, H, W, C] where C > gaussian_num_channels and H == W
                        H, W, C = dim1, dim2, gaussian_num_channels
                        gaussian_params_flat = gaussian_params_preds[:, :, :, :, :gaussian_num_channels].view(B * S, H, W, C).permute(0, 3, 1, 2)
                    elif dim1 > gaussian_num_channels:
                        # Assume dim1 is channels, dim2 and dim3 are spatial (even if not equal)
                        C = gaussian_num_channels
                        H, W = dim2, dim3
                        gaussian_params_flat = gaussian_params_preds[:, :, :gaussian_num_channels, :, :].view(B * S, C, H, W)
                    elif dim3 > gaussian_num_channels:
                        # Assume dim3 is channels, dim1 and dim2 are spatial
                        H, W, C = dim1, dim2, gaussian_num_channels
                        gaussian_params_flat = gaussian_params_preds[:, :, :, :, :gaussian_num_channels].view(B * S, H, W, C).permute(0, 3, 1, 2)
                    else:
                        raise ValueError(
                            f"Unexpected tensor shape: {gaussian_params_preds.shape}. "
                            f"Expected channels ({gaussian_num_channels}) in position 2 or 4, "
                            f"but got dim1={dim1}, dim2={dim2}, dim3={dim3}"
                        )
                
                # print(f"DEBUG: After processing - C={C}, H={H}, W={W}, gaussian_params_flat shape: {gaussian_params_flat.shape}")
                
                # Validate that H and W are reasonable spatial dimensions (not channel counts)
                # H and W should be similar values (e.g., 518, 518) and not equal to the original channel count
                if H == gaussian_num_channels or W == gaussian_num_channels:
                    raise ValueError(
                        f"Shape inference error: H={H}, W={W} but one matches channel count {gaussian_num_channels}. "
                        f"Original shape: {gaussian_params_preds.shape}. "
                        f"This suggests channels and spatial dimensions are confused."
                    )
                if abs(H - W) > 100:  # Spatial dimensions should be similar
                    print(f"WARNING: H={H} and W={W} differ significantly. This might indicate a shape issue.")
            else:
                raise ValueError(f"Unexpected number of dimensions: {len(gaussian_params_preds.shape)}")
            
            # Validate and correct H, W using world_points as reference BEFORE permuting and splitting
            # If inferred H, W don't match world_points, use world_points as ground truth
            H_old, W_old = H, W
            if H != H_vggt or W != W_vggt:
                # Check if one of H, W matches world_points (suggesting a swap or misidentification)
                if (H == H_vggt and W != W_vggt) or (H == W_vggt and W == H_vggt) or (W == H_vggt and H != W_vggt):
                    print(f"WARNING: Gaussian param spatial dims (H={H}, W={W}) don't match world_points ({H_vggt}, {W_vggt}). "
                            f"One dimension matches, correcting to use world_points dimensions.")
                    # Use world_points dimensions as reference
                    H, W = H_vggt, W_vggt
                    # Reshape gaussian_params_flat to match correct spatial dimensions
                    if gaussian_params_flat.shape[2] != H or gaussian_params_flat.shape[3] != W:
                        gaussian_params_flat = F.interpolate(
                            gaussian_params_flat,  # [B*S, C, H_old, W_old]
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False
                        )
                elif H != H_vggt and W != W_vggt:
                    # Neither matches - interpolate to match world_points
                    print(f"INFO: Interpolating Gaussian params from ({H}, {W}) to world_points ({H_vggt}, {W_vggt})")
                    H, W = H_vggt, W_vggt
                    if gaussian_params_flat.shape[2] != H or gaussian_params_flat.shape[3] != W:
                        gaussian_params_flat = F.interpolate(
                            gaussian_params_flat,  # [B*S, C, H_old, W_old]
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False
                        )
            
            # Permute to [B*S, H, W, C] for splitting along channel dimension
            fmap = gaussian_params_flat.permute(0, 2, 3, 1)  # [B*S, H, W, C]
            
            # Split into individual components
            # Order: offsets (3), scales (3), rotations (4), sh (3*sh_degree), opacities (1)
            splits = [3, 3, 4, 3 * self.sh_degree, 1]
            offset, scales, rotations, sh, opacities = torch.split(fmap, splits, dim=-1)
            
            # Post-process each component (now with correct H, W)
            offset = reg_dense_offsets(offset)
            scales = reg_dense_scales(scales)
            rotations = reg_dense_rotation(rotations)
            sh = reg_dense_sh(sh)
            opacities = reg_dense_opacities(opacities)
            
            # Reshape back to [B, S, H, W, ...] with correct H, W
            offset = offset.view(B, S, H, W, 3)
            scales = scales.view(B, S, H, W, 3)
            rotations = rotations.view(B, S, H, W, 4)
            sh = sh.view(B, S, H, W, 3, self.sh_degree)
            opacities = opacities.view(B, S, H, W, 1)
            
            # world_points already has correct spatial dimensions, no need to interpolate
            
            # Compute means: either world_points + offset or just world_points
            if self.use_offsets:
                means = world_points + offset
            else:
                means = world_points

            # Seed SH DC term with the input image colors for meaningful appearance
            base_sh = sh_utils.RGB2SH(einops.rearrange(views, "b s c h w -> b s h w c")).to(sh.dtype)
            # Clone to avoid in-place modification error (sh is a view from torch.split)
            sh = sh.clone()
            sh[..., 0] = sh[..., 0] + base_sh

            # Build covariances for downstream rendering compatibility
            covariances = utils.geometry.build_covariance(scales, rotations)

            # Recover camera intrinsics/extrinsics from the VGGT pose encoding
            camera_extrinsics, camera_intrinsics = None, None
            if "pose_enc" in frozen_vggt_preds:
                pose_enc = frozen_vggt_preds["pose_enc"]
                camera_extrinsics, camera_intrinsics = pose_encoding_to_extri_intri(
                    pose_enc, image_size_hw=(H_img, W_img)
                )
                # Convert extrinsics [B, S, 3, 4] to homogeneous [B, S, 4, 4]
                if camera_extrinsics is not None:
                    pad_row = torch.tensor(
                        [0, 0, 0, 1],
                        device=camera_extrinsics.device,
                        dtype=camera_extrinsics.dtype,
                    ).view(1, 1, 1, 4).expand(camera_extrinsics.shape[0], camera_extrinsics.shape[1], 1, 4)
                    camera_extrinsics = torch.cat([camera_extrinsics, pad_row], dim=2)

            # Build the Gaussian predictions dictionary
            gaussian_head_preds = {
                'pts3d': world_points,
                'means': means,
                'scales': scales,
                'covariances': covariances,
                'rotations': rotations,
                'sh': sh,
                'opacities': opacities,
                'offsets': offset if self.use_offsets else None,
                'camera_extrinsics': camera_extrinsics,
                'camera_intrinsics': camera_intrinsics,
            }

            return frozen_vggt_preds, gaussian_head_preds

    # ----------------------------
    # Training / validation / test
    # ----------------------------
    def _render_from_gaussians(self, gaussian_preds, image_shape_hw):
        """
        Render images from Gaussian predictions using the underlying CUDA splatter.

        Args:
            gaussian_preds: dict returned by forward()[1]
            image_shape_hw: tuple (H, W)

        Returns:
            color: [B, S, 3, H, W] tensor
        """
        from einops import rearrange, repeat  # local import to avoid namespace confusion

        means = gaussian_preds['means']               # [B, S, H, W, 3]
        covariances = gaussian_preds['covariances']   # [B, S, H, W, 3, 3]
        sh = gaussian_preds['sh']                     # [B, S, H, W, 3, sh_degree]
        opacities = gaussian_preds['opacities']       # [B, S, H, W, 1]
        extrinsics = gaussian_preds['camera_extrinsics']  # [B, S, 4, 4]
        intrinsics = gaussian_preds['camera_intrinsics']  # [B, S, 3, 3]

        assert extrinsics is not None and intrinsics is not None, \
            "camera_extrinsics and camera_intrinsics must be present in gaussian_preds for rendering"

        B, S, H, W, _ = means.shape
        near = torch.full((B, S), 0.1, device=means.device, dtype=torch.float32)
        far = torch.full((B, S), 1000.0, device=means.device, dtype=torch.float32)

        # Convert all inputs to float32 for CUDA rasterization (rasterizer expects float32, not bfloat16)
        means_f32 = means.float()
        covariances_f32 = covariances.float()
        sh_f32 = sh.float()
        opacities_f32 = opacities.float()
        extrinsics_f32 = extrinsics.float()
        intrinsics_f32 = intrinsics.float()

        color = pixelsplat_decoder.render_cuda(
            rearrange(extrinsics_f32, "b v i j -> (b v) i j"),
            rearrange(intrinsics_f32, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape_hw,
            repeat(self.decoder.background_color, "c -> (b v) c", b=B, v=S),
            repeat(rearrange(means_f32, "b v h w xyz -> b (v h w) xyz"), "b g xyz -> (b v) g xyz", v=S),
            repeat(rearrange(covariances_f32, "b v h w i j -> b (v h w) i j"), "b g i j -> (b v) g i j", v=S),
            repeat(rearrange(sh_f32, "b v h w c d_sh -> b (v h w) c d_sh"), "b g c d_sh -> (b v) g c d_sh", v=S),
            repeat(rearrange(opacities_f32, "b v h w 1 -> b (v h w)"), "b g -> (b v) g", v=S),
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=B, v=S)
        
        # Clean up intermediate tensors to free memory
        del means_f32, covariances_f32, sh_f32, opacities_f32, extrinsics_f32, intrinsics_f32, near, far
        
        return color

    def _compute_losses(self, batch, rendered_color, views_tensor, mask=None, apply_mask=True, average_over_mask=True, calculate_ssim=False):
        """
        Compute MSE + LPIPS losses between rendered images and target views, with optional masking and SSIM.

        Args:
            batch: data batch (expects batch['context'][i]['original_img'] or batch['target'][i]['original_img'])
            rendered_color: [B, S, C, H, W] rendered images (S = number of views rendered)
            views_tensor:   [B, S, C, H, W] input views (used as fallback target)
            mask: Optional [B, V, H, W] mask tensor (from loss_mask.calculate_loss_mask)
            apply_mask: If True, apply mask to target and predicted colors
            average_over_mask: If True, average losses over masked pixels only
            calculate_ssim: If True, compute and return SSIM metric

        Returns:
            (loss, mse_loss, lpips_loss) or (loss, mse_loss, lpips_loss, ssim_val) if calculate_ssim=True
        """
        B_rendered, S_rendered, C_rendered, H_rendered, W_rendered = rendered_color.shape
        
        # For VGGTGaussians, we render from context views, so we compare against context views
        # This ensures shape matching and semantic correctness
        if 'context' in batch and isinstance(batch['context'], (list, tuple)) and len(batch['context']) >= S_rendered:
            target_color = torch.stack(
                [ctx['original_img'] for ctx in batch['context'][:S_rendered]],
                dim=1
            )  # [B, S, C, H, W]
        elif 'target' in batch and isinstance(batch['target'], (list, tuple)) and len(batch['target']) >= S_rendered:
            # Fallback: use target views if context views not available
            target_color = torch.stack(
                [target_view['original_img'] for target_view in batch['target'][:S_rendered]],
                dim=1
            )  # [B, S, C, H, W]
        else:
            # Final fallback: use input views
            target_color = views_tensor[:, :S_rendered]  # [B, S, C, H, W]

        # Ensure shapes match
        assert rendered_color.shape == target_color.shape, \
            f"rendered_color {rendered_color.shape} and target_color {target_color.shape} must match"

        predicted_color = rendered_color

        # Check if mask shape matches rendered views before applying
        # The mask from calculate_loss_mask is for target views, which may not match context views
        if apply_mask and mask is not None:
            B_mask, V_mask, H_mask, W_mask = mask.shape
            if V_mask == S_rendered and H_mask == H_rendered and W_mask == W_rendered:
                # Mask shape matches - apply it
                assert mask.sum() > 0, "There are no valid pixels in the mask!"
                target_color = target_color * mask[..., None, :, :]
                predicted_color = predicted_color * mask[..., None, :, :]
            else:
                # Mask shape doesn't match - disable masking for this batch
                # This happens when mask is for target views but we're training on context views
                apply_mask = False
                mask = None

        # Flatten for LPIPS and SSIM
        B, V, C, H, W = predicted_color.shape
        flattened_color = einops.rearrange(predicted_color, 'b v c h w -> (b v) c h w')
        flattened_target_color = einops.rearrange(target_color, 'b v c h w -> (b v) c h w')
        flattened_mask = einops.rearrange(mask, 'b v h w -> (b v) h w') if mask is not None else None

        # MSE loss
        rgb_l2_loss = (predicted_color - target_color) ** 2
        if average_over_mask and mask is not None:
            mse_loss = (rgb_l2_loss * mask[:, None, ...]).sum() / mask.sum()
        else:
            mse_loss = rgb_l2_loss.mean()

        # LPIPS loss
        lpips_loss = self.lpips_criterion(flattened_target_color, flattened_color, normalize=True)
        if average_over_mask and flattened_mask is not None:
            lpips_loss = (lpips_loss * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
        else:
            lpips_loss = lpips_loss.mean()

        # Calculate the total loss using config weights
        loss = 0.0
        mse_weight = self.config.loss.get('mse_loss_weight', 1.0) if isinstance(self.config.loss, dict) else getattr(self.config.loss, 'mse_loss_weight', 1.0)
        lpips_weight = self.config.loss.get('lpips_loss_weight', 0.25) if isinstance(self.config.loss, dict) else getattr(self.config.loss, 'lpips_loss_weight', 0.25)
        loss += mse_weight * mse_loss
        loss += lpips_weight * lpips_loss

        # SSIM computation (if requested)
        if calculate_ssim:
            if average_over_mask and flattened_mask is not None:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=True)
                ssim_val = (ssim_val * flattened_mask[:, None, ...]).sum() / flattened_mask.sum()
            else:
                ssim_val = compute_ssim.compute_ssim(flattened_target_color, flattened_color, full=False)
                ssim_val = ssim_val.mean()
            return loss, mse_loss, lpips_loss, ssim_val

        return loss, mse_loss, lpips_loss

    def log_metrics(self, prefix, loss, mse, lpips, ssim=None):
        """
        Log metrics in a format matching Splatt3R's pattern.
        """
        values = {
            f'{prefix}/loss': loss,
            f'{prefix}/mse': mse,
            f'{prefix}/psnr': -10.0 * mse.log10(),
            f'{prefix}/lpips': lpips,
        }
        if ssim is not None:
            values[f'{prefix}/ssim'] = ssim

        prog_bar = prefix != 'val'
        sync_dist = prefix != 'train'
        self.log_dict(values, prog_bar=prog_bar, sync_dist=sync_dist, batch_size=self.config.batch_size)

    def training_step(self, batch, batch_idx):
        """
        Train only the Gaussian head on top of frozen VGGT features.
        We use the two context views as both inputs and reconstruction targets,
        analogous to Splatt3R's reconstruction loss.
        """
        # Build [B, S, C, H, W] tensor from context images
        assert 'context' in batch and len(batch['context']) >= 2, \
            "Expected batch['context'] to contain at least two views"
        # Convert from [-1, 1] (ImgNorm) to [0, 1] (VGGT expects)
        views_normalized = torch.stack(
            [ctx['img'] for ctx in batch['context'][:self.config.num_views]],
            dim=1
        )  # [B, S, C, H, W] in [-1, 1] range from ImgNorm
        # Convert to [0, 1] range: (x + 1) / 2
        views = (views_normalized + 1.0) / 2.0  # Now in [0, 1] range for VGGT

        B, S, C, H, W = views.shape
        with self.benchmarker.time("encoder"):
            vggt_preds, gaussian_preds = self(views)
            # Free VGGT predictions from memory if not needed (we only need gaussian_preds for rendering)
            del vggt_preds

        with self.benchmarker.time("decoder"):
            rendered_color = self._render_from_gaussians(gaussian_preds, image_shape_hw=(H, W))
            # Clear CUDA cache after rendering to free up memory
            torch.cuda.empty_cache()

        # Calculate loss mask if masking is enabled
        mask = None
        apply_mask_config = self.config.loss.get('apply_mask', True) if isinstance(self.config.loss, dict) else getattr(self.config.loss, 'apply_mask', True)
        if apply_mask_config:
            try:
                mask = loss_mask.calculate_loss_mask(batch)
            except Exception as e:
                # If mask calculation fails (e.g., missing depthmaps), continue without mask
                print(f"Warning: Could not calculate loss mask: {e}. Continuing without mask.")
                mask = None

        apply_mask_val = self.config.loss.get('apply_mask', True) if isinstance(self.config.loss, dict) else getattr(self.config.loss, 'apply_mask', True)
        average_over_mask_val = self.config.loss.get('average_over_mask', True) if isinstance(self.config.loss, dict) else getattr(self.config.loss, 'average_over_mask', True)
        loss, mse, lpips_val = self._compute_losses(
            batch, rendered_color, views,
            mask=mask,
            apply_mask=apply_mask_val,
            average_over_mask=average_over_mask_val,
            calculate_ssim=False
        )
        
        # Periodic memory cleanup every 10 batches to prevent OOM
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        self.log_metrics('train', loss, mse, lpips_val)
        return loss

    def validation_step(self, batch, batch_idx):
        # Convert from [-1, 1] (ImgNorm) to [0, 1] (VGGT expects)
        views_normalized = torch.stack(
            [ctx['img'] for ctx in batch['context'][:self.config.num_views]],
            dim=1
        )  # [B, S, C, H, W] in [-1, 1] range from ImgNorm
        # Convert to [0, 1] range: (x + 1) / 2
        views = (views_normalized + 1.0) / 2.0  # Now in [0, 1] range for VGGT
        B, S, C, H, W = views.shape
        vggt_preds, gaussian_preds = self(views)
        rendered_color = self._render_from_gaussians(gaussian_preds, image_shape_hw=(H, W))

        # Calculate loss mask if masking is enabled
        mask = None
        apply_mask_config = self.config.loss.get('apply_mask', True) if isinstance(self.config.loss, dict) else getattr(self.config.loss, 'apply_mask', True)
        if apply_mask_config:
            try:
                mask = loss_mask.calculate_loss_mask(batch)
            except Exception as e:
                print(f"Warning: Could not calculate loss mask: {e}. Continuing without mask.")
                mask = None

        apply_mask_val = self.config.loss.get('apply_mask', True) if isinstance(self.config.loss, dict) else getattr(self.config.loss, 'apply_mask', True)
        average_over_mask_val = self.config.loss.get('average_over_mask', True) if isinstance(self.config.loss, dict) else getattr(self.config.loss, 'average_over_mask', True)
        loss, mse, lpips_val = self._compute_losses(
            batch, rendered_color, views,
            mask=mask,
            apply_mask=apply_mask_val,
            average_over_mask=average_over_mask_val,
            calculate_ssim=False
        )
        self.log_metrics('val', loss, mse, lpips_val)
        return loss

    def test_step(self, batch, batch_idx):
        # Convert from [-1, 1] (ImgNorm) to [0, 1] (VGGT expects)
        views_normalized = torch.stack(
            [ctx['img'] for ctx in batch['context'][:self.config.num_views]],
            dim=1
        )  # [B, S, C, H, W] in [-1, 1] range from ImgNorm
        # Convert to [0, 1] range: (x + 1) / 2
        views = (views_normalized + 1.0) / 2.0  # Now in [0, 1] range for VGGT
        B, S, C, H, W = views.shape
        num_targets = len(batch.get('target', []))
        with self.benchmarker.time("encoder"):
            vggt_preds, gaussian_preds = self(views)
        with self.benchmarker.time("decoder", num_calls=num_targets):
            rendered_color = self._render_from_gaussians(gaussian_preds, image_shape_hw=(H, W))

        # Calculate loss mask if masking is enabled
        mask = None
        apply_mask_config = self.config.loss.get('apply_mask', True) if isinstance(self.config.loss, dict) else getattr(self.config.loss, 'apply_mask', True)
        if apply_mask_config:
            try:
                mask = loss_mask.calculate_loss_mask(batch)
            except Exception as e:
                print(f"Warning: Could not calculate loss mask: {e}. Continuing without mask.")
                mask = None

        apply_mask_val = self.config.loss.get('apply_mask', True) if isinstance(self.config.loss, dict) else getattr(self.config.loss, 'apply_mask', True)
        average_over_mask_val = self.config.loss.get('average_over_mask', True) if isinstance(self.config.loss, dict) else getattr(self.config.loss, 'average_over_mask', True)
        loss, mse, lpips_val, ssim_val = self._compute_losses(
            batch, rendered_color, views,
            mask=mask,
            apply_mask=apply_mask_val,
            average_over_mask=average_over_mask_val,
            calculate_ssim=True
        )
        self.log_metrics('test', loss, mse, lpips_val, ssim=ssim_val)
        return loss

    def configure_optimizers(self):
        """
        Optimizer for the Gaussian head only; VGGT backbone remains frozen.
        """
        # Ensure VGGT stays frozen
        self.vggt.requires_grad_(False)

        # Train only Gaussian DPT head parameters
        params = [p for p in self.gaussian_dpt.parameters() if p.requires_grad]
        opt_lr = self.config.opt.get('lr', 0.0001) if isinstance(self.config.opt, dict) else getattr(self.config.opt, 'lr', 0.0001)
        optimizer = torch.optim.Adam(params, lr=opt_lr)
        
        # Add learning rate scheduler matching Splatt3R
        opt_epochs = self.config.opt.get('epochs', 20) if isinstance(self.config.opt, dict) else getattr(self.config.opt, 'epochs', 20)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[opt_epochs // 2],
            gamma=0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

@dataclass
class Config:
    """Model parameters defined here
    
    Can be used as a dataclass or will be normalized from OmegaConf in VGGTGaussians.__init__
    """
    # Hyperparameters
    image_size: int = 518
    batch_size: int = 1
    num_views: int = 2

    # Optimization parameters (nested structure for Splatt3R compatibility)
    opt: dict = None
    
    # Loss parameters (nested structure matching Splatt3R)
    loss: dict = None
    
    # Rasterizer parameters
    sh_degree: int = 1
    render_res: tuple = (518, 518)
    use_offsets: bool = True
    
    # Legacy fields for backward compatibility
    use_mse_lpips: bool = True
    lambda_mse: float = 1.0
    lambda_ssim: float = 0.05
    mask_strictness: float = 1.0
    lr: float = 0.0001  # Legacy, maps to opt.lr
    
    def __post_init__(self):
        # Set default opt dict if not provided
        if self.opt is None:
            self.opt = {
                'lr': getattr(self, 'lr', 0.0001),
                'epochs': 20,
                'gradient_clip_val': 0.5,
            }
        
        # Set default loss dict if not provided
        if self.loss is None:
            self.loss = {
                'mse_loss_weight': getattr(self, 'lambda_mse', 1.0),
                'lpips_loss_weight': 0.25,
                'apply_mask': True,
                'average_over_mask': True,
            }


def get_args_parser():
    parser = argparse.ArgumentParser(description='VGGT Inference')
    parser.add_argument('--image_size', type=int, default=518,
                    help='Image size')
    # Added this argument to specify which episodes to run inferece on
    parser.add_argument('--do_episodes', type=str, default=None,
                    help='Episodes to process: single value (e.g., 0) or range (e.g., 1-5)')
    parser.prog = 'vggt gaussians'
    return parser


def main():
    print("************************* Begin VGGTGaussians Inference *************************")

    # Instantiate the model
    config = Config()
    model = VGGTGaussians(config)

    total_params = [p for p in model.parameters()]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = [p for p in model.parameters() if not p.requires_grad]
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"total number of model params: {len(total_params)}")
    print(f"number of trainable params: {len(trainable_params)}")
    print(f"number of frozen params: {len(frozen_params)}")
    
    if len(trainable_params) == 0 and len(frozen_params) == len(total_params):
        print("Model frozen successfully")
    else:
        print("Model NOT fully frozen")
        # print(f"un-frozen modules: {trainable_names}")

    # Looping over dataset
    # Parse do_episodes argument
    parser = get_args_parser()
    args = parser.parse_args()
    episodes_to_process = None
    if args.do_episodes:
        if '-' in args.do_episodes:
            start, end = map(int, args.do_episodes.split('-'))
            episodes_to_process = set(range(start, end + 1))
        else:
            episodes_to_process = {int(args.do_episodes)}


    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    episodes_dir = '../episodes'
    for i, episode in enumerate(os.listdir(episodes_dir)):
        # Skip if episode not specified through command line args
        if episodes_to_process is not None and i not in episodes_to_process:
            continue

        print(f"episode_{i:06d} --------------------------------------------------")

        # Access subdirectories for all images from both stationary cameras
        view1_dir = os.path.join(episodes_dir, f'episode_{i:06d}', 'exterior_image_1_left', 'images')
        view2_dir = os.path.join(episodes_dir, f'episode_{i:06d}', 'exterior_image_2_left', 'images')
        view3_dir = os.path.join(episodes_dir, f'episode_{i:06d}', 'wrist_image_left', 'images')
        print(f"view1_dir: {view1_dir}")
        print(f"view2_dir: {view2_dir}")
        print(f"view3_dir: {view3_dir}")

        # Access and sort the files so that the frames are properly ordered
        view1_sorted_files = sorted([file for file in os.listdir(view1_dir) if file.endswith('.png')])
        view2_sorted_files = sorted([file for file in os.listdir(view2_dir) if file.endswith('.png')])
        
        # Initialize lists to store predictions across an episode
        episode_vggt_preds_list = []
        episode_gaussian_preds_list = []

        # Loop over all image pairs step-by-step
        # TODO: try loading third image and/or changing target image
        for step_idx, (view1_img, view2_img) in enumerate(zip(view1_sorted_files, view2_sorted_files)):
            print(f"step_idx: {step_idx}")
            img1_path = os.path.join(view1_dir, view1_img)
            img2_path = os.path.join(view2_dir, view2_img)
            image_names = [img1_path, img2_path]
            images = load_and_preprocess_images(image_names, mode="crop").to(device)

            # Pass the loaded images into VGGT and save the outputs
            # Disable gradients, we are just doing inference
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    # Predict attributes including cameras, depth maps, and point maps.
                    vggt_preds, gaussian_preds = model(images)
                    # print(f"gaussian preds: {gaussian_preds}")

            print(f"len vggt preds items: {len(vggt_preds.items())}")
            print(f"len gaussian preds items: {len(gaussian_preds.items())}")

            # Store the model outputs
            vggt_preds_numpy = {k: v.detach().cpu().float().numpy() if torch.is_tensor(v) else v for k, v in vggt_preds.items()}
            gaussian_preds_numpy = {k: v.detach().cpu().float().numpy() if torch.is_tensor(v) else v for k, v in gaussian_preds.items()}

            # Append to the per-episode prediction lists
            episode_vggt_preds_list.append(vggt_preds_numpy)
            episode_gaussian_preds_list.append(gaussian_preds_numpy)

            print("Got outputs")

            # Clear GPU cache after every step
            torch.cuda.empty_cache()
            del images

        # After all steps, save episode results as .npz in external SSD
        bora_ssd_path = '/media/bora/Extreme Pro/new_proj/vggt_gaussians_outputs'
        os.makedirs(bora_ssd_path, exist_ok=True)
        output_path = os.path.join(bora_ssd_path, f'vggt_gaussians_outputs_{i:06d}.npz')
        np.savez(output_path, vggt_preds=episode_vggt_preds_list, gaussian_preds=episode_gaussian_preds_list)
        print(f"Outputs saved to {output_path}: {os.path.exists(output_path)}")


    print("************************* Finished VGGTGaussians Inference *************************")



if __name__ == "__main__":
    main()
