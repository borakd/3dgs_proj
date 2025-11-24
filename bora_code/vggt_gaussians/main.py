"""
Re-implementing Splatt3R but with VGGT backbone. Freeze the VGGT, train the Gaussian head,
then freeze everything for inference.
"""

import os
import sys
import torch
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
        
        # Save the config
        self.config = config
        # Not sure what this is for exactly, Gemini 3 suggested it as good practice in Lightning
        self.save_hyperparameters(config.__dict__)
        # Instantiate generic benchmarker
        self.benchmarker = benchmarker.Benchmarker()
        
        # Load pretrained VGGT and freeze it completely
        # VGGT does not have separate encoder/decoder architecture, rather it is one deep transformer
        print("Loading VGGT base model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device: {device}")
        self.vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        self.vggt.requires_grad_(False)
        # TODO: eval mode needed?
        # self.vggt.eval()

        # Debug print
        # print(f"Aggregator instance attributes: {vars(self.vggt.aggregator)}")
        print(f"input token shape: {self.vggt.aggregator.camera_token.shape}")
        embed_dim = self.vggt.aggregator.camera_token.shape[-1]
        print(f"embed dim (last dim of input): {embed_dim}")

        # Define the Gaussian head
        # Computing the output dim:
        # position offset = 3
        # opacity = 1
        # scale = 3
        # rotation = 4
        # color (SH) = 3
        # total = 14
        sh_degree = config.sh_degree
        self.sh_degree = sh_degree
        gaussian_num_channels = 3 + 3 + 4 + 3 * sh_degree + 1        # TODO: consider changing SH degree from 0 to 1 for view-dependent effects
        #       degree 0 = 1 coeeficient per color (3 total)
        #       degree 1 = 4 coefficients per color (12 total)
        #       with degree 1 SH output dim becomes 23
        self.gaussian_dpt = DPTHead(
            dim_in=2 * embed_dim,
            output_dim = gaussian_num_channels,
            activation='linear',
            conf_activation='expp1'
        ).to(device)

        # The decoder which we use to render the predicted Gaussians into
        # images, lightly modified from PixelSplat
        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        # TODO: add losses?


    def forward(self, views):
        print("DEBUG: forward called")

        if len(views.shape) == 4: views = views.unsqueeze(0)

        with torch.no_grad():
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                # print(f"views: {views}")
                # Predict attributes including cameras, depth maps, and point maps.
                print("Making frozen vggt predictions")
                frozen_vggt_preds = self.vggt(views)

                # Gaussian head is a DPT head and requires certain input tokens
                aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(views)
                # print(f"aggregated tokens list: {aggregated_tokens_list}")
                # print(f"patch start idx: {patch_start_idx}")

                print("Making gaussian head predictions")
                gaussian_params_preds, gaussian_conf_preds = self.gaussian_dpt(
                    aggregated_tokens_list, views, patch_start_idx
                )
                gaussian_head_preds = {
                    'gaussian_params': gaussian_params_preds,
                    'gaussian_conf': gaussian_conf_preds
                }

            return frozen_vggt_preds, gaussian_head_preds

    # TODO: these
    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

@dataclass(frozen=True)
class Config:
    """Model parameters defined here
    """
    # TODO: review these values, they are placeholders
    # Hyperparameters
    image_size: int = 512
    lr: float = 0.0001
    batch_size: int = 1
    num_views: int = 2

    # Loss params
    use_mse_lpips: bool = True
    lambda_mse: float = 1.0
    lambda_ssim: float = 0.05

    # Masking strength
    mask_strictness: float = 1.0

    # Rasterizer parameters
    sh_degree: int = 1
    render_res: tuple = (512, 512)


def get_args_parser():
    parser = argparse.ArgumentParser(description='VGGT Inference')
    parser.add_argument('--image_size', type=int, default=512,
                    help='Image size')
    # Added this argument to specify which episodes to run inferece on
    parser.add_argument('--do_episodes', type=str, default=None,
                    help='Episodes to process: single value (e.g., 0) or range (e.g., 1-5)')
    parser.prog = 'vggt gaussians'
    return parser


if __name__ == "__main__":
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
            images = load_and_preprocess_images(image_names).to(device)

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

    # End of main