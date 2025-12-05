"""
VGGTGaussians Inference Script
Loads a trained checkpoint and runs inference on image pairs, saving outputs for visualization.
"""

import os
import sys
import torch
import numpy as np
import argparse
import lightning as L

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'src/vggt'))

from bora_code.vggt_gaussians.main_cursor import VGGTGaussians, Config
from vggt.vggt.utils.load_fn import load_and_preprocess_images


def get_args_parser():
    parser = argparse.ArgumentParser(description='VGGTGaussians Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (.ckpt file)')
    parser.add_argument('--episodes_dir', type=str, default='bora_code/episodes',
                        help='Directory containing episode subdirectories')
    parser.add_argument('--output_dir', type=str, 
                        default='/media/bora/Extreme Pro/new_proj/vggt_gaussians_outputs/inference',
                        help='Directory to save .npz output files')
    parser.add_argument('--do_episodes', type=str, default=None,
                        help='Episodes to process: single value (e.g., 0) or range (e.g., 1-5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='PyTorch device (cuda or cpu)')
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    print("=" * 80)
    print("VGGTGaussians Inference")
    print("=" * 80)
    
    # Load checkpoint
    print(f"\n1. Loading checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Load checkpoint to extract hyperparameters first
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Extract hyperparameters and create Config
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"   Found hyperparameters in checkpoint")
        # Reconstruct Config from saved hyperparameters
        config = Config(
            image_size=hparams.get('image_size', 518),
            batch_size=hparams.get('batch_size', 1),
            num_views=hparams.get('num_views', 2),
            sh_degree=hparams.get('sh_degree', 1),
            render_res=tuple(hparams.get('render_res', [518, 518])),
            use_offsets=hparams.get('use_offsets', True),
            opt=hparams.get('opt', {}),
            loss=hparams.get('loss', {})
        )
    else:
        # Fallback to default config if hyperparameters not found
        print("   Warning: No hyperparameters found in checkpoint, using default config")
        config = Config()
    
    # Load model from checkpoint with config
    model = VGGTGaussians.load_from_checkpoint(
        args.checkpoint,
        config=config,
        map_location=args.device,
        strict=False  # Allow some flexibility if config changed slightly
    )
    model = model.to(args.device)
    model.eval()  # Set to evaluation mode
    
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Model on device: {args.device}")
    
    # Parse episodes to process
    episodes_to_process = None
    if args.do_episodes:
        if '-' in args.do_episodes:
            start, end = map(int, args.do_episodes.split('-'))
            episodes_to_process = set(range(start, end + 1))
        elif ',' in args.do_episodes:
            episodes_to_process = set(map(int, args.do_episodes.split(',')))
        else:
            episodes_to_process = {int(args.do_episodes)}
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n2. Output directory: {args.output_dir}")
    
    # Set up for inference
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"   Using dtype: {dtype}")
    
    # Process episodes
    print(f"\n3. Processing episodes from: {args.episodes_dir}")
    episodes_list = sorted([d for d in os.listdir(args.episodes_dir) if d.startswith('episode_')])
    
    for episode_dir in episodes_list:
        # Extract episode index from directory name (e.g., "episode_000001" -> 1)
        try:
            episode_idx = int(episode_dir.split('_')[-1])
        except ValueError:
            print(f"   ⚠ Skipping invalid episode directory: {episode_dir}")
            continue
        
        # Skip if not in episodes_to_process
        if episodes_to_process is not None and episode_idx not in episodes_to_process:
            continue
        
        print(f"\n   Processing episode_{episode_idx:06d}")
        
        # Find image directories
        episode_path = os.path.join(args.episodes_dir, episode_dir)
        view1_dir = os.path.join(episode_path, 'exterior_image_1_left', 'images')
        view2_dir = os.path.join(episode_path, 'exterior_image_2_left', 'images')
        
        if not os.path.exists(view1_dir) or not os.path.exists(view2_dir):
            print(f"   ⚠ Skipping: image directories not found")
            continue
        
        # Get sorted image files
        view1_files = sorted([f for f in os.listdir(view1_dir) if f.endswith('.png')])
        view2_files = sorted([f for f in os.listdir(view2_dir) if f.endswith('.png')])
        
        if len(view1_files) != len(view2_files):
            print(f"   ⚠ Skipping: mismatched number of images ({len(view1_files)} vs {len(view2_files)})")
            continue
        
        # Initialize lists to store predictions
        episode_vggt_preds_list = []
        episode_gaussian_preds_list = []
        
        # Process each image pair
        for step_idx, (view1_file, view2_file) in enumerate(zip(view1_files, view2_files)):
            if step_idx % 10 == 0:
                print(f"      Step {step_idx}/{len(view1_files)}")
            
            img1_path = os.path.join(view1_dir, view1_file)
            img2_path = os.path.join(view2_dir, view2_file)
            image_paths = [img1_path, img2_path]
            
            # Load and preprocess images
            images = load_and_preprocess_images(image_paths, mode="crop").to(args.device)
            
            # Run inference
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    vggt_preds, gaussian_preds = model(images)
            
            # Convert to numpy and store
            vggt_preds_numpy = {
                k: v.detach().cpu().float().numpy() if torch.is_tensor(v) else v
                for k, v in vggt_preds.items()
            }
            gaussian_preds_numpy = {
                k: v.detach().cpu().float().numpy() if torch.is_tensor(v) else v
                for k, v in gaussian_preds.items()
            }
            
            episode_vggt_preds_list.append(vggt_preds_numpy)
            episode_gaussian_preds_list.append(gaussian_preds_numpy)
            
            # Clean up
            torch.cuda.empty_cache()
            del images
        
        # Save episode outputs
        output_path = os.path.join(args.output_dir, f'vggt_gaussians_outputs_{episode_idx:06d}.npz')
        np.savez(output_path, 
                 vggt_preds=episode_vggt_preds_list, 
                 gaussian_preds=episode_gaussian_preds_list)
        print(f"   ✓ Saved {len(episode_vggt_preds_list)} steps to {output_path}")
    
    print("\n" + "=" * 80)
    print("Inference Complete!")
    print("=" * 80)
    print(f"\nNext step: Run visualization with:")
    print(f"  python bora_code/vggt_gaussians/check_outputs_vggt_gaussians.py \\")
    print(f"    --model vggt \\")
    print(f"    --do_episodes <episode_numbers> \\")
    print(f"    --render_gaussians")


if __name__ == "__main__":
    main()

