"""
VGGTGaussians Inference Script
Loads a trained checkpoint and runs inference on image pairs, saving outputs for visualization.
"""

import os
import sys
import torch
import numpy as np
import argparse
import re
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
                        default='/media/bora/Extreme Pro/new_proj/vggt_gaussians_outputs/training',
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
    
    # ===== STRICT VERIFICATION: Ensure trained weights are actually loaded =====
    print(f"\n   [VERIFICATION] Checking that trained weights were loaded...")
    checkpoint_state = checkpoint.get('state_dict', checkpoint)
    
    # Check what Gaussian head parameters are in the checkpoint
    gaussian_keys_in_ckpt = [k for k in checkpoint_state.keys() if 'gaussian_dpt' in k]
    print(f"   [VERIFICATION] Gaussian head keys in checkpoint: {len(gaussian_keys_in_ckpt)}")
    if len(gaussian_keys_in_ckpt) == 0:
        raise RuntimeError("   ❌ ERROR: No Gaussian head weights found in checkpoint! Model may not have trained properly.")
    
    print(f"   [VERIFICATION] Sample checkpoint keys: {gaussian_keys_in_ckpt[:3]}...")
    
    # Check a sample weight value from checkpoint
    sample_key = gaussian_keys_in_ckpt[0]
    sample_weight_ckpt = checkpoint_state[sample_key]
    if torch.is_tensor(sample_weight_ckpt):
        print(f"   [VERIFICATION] Sample checkpoint weight shape: {sample_weight_ckpt.shape}")
        print(f"   [VERIFICATION] Sample checkpoint weight - mean: {sample_weight_ckpt.float().mean().item():.6f}, std: {sample_weight_ckpt.float().std().item():.6f}")
    
    # Check what's actually in the loaded model
    model_state_dict = model.state_dict()
    model_gaussian_keys = [k for k in model_state_dict.keys() if 'gaussian_dpt' in k]
    print(f"   [VERIFICATION] Gaussian head keys in loaded model: {len(model_gaussian_keys)}")
    
    if len(model_gaussian_keys) == 0:
        raise RuntimeError("   ❌ ERROR: No Gaussian head found in loaded model!")
    
    # Verify weights match between checkpoint and loaded model
    print(f"   [VERIFICATION] Verifying weights match checkpoint...")
    matched_keys = 0
    mismatched_keys = 0
    missing_keys = []
    
    for key in gaussian_keys_in_ckpt:
        if key in model_state_dict:
            ckpt_val = checkpoint_state[key]
            model_val = model_state_dict[key]
            if torch.is_tensor(ckpt_val) and torch.is_tensor(model_val):
                if torch.allclose(ckpt_val.float().cpu(), model_val.float().cpu(), atol=1e-4):
                    matched_keys += 1
                else:
                    mismatched_keys += 1
                    if mismatched_keys == 1:  # Print details for first mismatch
                        print(f"   [VERIFICATION] ⚠ First mismatch at key: {key}")
                        print(f"      Checkpoint mean: {ckpt_val.float().mean().item():.6f}, std: {ckpt_val.float().std().item():.6f}")
                        print(f"      Model mean: {model_val.float().mean().item():.6f}, std: {model_val.float().std().item():.6f}")
        else:
            missing_keys.append(key)
    
    print(f"   [VERIFICATION] Weight matching: {matched_keys} matched, {mismatched_keys} mismatched, {len(missing_keys)} missing")
    
    if mismatched_keys > 0 or len(missing_keys) > 0:
        print(f"   [VERIFICATION] ⚠ WARNING: Some weights don't match! This may indicate loading issues.")
    
    # Create a fresh untrained model to compare weights
    print(f"   [VERIFICATION] Creating fresh untrained model for comparison...")
    fresh_model = VGGTGaussians(config)
    fresh_model = fresh_model.to(args.device)
    fresh_model.eval()
    fresh_state_dict = fresh_model.state_dict()
    
    # Compare sample weights between trained and untrained models
    print(f"   [VERIFICATION] Comparing trained vs. untrained weights...")
    sample_key = gaussian_keys_in_ckpt[0]
    trained_val = None
    untrained_val = None
    weights_match_initialization = False
    
    if sample_key in model_state_dict and sample_key in fresh_state_dict:
        trained_val = model_state_dict[sample_key].float()
        untrained_val = fresh_state_dict[sample_key].float()
        
        if torch.allclose(trained_val, untrained_val, atol=1e-4):
            weights_match_initialization = True
            raise RuntimeError(
                f"   ❌ ERROR: Trained model weights match untrained initialization!\n"
                f"      This means the checkpoint weights were NOT loaded properly.\n"
                f"      Sample key: {sample_key}\n"
                f"      Both have mean: {trained_val.mean().item():.6f}, std: {trained_val.std().item():.6f}"
            )
        else:
            print(f"   [VERIFICATION] ✓ Verified: Trained weights differ from initialization")
            print(f"      Trained mean: {trained_val.mean().item():.6f}, std: {trained_val.std().item():.6f}")
            print(f"      Untrained mean: {untrained_val.mean().item():.6f}, std: {untrained_val.std().item():.6f}")
            print(f"      Difference (mean): {abs(trained_val.mean() - untrained_val.mean()).item():.6f}")
    else:
        print(f"   [VERIFICATION] ⚠ WARNING: Could not compare weights (sample key not found in both models)")
    
    # Check checkpoint metadata to verify training occurred
    print(f"   [VERIFICATION] Checking checkpoint training metadata...")
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
        print(f"   [VERIFICATION] Checkpoint epoch: {epoch}")
        if epoch == 0:
            print(f"   [VERIFICATION] ⚠ WARNING: Checkpoint is from epoch 0 - may not have trained")
    else:
        print(f"   [VERIFICATION] ⚠ WARNING: No epoch information in checkpoint")
    
    if 'global_step' in checkpoint:
        step = checkpoint['global_step']
        print(f"   [VERIFICATION] Checkpoint global_step: {step}")
        if step == 0:
            print(f"   [VERIFICATION] ⚠ WARNING: Checkpoint is from step 0 - may not have trained")
    
    # Check hyperparameters for training loss (if logged)
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"   [VERIFICATION] Hyperparameters found in checkpoint")
        # Note: Training loss might be in checkpoint filename or logged separately
    
    # Extract training loss from checkpoint filename if available
    checkpoint_name = os.path.basename(args.checkpoint)
    loss_match = re.search(r'train_loss=([\d.]+)', checkpoint_name)
    if loss_match:
        train_loss = float(loss_match.group(1))
        print(f"   [VERIFICATION] Training loss from filename: {train_loss:.6f}")
        if train_loss == 0.0:
            print(f"   [VERIFICATION] ⚠ WARNING: Training loss is 0.0 - this is suspicious!")
            print(f"   [VERIFICATION] ⚠ This might indicate the loss wasn't logged properly or model didn't train")
    
    # Check for optimizer state (indicates training occurred)
    if 'optimizer_states' in checkpoint or 'lr_schedulers' in checkpoint:
        print(f"   [VERIFICATION] ✓ Optimizer/scheduler state found (training occurred)")
    else:
        print(f"   [VERIFICATION] ⚠ WARNING: No optimizer state found in checkpoint")
    
    # Check if weights match initialization pattern (would indicate untrained model)
    print(f"   [VERIFICATION] Checking if weights match initialization pattern...")
    final_conv_key = None
    for key in gaussian_keys_in_ckpt:
        if 'scratch.output_conv2' in key and 'weight' in key:
            final_conv_key = key
            break
    
    if final_conv_key and final_conv_key in model_state_dict:
        final_conv_weight = model_state_dict[final_conv_key].float()
        # Check bias initialization pattern (scales should have bias ~-7.0, opacity ~-2.0)
        bias_key = final_conv_key.replace('.weight', '.bias')
        if bias_key in model_state_dict:
            final_conv_bias = model_state_dict[bias_key].float()
            # Scales bias should be around -7.0 (channels 3-5), opacity should be around -2.0 (last channel)
            scales_bias = final_conv_bias[3:6].mean().item()
            opacity_bias = final_conv_bias[-1].item()
            
            print(f"   [VERIFICATION] Final conv layer - scales bias: {scales_bias:.6f} (expected ~-7.0), opacity bias: {opacity_bias:.6f} (expected ~-2.0)")
            
            # If biases match initialization exactly, weights might not be trained
            if abs(scales_bias - (-7.0)) < 0.01 and abs(opacity_bias - (-2.0)) < 0.01:
                print(f"   [VERIFICATION] ⚠ WARNING: Biases match initialization pattern exactly - weights may not be trained!")
            else:
                print(f"   [VERIFICATION] ✓ Biases differ from initialization (likely trained)")
    
    # Clean up fresh model to free memory
    del fresh_model, fresh_state_dict
    torch.cuda.empty_cache()
    
    # Final verification summary
    print(f"   [VERIFICATION] ========== VERIFICATION SUMMARY ==========")
    verification_passed = True
    verification_warnings = []
    
    if len(gaussian_keys_in_ckpt) == 0:
        verification_passed = False
        verification_warnings.append("No Gaussian head weights in checkpoint")
    
    if mismatched_keys > len(gaussian_keys_in_ckpt) * 0.1:  # More than 10% mismatched
        verification_passed = False
        verification_warnings.append(f"Too many mismatched weights ({mismatched_keys}/{len(gaussian_keys_in_ckpt)})")
    
    if weights_match_initialization:
        verification_passed = False
        verification_warnings.append("Trained weights match untrained initialization")
    
    if trained_val is None or untrained_val is None:
        verification_warnings.append("Could not compare trained vs. untrained weights")
    
    if verification_passed:
        print(f"   [VERIFICATION] ✓ VERIFICATION PASSED: Trained weights confirmed loaded")
    else:
        print(f"   [VERIFICATION] ❌ VERIFICATION FAILED:")
        for warning in verification_warnings:
            print(f"      - {warning}")
        raise RuntimeError("Verification failed: Trained weights may not be loaded correctly!")
    print(f"   [VERIFICATION] ==========================================\n")
    # ===== END VERIFICATION =====
    
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

