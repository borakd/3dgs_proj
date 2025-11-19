"""
Simple script meant to be run after processing some episodes and obtaining model outputs as .npz files.
Verifies that the model outputted everything as intended.
"""

import argparse
import numpy as np
import os

if __name__ == "__main__":

    # Add command line arguments for the type of model, the episode number, and the step number
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mast3r', 'vggt', 'splatt3r'], help="Model type to check")
    parser.add_argument('--do_episodes', type=str, default=None,
                    help='Episodes to process: zero-indexed single value (e.g., 0) or range (e.g., 1-5)')
    parser.add_argument('--do_steps', type=str, default=None,
                    help='Steps to process: zero-indexed single value (e.g., 0) or range (e.g., 1-5)')
    args = parser.parse_args()

    # Parse do_episodes argument
    episodes_to_process = None
    if args.do_episodes:
        if '-' in args.do_episodes:
            start, end = map(int, args.do_episodes.split('-'))
            episodes_to_process = set(range(start, end + 1))
        else:
            episodes_to_process = {int(args.do_episodes)}
    
    # Parse do_steps argument
    steps_to_process = None
    if args.do_steps:
        if '-' in args.do_steps:
            start, end = map(int, args.do_steps.split('-'))
            steps_to_process = set(range(start, end + 1))
        else:
            steps_to_process = {int(args.do_steps)}

    # Check which model type was passed in
    model_type = args.model

    #
    # Checking MASt3R outputs
    #
    if model_type == 'mast3r':
        print(f"Verifying {model_type} outputs")

        # MASt3R outputs are saved as: pred1 (list of dicts), pred2 (list of dicts)
        bora_ssd_dir = '/media/bora/Extreme Pro/new_proj/mast3r_outputs'
        
        for episode_idx in (episodes_to_process if episodes_to_process else range(1000)):  # Adjust max as needed
            npz_path = os.path.join(bora_ssd_dir, f'mast3r_outputs_{episode_idx:06d}.npz')
            
            if not os.path.exists(npz_path):
                if episodes_to_process:  # Only warn if explicitly requested
                    print(f"Warning: {npz_path} not found")
                continue
                
            data = np.load(npz_path, allow_pickle=True)
            pred1_list = data['pred1']
            pred2_list = data['pred2']
            
            print(f"\nEpisode {episode_idx}:")
            print(f"  Total steps: {len(pred1_list)}")
            print(f"  Keys in pred1[0]: {list(pred1_list[0].keys())}")
            
            # Check specified steps
            steps_to_check = steps_to_process if steps_to_process else range(len(pred1_list))
            for step_idx in steps_to_check:
                if step_idx >= len(pred1_list):
                    print(f"  Step {step_idx}: OUT OF RANGE")
                    continue
                    
                pred1 = pred1_list[step_idx]
                pred2 = pred2_list[step_idx]
                
                print(f"  Step {step_idx}:")
                for key in pred1.keys():
                    val = pred1[key]
                    if isinstance(val, np.ndarray):
                        print(f"    pred1['{key}']: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"    pred1['{key}']: {type(val)}")
    #
    # Checking VGGT outputs
    #
    elif model_type == 'vggt':
        print(f"Verifying {model_type} outputs")
    #
    # Checking Splatt3R outputs
    #
    elif model_type == 'splatt3r':
        print(f"Verifying {model_type} outputs")

