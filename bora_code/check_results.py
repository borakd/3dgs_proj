"""
Simple script meant to be run after processing some episodes and obtaining model outputs as .npz files.
Verifies that the model outputted everything as intended.
"""

import argparse
import numpy as np
import os
from visualization_utils import create_visualizations_mast3r, create_visualizations_splatt3r, create_visualizations_vggt

# TODO: make variable names more consistent across the three conditional branches
# TODO: fix do_episodes arg being ignored!!

if __name__ == "__main__":
    # Path to the folder containing the subdirectories for per-model inference outputs
    results_base_dir = '/media/bora/Extreme Pro/new_proj'

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
        elif ',' in args.do_episodes:
            episodes_to_process = set(map(int, args.do_episodes.split(',')))
        else:
            episodes_to_process = {int(args.do_episodes)}
    
    # Parse do_steps argument
    steps_to_process = None
    if args.do_steps:
        if '-' in args.do_steps:
            start, end = map(int, args.do_steps.split('-'))
            steps_to_process = set(range(start, end + 1))
        elif ',' in args.do_steps:
            steps_to_process = set(map(int, args.do_steps.split(',')))
        else:
            steps_to_process = {int(args.do_steps)}

    # Check which model type was passed in
    model_type = args.model

    #
    # Checking MASt3R outputs
    #
    if model_type == 'mast3r':
        print(f"Verifying {model_type} outputs")

        # Code below is generated via Cursor
        # TODO: rename these awful dir variables lol
        # MASt3R outputs are saved as: pred1 (list of dicts), pred2 (list of dicts)
        bora_ssd_dir = '/media/bora/Extreme Pro/new_proj/mast3r_outputs'
        output_base_dir = '/media/bora/Extreme Pro/new_proj/mast3r_visuals'

        results_sorted_files = sorted([file for file in os.listdir(bora_ssd_dir) if file.endswith('.npz')])
        
        for episode_idx, file in enumerate(results_sorted_files):
            if episodes_to_process is not None and episode_idx not in episodes_to_process:
                continue
            npz_path = os.path.join(bora_ssd_dir, f'mast3r_outputs_{episode_idx:06d}.npz')
            
            if not os.path.exists(npz_path):
                if episodes_to_process:  # Only warn if explicitly requested
                    print(f"Warning: {npz_path} not found")
                continue
                
            data = np.load(npz_path, allow_pickle=True)
            pred1_list = data['pred1']
            pred2_list = data['pred2']

            # Parse do_steps argument
            # Do it in here because it can change every episode
            episode_steps_to_process = steps_to_process
            if episode_steps_to_process is None:
                # By default, if no steps to process is specific via CLI, process three evenly spaced apart steps
                num_steps = len(pred1_list)
                episode_steps_to_process = sorted(set(np.linspace(0, num_steps - 1, 3, dtype=int).tolist()))
            
            print(f"\nEpisode {episode_idx}:")
            print(f"  Total steps: {len(pred1_list)}")
            print(f"  Keys in pred1[0]: {list(pred1_list[0].keys())}")
            
            # Check specified steps
            steps_to_check = sorted(episode_steps_to_process) if episode_steps_to_process else range(len(pred1_list))
            for step_idx in steps_to_check:
                if step_idx >= len(pred1_list):
                    print(f"  Step {step_idx}: OUT OF RANGE")
                    continue
                    
                pred1 = pred1_list[step_idx]
                pred2 = pred2_list[step_idx]
                
                # Unwrap if needed (npz with allow_pickle=True sometimes wraps in arrays)
                if isinstance(pred1, np.ndarray) and pred1.dtype == object:
                    pred1 = pred1.item()
                if isinstance(pred2, np.ndarray) and pred2.dtype == object:
                    pred2 = pred2.item()
                
                # Debug: check pred2 structure
                print(f"  pred2 type: {type(pred2)}")
                if isinstance(pred2, dict):
                    print(f"  pred2 keys: {list(pred2.keys())}")
                elif isinstance(pred2, np.ndarray):
                    print(f"  pred2 shape: {pred2.shape}, dtype: {pred2.dtype}")
                
                print(f"  Step {step_idx}:")
                for key in pred1.keys():
                    val = pred1[key]
                    if isinstance(val, np.ndarray):
                        print(f"    pred1['{key}']: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"    pred1['{key}']: {type(val)}")
                
                # Create visualizations for each camera
                create_visualizations_mast3r(pred1, episode_idx, 'exterior_image_1_left', step_idx, output_base_dir)
                create_visualizations_mast3r(pred2, episode_idx, 'exterior_image_2_left', step_idx, output_base_dir)
    #
    # Checking VGGT outputs
    #
    elif model_type == 'vggt':
        print(f"Verifying {model_type} outputs")

        # Open the directory containing the Splatt3R outputs
        vggt_results_dir = os.path.join(results_base_dir, 'vggt_outputs')
        vggt_visuals_dir = os.path.join(results_base_dir, 'vggt_visuals')
        print(f"Checking results in {vggt_results_dir}")

        # Loop over the .npz files in the results directory
        results_sorted_files = sorted([file for file in os.listdir(vggt_results_dir) if file.endswith('.npz')])
        for episode_idx, file in enumerate(results_sorted_files):
            if episodes_to_process is not None and episode_idx not in episodes_to_process:
                continue
            print(f"Episode {episode_idx}, {file} ------------------------------------------------")
            file_path = os.path.join(vggt_results_dir, str(file))
            print(f"file_path: {file_path}")
            data = np.load(file_path, allow_pickle=True)
            preds = [data['pred1']]
            # TODO: print keys for each pred
            # Debug prints
            # data.files is the list of key names
            # data['pred'] is the actual array data
            contents = data.files
            print(f"data.files: {contents}")
            for j, elem in enumerate(contents):
                print(f"Data contents {j}: {elem}")
                arr = data[elem]
                if hasattr(arr, 'shape'):
                    print(f"{elem}.shape = {arr.shape}")
                    print(f"{elem}.dtype = {arr.dtype}")
                else:
                    print(f"{elem} type = {type(arr)}")

            # Check which steps to process for the current episode
            episode_steps_to_process = steps_to_process
            if episode_steps_to_process is None:
                num_steps = len(preds[0])
                episode_steps_to_process = sorted(set(np.linspace(0, num_steps - 1, 3, dtype=int).tolist()))
            print(f"episode_steps_to_process = {episode_steps_to_process}")
    
            create_visualizations_vggt(preds[0], episode_idx, vggt_visuals_dir, episode_steps_to_process)

    #
    # Checking Splatt3R outputs
    #
    elif model_type == 'splatt3r':
        print(f"Verifying {model_type} outputs")

        # Open the directory containing the Splatt3R outputs
        splatt3r_results_dir = os.path.join(results_base_dir, 'splatt3r_outputs')
        splatt3r_visuals_dir = os.path.join(results_base_dir, 'splatt3r_visuals')
        print(f"Checking results in {splatt3r_results_dir}")

        # Loop over the .npz files in the results directory
        results_sorted_files = sorted([file for file in os.listdir(splatt3r_results_dir) if file.endswith('.npz')])
        for episode_idx, file in enumerate(results_sorted_files):
            if episodes_to_process is not None and episode_idx not in episodes_to_process:
                continue
            print(f"Episode {episode_idx}, {file} ------------------------------------------------")
            file_path = os.path.join(splatt3r_results_dir, str(file))
            print(f"file_path: {file_path}")
            data = np.load(file_path, allow_pickle=True)
            preds = [data['pred1'], data['pred2']]
            # TODO: print keys for each pred
            # # Debug prints
            # contents = data.files
            # print(f"data.files: {contents}")
            # for j, elem in enumerate(contents):
            #     print(f"Data contents {j}: {elem}")
            #     preds.append(data[elem])
            #     # print(f"preds[{i}]: {preds[i]}")
            #     print(f"preds[{j}].shape = {preds[j].shape}")
            #     print(f"preds[{j}].dtype = {preds[j].dtype}")

            # Check which steps to process for the current episode
            episode_steps_to_process = steps_to_process
            if episode_steps_to_process is None:
                num_steps = len(preds[0])
                episode_steps_to_process = sorted(set(np.linspace(0, num_steps - 1, 3, dtype=int).tolist()))
            print(f"episode_steps_to_process = {episode_steps_to_process}")

            # Preds are extracted, now call the visualization helper
            create_visualizations_splatt3r(preds, episode_idx, splatt3r_visuals_dir, episode_steps_to_process)

                