"""
Checking the outputs stored in .npz files after running VGGTGaussians
"""

import argparse
import numpy as np
import os
import sys

if __name__ == "__main__":
    print(f"Verifying VGGTGaussians outputs")

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

    # Open the directory containing the Splatt3R outputs
    vggt_gaussians_results_dir = os.path.join(results_base_dir, 'vggt_gaussians_outputs')
    vggt_gaussians_visuals_dir = os.path.join(results_base_dir, 'vggt_gaussians_visuals')
    print(f"Checking results in {vggt_gaussians_results_dir}")

    # Loop over the .npz files in the results directory
    results_sorted_files = sorted([file for file in os.listdir(vggt_gaussians_results_dir) if file.endswith('.npz')])
    for episode_idx, file in enumerate(results_sorted_files):
        if episodes_to_process is not None and episode_idx not in episodes_to_process:
            continue
        print(f"Episode {episode_idx}, {file} ------------------------------------------------")
        file_path = os.path.join(vggt_gaussians_results_dir, str(file))
        print(f"file_path: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        data_keys = list(data.keys())
        print(f"List of keys in data: {data_keys}")
        # predictions = [data[key] for key in data_keys]
        # for i, key in enumerate(data_keys):
        #     print(f"key: {key}")
        #     print(f"predictions[{i}].shape = {predictions[i].shape}")
        vggt_preds = data['vggt_preds']
        gaussian_preds = data['gaussian_preds']

        # Check which steps to process for the current episode
        episode_steps_to_process = steps_to_process
        if episode_steps_to_process is None:
            num_steps = len(vggt_preds)
            episode_steps_to_process = sorted(set(np.linspace(0, num_steps - 1, 3, dtype=int).tolist()))
        print(f"episode_steps_to_process = {episode_steps_to_process}")


        # For each step, check the predictions
        for step_idx in episode_steps_to_process:
            vggt_dict = vggt_preds[step_idx]
            gaussian_dict = gaussian_preds[step_idx]

            print("------------ VGGT PREDICTIONS ------------")
            for key, value in vggt_dict.items():
                print(f"vggt_dict[{key}] first 2 values: {vggt_dict[key][:2]}")
                print()
            
            print("------------ GAUSSIAN PREDICTIONS ------------")
            for key, value in gaussian_dict.items():
                print(f"gaussian_dict[{key}] first 2 values: {gaussian_dict[key][:2]}")
                print()
