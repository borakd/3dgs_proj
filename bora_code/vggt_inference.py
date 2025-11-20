"""
Perform inference with VGGT on DROID 100 dataset

Inputs:
    - pair of images
    - triple of images
Be mindful of which image is chosen as the 'target'
"""

import argparse
import sys
import io
import os
import torch
import numpy as np
sys.path.append('../src/vggt')
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map



def get_args_parser():
    parser = argparse.ArgumentParser(description='VGGT Inference')
    parser.add_argument('--image_size', type=int, default=512,
                    help='Image size')
    # Added this argument to specify which episodes to run inferece on
    parser.add_argument('--do_episodes', type=str, default=None,
                    help='Episodes to process: single value (e.g., 0) or range (e.g., 1-5)')
    parser.prog = 'vggt demo'
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Parse do_episodes argument
    episodes_to_process = None
    if args.do_episodes:
        if '-' in args.do_episodes:
            start, end = map(int, args.do_episodes.split('-'))
            episodes_to_process = set(range(start, end + 1))
        else:
            episodes_to_process = {int(args.do_episodes)}

    # Loop over episodes
    print("************************* Begin VGGT Inference *************************")
    for i, episode in enumerate(os.listdir('episodes')):
        # Skip if episode not specified through command line args
        if episodes_to_process is not None and i not in episodes_to_process:
            continue

        print(f"episode_{i:06d} --------------------------------------------------")

        # Access subdirectories for all images from both stationary cameras
        view1_dir = os.path.join('episodes', f'episode_{i:06d}', 'exterior_image_1_left', 'images')
        view2_dir = os.path.join('episodes', f'episode_{i:06d}', 'exterior_image_2_left', 'images')
        view3_dir = os.path.join('episodes', f'episode_{i:06d}', 'wrist_image_left', 'images')
        print(f"view1_dir: {view1_dir}")
        print(f"view2_dir: {view2_dir}")
        print(f"view3_dir: {view3_dir}")

        # Access and sort the files so that the frames are properly ordered
        view1_sorted_files = sorted([file for file in os.listdir(view1_dir) if file.endswith('.png')])
        view2_sorted_files = sorted([file for file in os.listdir(view2_dir) if file.endswith('.png')])
        # print(f"view1_sorted_files: {view1_sorted_files}")
        # print(f"view2_sorted_files: {view2_sorted_files}")

        # Instantiate lists to store model outputs per-episode
        # VGGT only has one prediction vector because it takes an arbitrary number of frames as input
        # and the output uses the coordinate frame of just the target input frame, so one set of predictions
        episode_preds_list = []

        # Loop over all image pairs step-by-step
        for step_idx, (view1_img, view2_img) in enumerate(zip(view1_sorted_files, view2_sorted_files)):
            print(f"step_idx: {step_idx}")
            img1_path = os.path.join(view1_dir, view1_img)
            img2_path = os.path.join(view2_dir, view2_img)
            # print(f"view1_img: {view1_img}")
            # print(f"view2_img: {view2_img}")
            # print(f"img1_path: {img1_path}")
            # print(f"img2_path: {img2_path}")

            # TODO: try loading third image and/or changing target image
            image_names = [img1_path, img2_path]  
            images = load_and_preprocess_images(image_names).to(device)

            # Pass the loaded images into VGGT and save the outputs
            # Disable gradients, we are just doing inference
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    # Predict attributes including cameras, depth maps, and point maps.
                    predictions = model(images)

            # Store the model outputs
            predictions_numpy = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in predictions.items()}
            episode_preds_list.append(predictions_numpy)

            print("Got outputs")

            # Clear GPU cache after every step
            torch.cuda.empty_cache()
            del images

        # After all steps, save episode results as .npz in external SSD
        bora_ssd_path = '/media/bora/Extreme Pro/new_proj/vggt_outputs'
        os.makedirs(bora_ssd_path, exist_ok=True)
        output_path = os.path.join(bora_ssd_path, f'vggt_outputs_{i:06d}.npz')
        np.savez(output_path, pred1=episode_preds_list)
        print(f"Outputs saved to {output_path}: {os.path.exists(output_path)}")

    print("************************* Finished VGGT Inference *************************")

    # End of main
