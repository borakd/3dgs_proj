"""
Perform inference using the original Splatt3R model, takes two image inputs and passes them through a frozen MASt3R
backbone. The key modification is that MASt3R's transformer decoders both have Gaussian heads which output the model's
learned Gaussian parameters. These are used alongside the point clouds outputted by MASt3R to produce 3D Gaussian splats
and then use differentiable Gaussian rasterization to render 2D image outputs.
"""

import numpy as np
import argparse
import torch
import sys
import os
import io
from dust3r import inference
from dust3r.utils.image import load_images  # for loading image pairs
from mast3r.utils.misc import hash_md5

sys.path.append('..')
import main # main.py from root directory (Splatt3R)

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1


def get_args_parser():
    parser = argparse.ArgumentParser(description='Splatt3R Inference')
    parser.add_argument('--device', type=str, default='cuda',
                    help='PyTorch device')
    parser.add_argument('--image_size', type=int, default=512,
                    help='Image size')
    # Added this argument to specify which episodes to run inferece on
    parser.add_argument('--do_episodes', type=str, default=None,
                    help='Episodes to process: single value (e.g., 0) or range (e.g., 1-5)')
    parser.prog = 'splatt3r demo'
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    weights_path = "/home/bora/Projects/work/splatt3r/checkpoints/splatt3r_v1.0/epoch=19-step=1200.ckpt"
    device = 'cuda'

    # Instantiate the model, MASt3R with Gaussian head, with pretrained weights
    model = main.MAST3RGaussians.load_from_checkpoint(weights_path, map_location='cuda')
    model = model.to('cuda')

    chkpt_tag = hash_md5(weights_path)
    print(f"Model loaded from: {weights_path}")

    # Parse do_episodes argument
    episodes_to_process = None
    if args.do_episodes:
        if '-' in args.do_episodes:
            start, end = map(int, args.do_episodes.split('-'))
            episodes_to_process = set(range(start, end + 1))
        else:
            episodes_to_process = {int(args.do_episodes)}

    # Loop over episodes
    print("************************* Begin Splatt3R Inference *************************")
    for i, episode in enumerate(os.listdir('episodes')):
        # Skip if episode not specified through command line args
        if episodes_to_process is not None and i not in episodes_to_process:
            continue

        print(f"episode_{i:06d} --------------------------------------------------")

        # Access subdirectories for all images from both stationary cameras
        view1_dir = os.path.join('episodes', f'episode_{i:06d}', 'exterior_image_1_left', 'images')
        view2_dir = os.path.join('episodes', f'episode_{i:06d}', 'exterior_image_2_left', 'images')
        print(f"view1_dir: {view1_dir}")
        print(f"view2_dir: {view2_dir}")

        # Access and sort the files so that the frames are properly ordered
        view1_sorted_files = sorted([file for file in os.listdir(view1_dir) if file.endswith('.png')])
        view2_sorted_files = sorted([file for file in os.listdir(view2_dir) if file.endswith('.png')])
        # print(f"view1_sorted_files: {view1_sorted_files}")
        # print(f"view2_sorted_files: {view2_sorted_files}")

        # Instantiate lists to store model outputs per-episode
        episode_pred1_list = []
        episode_pred2_list = []

        # Loop over all image pairs step-by-step
        for step_idx, (view1_img, view2_img) in enumerate(zip(view1_sorted_files, view2_sorted_files)):
            print(f"step_idx: {step_idx}")
            img1_path = os.path.join(view1_dir, view1_img)
            img2_path = os.path.join(view2_dir, view2_img)
            # print(f"view1_img: {view1_img}")
            # print(f"view2_img: {view2_img}")
            # print(f"img1_path: {img1_path}")
            # print(f"img2_path: {img2_path}")

            # Load both images
            img_pair = load_images([img1_path, img2_path], size=args.image_size, verbose=False)

            # Verify that the image pair was loaded properly
            # print(f"Loaded {len(img_pair)} images: {img1_path} -> {img_pair[0]['img'].shape}, {img2_path} -> {img_pair[1]['img'].shape}")

            # Prepare to pass the loaded images into the model,
            # copied the fields which need device placement from original demo code
            # print(f"args.device: {args.device}")
            for img in img_pair:
                img['img'] = img['img'].to(args.device)
                img['original_img'] = img['original_img'].to(args.device)
                img['true_shape'] = torch.from_numpy(img['true_shape'])

            # Pass the loaded images into MASt3R and save the outputs
            # Disable gradients, we are just doing inference
            with torch.no_grad():
                pred1, pred2 = model(img_pair[0], img_pair[1])

            # Store the model outputs
            pred1_numpy = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in pred1.items()}
            pred2_numpy = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in pred2.items()}
            episode_pred1_list.append(pred1_numpy)
            episode_pred2_list.append(pred2_numpy)

            print("Got outputs")

            # Clear GPU cache after every step
            torch.cuda.empty_cache()
            del img_pair

        # After all steps, save episode results as .npz in external SSD
        bora_ssd_path = '/media/bora/Extreme Pro/new_proj/splatt3r_outputs'
        os.makedirs(bora_ssd_path, exist_ok=True)
        output_path = os.path.join(bora_ssd_path, f'splatt3r_outputs_{i:06d}.npz')
        np.savez(output_path, pred1=episode_pred1_list, pred2=episode_pred2_list)
        print(f"Outputs saved to {output_path}: {os.path.exists(output_path)}")

    print("************************* Finished Splatt3R Inference *************************")

    # End of main
