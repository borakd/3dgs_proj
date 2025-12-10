Key files: `train_cursor.py`, `configs/vggt_gaussians.yaml`, `inference_cursor.py`, `check_outputs_vggt_gaussians.py`

Train using the config file: `python bora_code/vggt_gaussians/train_cursor.py configs/vggt_gaussians.yaml`
Be sure to modify {save_dir}/{name}/checkpoints and every_n_epochs to save checkpoints

Once training finishes, do inference:
`python bora_code/vggt_gaussians/inference_cursor.py --checkpoint <path_to_ckpt> --episodes_dir <path_to_episodes> --output_dir <path_to_save_outputs> --do_episodes 0-99`

Lastly, visualize the inference results and render the Gaussians:
`python bora_code/vggt_gaussians/check_outputs_vggt_gaussians.py --model vggt --do_episodes 0-10 --do_steps 0,50,100 --render_gaussians`
