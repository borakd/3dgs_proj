## Quickstart Pipeline for Training VGGTGaussians

Key files:
- `train_cursor.py`
- `configs/vggt_gaussians.yaml`
- `inference_cursor.py`
- `check_outputs_vggt_gaussians.py`

Datasets required: ScanNet++

Files needed:
- `nvs_sem_train.txt`, path: `{data_root}/splits/nvs_sem_train.txt`
- `{sequence_id}` subfolders, path: `{data_root}/data/{sequence_id}`
- `dslr/train_test_lists.json` for all sequence IDs, path: `{data_root}/data/{sequence_id}/dslr/train_test_lists.json`
- `dslr/resized_undistorted_images`, originally just named `undistorted_images`, but the folders have the same contents. Use the former for ease since that is how it came downloaded for every sequence ID.
- `dslr/undistorted_depths` 
- `nerfstudio/transforms.json`, **NOTE:** need to run preprocessing script to obtain depth files. Script is located at: `/src/mast3r_src/dust3r/datasets_preprocess/preprocess_scannetpp.py`
- `nerfstudio/transforms_undistorted.json`

1. Train using the config file: `python bora_code/vggt_gaussians/train_cursor.py configs/vggt_gaussians.yaml`
Be sure to modify {save_dir}/{name}/checkpoints and every_n_epochs to save checkpoints

2. Once training finishes, do inference:
`python bora_code/vggt_gaussians/inference_cursor.py --checkpoint <path_to_ckpt> --episodes_dir <path_to_episodes> --output_dir <path_to_save_outputs> --do_episodes 0-99`

3. Lastly, visualize the inference results and render the Gaussians:
`python bora_code/vggt_gaussians/check_outputs_vggt_gaussians.py --model vggt --do_episodes 0-10 --do_steps 0,50,100 --render_gaussians`
