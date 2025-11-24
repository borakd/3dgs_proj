"""
Functions for visualizing the outputs stored in .npz files for MASt3R, VGGT, and Splatt3R
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

sys.path.append('..')
from src.pixelsplat_src.cuda_splatting import render_cuda
from utils.geometry import build_covariance

# TODO: review this file and improve the quality of the visualiizations. All vibe coded at the moment

def create_visualizations_mast3r(pred, episode_idx, camera_name, step_idx, output_base_dir):
    """Create all visualizations for a single step and camera"""
    
    # Create directory structure
    step_dir = os.path.join(output_base_dir, f'episode_{episode_idx:06d}', camera_name, f'step_{step_idx:06d}')
    os.makedirs(step_dir, exist_ok=True)
    
    # Extract arrays (remove batch dimension)
    # Handle both 'pts3d' and 'pts3d_in_other_view' keys
    pts3d_key = 'pts3d' if 'pts3d' in pred else 'pts3d_in_other_view'
    pts3d = pred[pts3d_key][0] if pred[pts3d_key].ndim > 3 else pred[pts3d_key]  # (512, 512, 3)
    conf = pred['conf'][0] if pred['conf'].ndim > 2 else pred['conf']  # (512, 512)
    desc = pred['desc'][0] if pred['desc'].ndim > 3 else pred['desc']  # (512, 512, 24)
    desc_conf = pred['desc_conf'][0] if pred['desc_conf'].ndim > 2 else pred['desc_conf']  # (512, 512)
    
    # 1. Point cloud visualization
    fig = plt.figure(figsize=(10, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    pts_flat = pts3d.reshape(-1, 3)
    # Sample if too many points
    if len(pts_flat) > 50000:
        indices = np.random.choice(len(pts_flat), 50000, replace=False)
        pts_flat = pts_flat[indices]
    
    # Plot with image-aligned orientation
    ax1.scatter(pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2], s=0.1, alpha=0.5)
    ax1.set_xlabel('X (right)')
    ax1.set_ylabel('Y (down)')
    ax1.set_zlabel('Z (forward)')
    ax1.set_title(f'Point Cloud (step {step_idx})')
    # Set viewing angle to match camera perspective (looking along +Z, Y down)
    ax1.view_init(elev=0, azim=-90)  # Top-down view initially, adjust as needed
    
    # Depth visualization
    ax2 = fig.add_subplot(122, projection='3d')
    depth = np.linalg.norm(pts3d, axis=2)
    x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    ax2.plot_surface(x, y, depth, cmap='viridis', alpha=0.7)
    ax2.set_xlabel('Image X (columns)')
    ax2.set_ylabel('Image Y (rows)')
    ax2.set_zlabel('Depth')
    ax2.set_title(f'Depth Map (step {step_idx})')
    # Match image orientation (Y axis inverted for typical 3D convention)
    ax2.view_init(elev=45, azim=-45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'pointcloud_3d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Confidence heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    im1 = axes[0].imshow(conf, cmap='hot', interpolation='nearest')
    axes[0].set_title('Confidence')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(desc_conf, cmap='hot', interpolation='nearest')
    axes[1].set_title('Descriptor Confidence')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'confidence_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Descriptor PCA visualization
    def desc_to_rgb(desc):
        """Convert descriptor to RGB via PCA"""
        h, w, d = desc.shape
        desc_flat = desc.reshape(-1, d)
        pca = PCA(n_components=3)
        rgb_flat = pca.fit_transform(desc_flat)
        # Normalize to [0, 1]
        rgb_flat = (rgb_flat - rgb_flat.min(axis=0)) / (rgb_flat.max(axis=0) - rgb_flat.min(axis=0) + 1e-8)
        return rgb_flat.reshape(h, w, 3)
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    
    rgb = desc_to_rgb(desc)
    axes.imshow(rgb)
    axes.set_title('Descriptors (PCA to RGB)')
    axes.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'descriptor_pca.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Comprehensive comparison view
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Confidence
    axes[0, 0].imshow(conf, cmap='hot')
    axes[0, 0].set_title('Confidence')
    axes[0, 0].axis('off')
    
    # Descriptors
    axes[0, 1].imshow(rgb)
    axes[0, 1].set_title('Descriptors (PCA)')
    axes[0, 1].axis('off')
    
    # Depth
    depth = np.linalg.norm(pts3d, axis=2)
    im = axes[1, 0].imshow(depth, cmap='viridis')
    axes[1, 0].set_title('Depth')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Descriptor confidence
    axes[1, 1].imshow(desc_conf, cmap='hot')
    axes[1, 1].set_title('Descriptor Confidence')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'comprehensive_view.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Statistics
    stats_file = os.path.join(step_dir, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Statistics for Episode {episode_idx}, Camera {camera_name}, Step {step_idx}\n")
        f.write("=" * 60 + "\n\n")
        
        for key, val in pred.items():
            if isinstance(val, np.ndarray):
                arr = val[0] if val.ndim > 2 else val
                f.write(f"{key}:\n")
                f.write(f"  Shape: {arr.shape}\n")
                f.write(f"  Dtype: {arr.dtype}\n")
                f.write(f"  Min: {arr.min():.6f}\n")
                f.write(f"  Max: {arr.max():.6f}\n")
                f.write(f"  Mean: {arr.mean():.6f}\n")
                f.write(f"  Std: {arr.std():.6f}\n")
                f.write("\n")


def create_visualizations_splatt3r(preds, episode_idx, output_base_dir, steps_to_process=None):
    """
    Comprehensive visualization function for Splatt3R outputs.
    
    Args:
        preds: List containing [pred1_list, pred2_list] from npz file
        episode_idx: Episode index
        output_base_dir: Base directory to save all visualizations
        steps_to_process: Optional set of step indices to process (None = process all)
    """
    pred1_list, pred2_list = preds[0], preds[1]
    
    print(f"\nEpisode {episode_idx}:")
    print(f"  Total steps: {len(pred1_list)}")
    if len(pred1_list) > 0:
        print(f"  Keys in pred1[0]: {list(pred1_list[0].keys())}")
    
    # Determine which steps to process
    if steps_to_process is None:
        # By default, process three evenly spaced steps
        num_steps = len(pred1_list)
        steps_to_check = set(np.linspace(0, num_steps - 1, 3, dtype=int).tolist())
    else:
        steps_to_check = steps_to_process
    
    steps_to_check = sorted(steps_to_check) if isinstance(steps_to_check, set) else sorted(list(steps_to_check))
    
    for step_idx in steps_to_check:
        if step_idx >= len(pred1_list):
            print(f"  Step {step_idx}: OUT OF RANGE")
            continue
        
        pred1 = pred1_list[step_idx]
        pred2 = pred2_list[step_idx]
        
        # Unwrap if needed
        if isinstance(pred1, np.ndarray) and pred1.dtype == object:
            pred1 = pred1.item()
        if isinstance(pred2, np.ndarray) and pred2.dtype == object:
            pred2 = pred2.item()
        
        print(f"  Processing step {step_idx}...")
        
        # Create visualizations for each camera
        _visualize_single_prediction(pred1, episode_idx, 'exterior_image_1_left', step_idx, output_base_dir)
        _visualize_single_prediction(pred2, episode_idx, 'exterior_image_2_left', step_idx, output_base_dir)
        
        # Save combined PLY file (combines both camera views)
        ply_output_dir = os.path.join(output_base_dir, f'episode_{episode_idx:06d}', 'combined_ply')
        os.makedirs(ply_output_dir, exist_ok=True)
        ply_path = os.path.join(ply_output_dir, f'step_{step_idx:06d}.ply')
        save_splatt3r_ply_from_npz(pred1, pred2, ply_path)
    
    print(f"  All visualizations saved to {output_base_dir}")


def _visualize_single_prediction(pred, episode_idx, camera_name, step_idx, output_base_dir):
    """Internal function to visualize a single prediction dictionary"""
    
    # Create directory structure
    step_dir = os.path.join(output_base_dir, f'episode_{episode_idx:06d}', camera_name, f'step_{step_idx:06d}')
    os.makedirs(step_dir, exist_ok=True)
    
    # Extract arrays (remove batch dimension)
    pts3d_key = 'pts3d' if 'pts3d' in pred else 'pts3d_in_other_view'
    means_key = 'means' if 'means' in pred else 'means_in_other_view'
    
    pts3d = pred[pts3d_key][0] if pred[pts3d_key].ndim > 3 else pred[pts3d_key]
    means = pred[means_key][0] if pred[means_key].ndim > 3 else pred[means_key]
    
    # Common outputs
    conf = pred['conf'][0] if pred['conf'].ndim > 2 else pred['conf']
    desc = pred['desc'][0] if pred['desc'].ndim > 3 else pred['desc']
    desc_conf = pred['desc_conf'][0] if pred['desc_conf'].ndim > 2 else pred['desc_conf']
    
    # Gaussian-specific outputs
    scales = pred['scales'][0] if pred['scales'].ndim > 3 else pred['scales']
    rotations = pred['rotations'][0] if pred['rotations'].ndim > 3 else pred['rotations']
    
    # Handle opacities
    opacities = pred['opacities']
    if opacities.ndim > 2:
        opacities = opacities[0]
    if opacities.ndim > 2:
        opacities = opacities.squeeze(-1)
    
    # Handle sh
    sh = pred['sh']
    if sh.ndim > 3:
        sh = sh[0]
    if sh.ndim > 3:
        sh = sh.squeeze(-1)
    
    # 1. Point cloud visualization (using means for Gaussian centers)
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    pts_flat = means.reshape(-1, 3)
    if len(pts_flat) > 50000:
        indices = np.random.choice(len(pts_flat), 50000, replace=False)
        pts_flat = pts_flat[indices]
    
    ax1.scatter(pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2], s=0.1, alpha=0.5)
    ax1.set_xlabel('X (right)')
    ax1.set_ylabel('Y (down)')
    ax1.set_zlabel('Z (forward)')
    ax1.set_title(f'Gaussian Means (step {step_idx})')
    ax1.view_init(elev=0, azim=-90)
    
    # Depth from means
    ax2 = fig.add_subplot(132, projection='3d')
    depth = np.linalg.norm(means, axis=2)
    x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    ax2.plot_surface(x, y, depth, cmap='viridis', alpha=0.7)
    ax2.set_xlabel('Image X (columns)')
    ax2.set_ylabel('Image Y (rows)')
    ax2.set_zlabel('Depth')
    ax2.set_title(f'Depth Map (step {step_idx})')
    ax2.view_init(elev=45, azim=-45)
    
    # Opacity visualization
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(opacities, cmap='hot', interpolation='nearest')
    ax3.set_title('Gaussian Opacities')
    plt.colorbar(im3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'pointcloud_3d.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Gaussian parameters visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Scales (3D -> visualize magnitude)
    scale_mag = np.linalg.norm(scales, axis=2)
    im1 = axes[0, 0].imshow(scale_mag, cmap='viridis', interpolation='nearest')
    axes[0, 0].set_title('Gaussian Scale Magnitude')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Opacities
    im2 = axes[0, 1].imshow(opacities, cmap='hot', interpolation='nearest')
    axes[0, 1].set_title('Gaussian Opacities')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Confidence
    im3 = axes[0, 2].imshow(conf, cmap='hot', interpolation='nearest')
    axes[0, 2].set_title('Confidence')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Spherical Harmonics (first component, RGB-like)
    sh_rgb = sh[..., :3] if sh.shape[-1] >= 3 else sh
    if sh_rgb.ndim > 3:
        sh_rgb = sh_rgb.squeeze()
    sh_rgb_norm = (sh_rgb - sh_rgb.min()) / (sh_rgb.max() - sh_rgb.min() + 1e-8)
    axes[1, 0].imshow(sh_rgb_norm)
    axes[1, 0].set_title('Spherical Harmonics (RGB)')
    axes[1, 0].axis('off')
    
    # Descriptor confidence
    im4 = axes[1, 1].imshow(desc_conf, cmap='hot', interpolation='nearest')
    axes[1, 1].set_title('Descriptor Confidence')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Depth
    depth = np.linalg.norm(means, axis=2)
    im5 = axes[1, 2].imshow(depth, cmap='viridis', interpolation='nearest')
    axes[1, 2].set_title('Depth')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'gaussian_parameters.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Descriptor PCA visualization
    def desc_to_rgb(desc):
        """Convert descriptor to RGB via PCA"""
        h, w, d = desc.shape
        desc_flat = desc.reshape(-1, d)
        pca = PCA(n_components=3)
        rgb_flat = pca.fit_transform(desc_flat)
        rgb_flat = (rgb_flat - rgb_flat.min(axis=0)) / (rgb_flat.max(axis=0) - rgb_flat.min(axis=0) + 1e-8)
        return rgb_flat.reshape(h, w, 3)
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    rgb = desc_to_rgb(desc)
    axes.imshow(rgb)
    axes.set_title('Descriptors (PCA to RGB)')
    axes.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'descriptor_pca.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Comprehensive comparison view
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0, 0].imshow(conf, cmap='hot')
    axes[0, 0].set_title('Confidence')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rgb)
    axes[0, 1].set_title('Descriptors (PCA)')
    axes[0, 1].axis('off')
    
    depth = np.linalg.norm(means, axis=2)
    im = axes[0, 2].imshow(depth, cmap='viridis')
    axes[0, 2].set_title('Depth')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Row 2
    scale_mag = np.linalg.norm(scales, axis=2)
    im = axes[1, 0].imshow(scale_mag, cmap='viridis')
    axes[1, 0].set_title('Gaussian Scale')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].imshow(opacities, cmap='hot')
    axes[1, 1].set_title('Opacities')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    axes[1, 2].imshow(sh_rgb_norm)
    axes[1, 2].set_title('Spherical Harmonics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, 'comprehensive_view.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Statistics
    stats_file = os.path.join(step_dir, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Statistics for Episode {episode_idx}, Camera {camera_name}, Step {step_idx}\n")
        f.write("=" * 60 + "\n\n")
        
        for key in ['pts3d', 'means', 'scales', 'rotations', 'opacities', 'sh', 'conf', 'desc', 'desc_conf', 'covariances']:
            if key in pred:
                val = pred[key]
                if isinstance(val, np.ndarray):
                    arr = val[0] if val.ndim > 2 else val
                    f.write(f"{key}:\n")
                    f.write(f"  Shape: {arr.shape}\n")
                    f.write(f"  Dtype: {arr.dtype}\n")
                    f.write(f"  Min: {arr.min():.6f}\n")
                    f.write(f"  Max: {arr.max():.6f}\n")
                    f.write(f"  Mean: {arr.mean():.6f}\n")
                    f.write(f"  Std: {arr.std():.6f}\n")
                    f.write("\n")


def save_splatt3r_ply_from_npz(pred1, pred2, save_path):
    """
    Save Splatt3R Gaussians as PLY file from numpy predictions.
    Works with outputs loaded from npz files.
    """
    def extract_array(pred, key, default_key=None):
        """Extract array from pred, handling batch dimensions"""
        lookup_key = key if key in pred else default_key
        if lookup_key is None or lookup_key not in pred:
            return None
        
        arr = pred[lookup_key]
        # Remove batch dimension if present
        if arr.ndim > 3:
            arr = arr[0]
        return arr
    
    # Extract means
    means1 = extract_array(pred1, 'means')
    means2 = extract_array(pred2, 'means', 'means_in_other_view')
    
    # Extract scales and rotations
    scales1 = extract_array(pred1, 'scales')
    scales2 = extract_array(pred2, 'scales')
    rotations1 = extract_array(pred1, 'rotations')  # quaternions
    rotations2 = extract_array(pred2, 'rotations')
    
    # Extract opacities
    opacities1 = extract_array(pred1, 'opacities')
    if opacities1.ndim > 2:
        opacities1 = opacities1.squeeze(-1)
    
    opacities2 = extract_array(pred2, 'opacities')
    if opacities2.ndim > 2:
        opacities2 = opacities2.squeeze(-1)
    
    # Extract spherical harmonics
    sh1 = extract_array(pred1, 'sh')
    if sh1.ndim > 3:
        sh1 = sh1.squeeze(-1)  # Remove last dim if (H, W, 3, 1)
    
    sh2 = extract_array(pred2, 'sh')
    if sh2.ndim > 3:
        sh2 = sh2.squeeze(-1)
    
    # Build covariances from scales and rotations if not present
    if 'covariances' in pred1:
        covariances1 = extract_array(pred1, 'covariances')
    else:
        # Convert to torch, build covariance, convert back
        scales1_t = torch.from_numpy(scales1).unsqueeze(0)
        rotations1_t = torch.from_numpy(rotations1).unsqueeze(0)
        covariances1_t = build_covariance(scales1_t, rotations1_t)
        covariances1 = covariances1_t[0].numpy()
    
    if 'covariances' in pred2:
        covariances2 = extract_array(pred2, 'covariances')
    else:
        scales2_t = torch.from_numpy(scales2).unsqueeze(0)
        rotations2_t = torch.from_numpy(rotations2).unsqueeze(0)
        covariances2_t = build_covariance(scales2_t, rotations2_t)
        covariances2 = covariances2_t[0].numpy()
    
    # Reshape to (N, ...) format
    h, w = means1.shape[:2]
    means1_flat = means1.reshape(-1, 3)  # (H*W, 3)
    means2_flat = means2.reshape(-1, 3)
    covariances1_flat = covariances1.reshape(-1, 3, 3)  # (H*W, 3, 3)
    covariances2_flat = covariances2.reshape(-1, 3, 3)
    sh1_flat = sh1.reshape(-1, sh1.shape[-1])  # (H*W, 3)
    sh2_flat = sh2.reshape(-1, sh2.shape[-1])
    opacities1_flat = opacities1.reshape(-1)
    opacities2_flat = opacities2.reshape(-1)
    
    # Combine both views
    means_combined = np.concatenate([means1_flat, means2_flat], axis=0)  # (2*H*W, 3)
    covariances_combined = np.concatenate([covariances1_flat, covariances2_flat], axis=0)  # (2*H*W, 3, 3)
    sh_combined = np.concatenate([sh1_flat, sh2_flat], axis=0)  # (2*H*W, 3)
    opacities_combined = np.concatenate([opacities1_flat, opacities2_flat], axis=0)  # (2*H*W,)
    
    # Convert covariances to quaternions and scales using SVD
    def covariance_to_quaternion_and_scale(covariance):
        """Convert covariance matrix to quaternion and scale"""
        U, S, V = np.linalg.svd(covariance)
        scale = np.sqrt(S)  # (3,)
        rotation_matrix = U @ V.T  # (3, 3)
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # (4,) - [x, y, z, w] format
        return quaternion, scale
    
    # Process all covariances
    num_gaussians = means_combined.shape[0]
    quaternions = np.zeros((num_gaussians, 4))
    scales = np.zeros((num_gaussians, 3))
    
    for i in range(num_gaussians):
        quat, scale = covariance_to_quaternion_and_scale(covariances_combined[i])
        quaternions[i] = quat
        scales[i] = scale
    
    # Use first 3 components of SH as RGB (f_dc)
    sh_rgb = sh_combined[:, :3] if sh_combined.shape[1] >= 3 else sh_combined
    
    # Construct PLY attributes
    # Format: x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
    rest_sh = np.zeros((num_gaussians, 0))  # No rest harmonics for now
    normals = np.zeros((num_gaussians, 3))  # Placeholder normals
    
    attributes = np.concatenate([
        means_combined,           # x, y, z (3)
        normals,                   # nx, ny, nz (3)
        sh_rgb,                    # f_dc_0, f_dc_1, f_dc_2 (3)
        rest_sh,                   # f_rest (0 for now)
        opacities_combined[:, None],  # opacity (1)
        np.log(scales),            # log(scale_0), log(scale_1), log(scale_2) (3)
        quaternions                # rot_0, rot_1, rot_2, rot_3 (4)
    ], axis=1)
    
    # Define PLY format
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    
    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    
    # Save PLY file
    point_cloud = PlyElement.describe(elements, "vertex")
    scene = PlyData([point_cloud])
    scene.write(save_path)
    print(f"  Saved PLY file to {save_path}")


def create_visualizations_vggt(preds_list, episode_idx, output_base_dir, steps_to_process=None):
    """
    Comprehensive visualization function for VGGT outputs.
    
    Args:
        preds_list: List of prediction dictionaries from npz file (one per step)
        episode_idx: Episode index
        output_base_dir: Base directory to save all visualizations
        steps_to_process: Optional set of step indices to process (None = process 3 evenly spaced)
    """
    print(f"\nEpisode {episode_idx}:")
    print(f"  Total steps: {len(preds_list)}")
    if len(preds_list) > 0:
        print(f"  Keys in preds_list[0]: {list(preds_list[0].keys())}")
    
    # Determine which steps to process
    if steps_to_process is None:
        num_steps = len(preds_list)
        steps_to_check = set(np.linspace(0, num_steps - 1, 3, dtype=int).tolist())
    else:
        steps_to_check = steps_to_process
    
    steps_to_check = sorted(steps_to_check) if isinstance(steps_to_check, set) else sorted(list(steps_to_check))
    
    for step_idx in steps_to_check:
        if step_idx >= len(preds_list):
            print(f"  Step {step_idx}: OUT OF RANGE")
            continue
        
        pred = preds_list[step_idx]
        
        # Unwrap if needed
        if isinstance(pred, np.ndarray) and pred.dtype == object:
            pred = pred.item()
        
        if not isinstance(pred, dict):
            print(f"  Step {step_idx}: Not a dict, skipping")
            continue
        
        print(f"  Processing step {step_idx}...")
        
        # Visualize this step's predictions
        _visualize_vggt_single_step(pred, episode_idx, step_idx, output_base_dir)
    
    print(f"  All visualizations saved to {output_base_dir}")


def _visualize_vggt_single_step(pred, episode_idx, step_idx, output_base_dir):
    """Visualize a single step's VGGT predictions"""
    
    # Create directory structure
    step_dir = os.path.join(output_base_dir, f'episode_{episode_idx:06d}', f'step_{step_idx:06d}')
    os.makedirs(step_dir, exist_ok=True)
    
    # Extract arrays (remove batch dimension if present)
    # VGGT outputs are [S, ...] where S is sequence length (number of images)
    
    # Depth maps
    if 'depth' in pred:
        depth = pred['depth']
        if depth.ndim == 5:  # [B, S, H, W, 1]
            depth = depth[0]  # Remove batch dim: [S, H, W, 1]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)  # [S, H, W]
    
    # Depth confidence
    if 'depth_conf' in pred:
        depth_conf = pred['depth_conf']
        if depth_conf.ndim == 4:  # [B, S, H, W]
            depth_conf = depth_conf[0]  # [S, H, W]
    
    # World points
    if 'world_points' in pred:
        world_points = pred['world_points']
        if world_points.ndim == 5:  # [B, S, H, W, 3]
            world_points = world_points[0]  # [S, H, W, 3]
    
    # World points confidence
    if 'world_points_conf' in pred:
        world_points_conf = pred['world_points_conf']
        if world_points_conf.ndim == 4:  # [B, S, H, W]
            world_points_conf = world_points_conf[0]  # [S, H, W]
    
    # Images
    if 'images' in pred:
        images = pred['images']
        if images.ndim == 5:  # [B, S, 3, H, W]
            images = images[0]  # [S, 3, H, W]
        # Convert to [S, H, W, 3] for visualization
        if images.ndim == 4 and images.shape[1] == 3:
            images = images.transpose(0, 2, 3, 1)  # [S, H, W, 3]
    
    # Pose encoding
    if 'pose_enc' in pred:
        pose_enc = pred['pose_enc']
        if pose_enc.ndim == 3:  # [B, S, 9]
            pose_enc = pose_enc[0]  # [S, 9]
    
    # Get sequence length (number of images)
    S = depth.shape[0] if 'depth' in pred else (world_points.shape[0] if 'world_points' in pred else 1)
    
    # Visualize each image in the sequence
    for img_idx in range(S):
        img_dir = os.path.join(step_dir, f'image_{img_idx:02d}')
        os.makedirs(img_dir, exist_ok=True)
        
        # 1. Input image (if available)
        if 'images' in pred:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            img = images[img_idx]
            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0
            ax.imshow(img.clip(0, 1))
            ax.set_title(f'Input Image {img_idx}')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'input_image.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Depth map
        if 'depth' in pred:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            depth_map = depth[img_idx]
            im1 = axes[0].imshow(depth_map, cmap='viridis', interpolation='nearest')
            axes[0].set_title(f'Depth Map {img_idx}')
            plt.colorbar(im1, ax=axes[0])
            
            # Depth histogram
            axes[1].hist(depth_map.flatten(), bins=50, edgecolor='black')
            axes[1].set_xlabel('Depth')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Depth Distribution {img_idx}')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'depth_map.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Depth confidence
        if 'depth_conf' in pred:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            conf_map = depth_conf[img_idx]
            im = ax.imshow(conf_map, cmap='hot', interpolation='nearest')
            ax.set_title(f'Depth Confidence {img_idx}')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'depth_confidence.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. World points (3D point cloud)
        if 'world_points' in pred:
            fig = plt.figure(figsize=(15, 5))
            
            pts3d = world_points[img_idx]  # [H, W, 3]
            pts_flat = pts3d.reshape(-1, 3)
            
            # Sample if too many points
            if len(pts_flat) > 50000:
                indices = np.random.choice(len(pts_flat), 50000, replace=False)
                pts_flat = pts_flat[indices]
            
            # 3D scatter plot
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.scatter(pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2], s=0.1, alpha=0.5)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title(f'World Points 3D {img_idx}')
            ax1.view_init(elev=20, azim=45)
            
            # XY projection
            ax2 = fig.add_subplot(132)
            ax2.scatter(pts_flat[:, 0], pts_flat[:, 1], s=0.1, alpha=0.5)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title(f'World Points XY {img_idx}')
            ax2.set_aspect('equal')
            
            # XZ projection
            ax3 = fig.add_subplot(133)
            ax3.scatter(pts_flat[:, 0], pts_flat[:, 2], s=0.1, alpha=0.5)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Z')
            ax3.set_title(f'World Points XZ {img_idx}')
            ax3.set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'world_points.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 5. World points confidence
        if 'world_points_conf' in pred:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            conf_map = world_points_conf[img_idx]
            im = ax.imshow(conf_map, cmap='hot', interpolation='nearest')
            ax.set_title(f'World Points Confidence {img_idx}')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'world_points_confidence.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 6. Comprehensive view (all outputs together)
        if 'images' in pred and 'depth' in pred and 'world_points' in pred:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Row 1: Image, Depth, Depth Conf
            img = images[img_idx]
            if img.max() > 1.0:
                img = img / 255.0
            axes[0, 0].imshow(img.clip(0, 1))
            axes[0, 0].set_title('Input Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(depth[img_idx], cmap='viridis')
            axes[0, 1].set_title('Depth Map')
            axes[0, 1].axis('off')
            
            if 'depth_conf' in pred:
                axes[0, 2].imshow(depth_conf[img_idx], cmap='hot')
                axes[0, 2].set_title('Depth Confidence')
                axes[0, 2].axis('off')
            else:
                axes[0, 2].axis('off')
            
            # Row 2: World points projections
            pts3d = world_points[img_idx]
            pts_flat = pts3d.reshape(-1, 3)
            if len(pts_flat) > 100000:
                indices = np.random.choice(len(pts_flat), 100000, replace=False)
                pts_flat = pts_flat[indices]
            
            axes[1, 0].scatter(pts_flat[:, 0], pts_flat[:, 1], s=0.1, alpha=0.5)
            axes[1, 0].set_title('World Points XY')
            axes[1, 0].set_aspect('equal')
            
            axes[1, 1].scatter(pts_flat[:, 0], pts_flat[:, 2], s=0.1, alpha=0.5)
            axes[1, 1].set_title('World Points XZ')
            axes[1, 1].set_aspect('equal')
            
            if 'world_points_conf' in pred:
                axes[1, 2].imshow(world_points_conf[img_idx], cmap='hot')
                axes[1, 2].set_title('World Points Confidence')
                axes[1, 2].axis('off')
            else:
                axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'comprehensive_view.png'), dpi=150, bbox_inches='tight')
            plt.close()
    
    # 7. Statistics file
    stats_file = os.path.join(step_dir, 'statistics.txt')
    with open(stats_file, 'w') as f:
        f.write(f"Statistics for Episode {episode_idx}, Step {step_idx}\n")
        f.write("=" * 60 + "\n\n")
        
        for key in ['depth', 'depth_conf', 'world_points', 'world_points_conf', 'pose_enc', 'images']:
            if key in pred:
                val = pred[key]
                # Remove batch dimension if present
                if isinstance(val, np.ndarray):
                    arr = val[0] if val.ndim > 3 else val
                    f.write(f"{key}:\n")
                    f.write(f"  Shape: {arr.shape}\n")
                    f.write(f"  Dtype: {arr.dtype}\n")
                    if arr.size > 0:
                        f.write(f"  Min: {arr.min():.6f}\n")
                        f.write(f"  Max: {arr.max():.6f}\n")
                        f.write(f"  Mean: {arr.mean():.6f}\n")
                        f.write(f"  Std: {arr.std():.6f}\n")
                    f.write("\n")

