"""
Functions for visualizing the outputs stored in .npz files for MASt3R, VGGT, Splatt3R, and VGGT Gaussians
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
try:
    import einops
except ImportError:
    einops = None  # Optional import

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.pixelsplat_src.cuda_splatting import render_cuda
except ImportError:
    render_cuda = None  # Optional import, may not be needed for all visualizations

try:
    from utils.geometry import build_covariance
    from utils.geometry import normalize_intrinsics
except ImportError:
    # Fallback: define build_covariance if import fails (only needed for Splatt3R PLY export)
    if einops is None:
        # If einops is not available, build_covariance will fail when called
        build_covariance = None
        normalize_intrinsics = None
    else:
        def build_covariance(scale, rotation_xyzw):
            '''Build the 3x3 covariance matrix from the three dimensional scale and the four dimension quaternion'''
            scale = scale.diag_embed()
            rotation = quaternion_to_matrix(rotation_xyzw)
            return (
                rotation
                @ scale
                @ einops.rearrange(scale, "... i j -> ... j i")
                @ einops.rearrange(rotation, "... i j -> ... j i")
            )
        
        def quaternion_to_matrix(quaternions, eps: float = 1e-8):
            '''Convert the 4-dimensional quaternions to 3x3 rotation matrices.'''
            i, j, k, r = torch.unbind(quaternions, dim=-1)
            two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)
            o = torch.stack(
                (
                    1 - two_s * (j * j + k * k),
                    two_s * (i * j - k * r),
                    two_s * (i * k + j * r),
                    two_s * (i * j + k * r),
                    1 - two_s * (i * i + k * k),
                    two_s * (j * k - i * r),
                    two_s * (i * k - j * r),
                    two_s * (j * k + i * r),
                    1 - two_s * (i * i + j * j),
                ),
                -1,
            )
            return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)
        
        normalize_intrinsics = None

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


def create_visualizations_splatt3r(preds, episode_idx, output_base_dir, steps_to_process=None, render_gaussians=False):
    """
    Comprehensive visualization function for Splatt3R outputs.
    
    Args:
        preds: List containing [pred1_list, pred2_list] from npz file
        episode_idx: Episode index
        output_base_dir: Base directory to save all visualizations
        steps_to_process: Optional set of step indices to process (None = process all)
        render_gaussians: If True, render Gaussians into 2D images using CUDA splatting (requires GPU)
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
        
        # Optional rendering of Gaussians into 2D images
        if render_gaussians:
            _render_splatt3r_single_step(pred1, episode_idx, step_idx, output_base_dir, view_name='view1')
            _render_splatt3r_single_step(pred2, episode_idx, step_idx, output_base_dir, view_name='view2')
        
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


def _render_splatt3r_single_step(pred, episode_idx, step_idx, output_base_dir, view_name='view'):
    """
    Render Splatt3R Gaussians into 2D images using CUDA splatting (if available).
    """
    if render_cuda is None:
        print("  WARNING: render_cuda not available; skipping Splatt3R Gaussian rendering.")
        return
    if normalize_intrinsics is None:
        print("  WARNING: normalize_intrinsics not available; skipping Splatt3R Gaussian rendering.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  WARNING: CUDA device not available; skipping Splatt3R Gaussian rendering.")
        return

    def extract(pred_dict, key, alt_key=None):
        k = key if key in pred_dict else alt_key
        if k is None or k not in pred_dict:
            return None
        arr = pred_dict[k]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = arr.item()
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        if isinstance(arr, torch.Tensor) and arr.ndim > 3:
            arr = arr[0]
        return arr

    means = extract(pred, 'means', 'means_in_other_view')
    covariances = extract(pred, 'covariances')
    if covariances is None:
        scales = extract(pred, 'scales')
        rotations = extract(pred, 'rotations')
        if scales is None or rotations is None:
            print("  WARNING: Missing scales/rotations for covariance; skipping Splatt3R rendering.")
            return
        if build_covariance is None:
            print("  WARNING: build_covariance unavailable; skipping Splatt3R rendering.")
            return
        covariances = build_covariance(scales.unsqueeze(0), rotations.unsqueeze(0))[0]

    sh = extract(pred, 'sh')
    opacities = extract(pred, 'opacities')
    if opacities is not None and opacities.ndim > 2:
        opacities = opacities.squeeze(-1)

    if any(x is None for x in [means, covariances, sh, opacities]):
        print("  WARNING: Missing fields for Splatt3R rendering; skipping.")
        return

    H, W = means.shape[:2]
    render_dir = os.path.join(output_base_dir, f'episode_{episode_idx:06d}', f'step_{step_idx:06d}', 'rendered_splatt3r')
    os.makedirs(render_dir, exist_ok=True)

    means_flat = means.reshape(-1, 3).to(device).float()
    cov_flat = covariances.reshape(-1, 3, 3).to(device).float()
    sh_flat = sh.reshape(-1, sh.shape[-2], sh.shape[-1]).to(device).float()
    opa_flat = opacities.reshape(-1).to(device).float()

    # Fallback camera: identity extrinsics, centered intrinsics
    extr = torch.eye(4, device=device).unsqueeze(0).float()
    intr = torch.zeros((1, 3, 3), device=device).float()
    fx = fy = max(H, W) / 2.0
    intr[..., 0, 0] = fx
    intr[..., 1, 1] = fy
    intr[..., 0, 2] = W / 2.0
    intr[..., 1, 2] = H / 2.0
    intr[..., 2, 2] = 1.0
    intr = normalize_intrinsics(intr, (H, W))

    bg = torch.tensor([1.0, 1.0, 1.0], device=device).unsqueeze(0)
    with torch.no_grad():
        color = render_cuda(
            extr,
            intr,
            torch.full((1,), 0.1, device=device),
            torch.full((1,), 1000.0, device=device),
            (H, W),
            bg,
            means_flat.unsqueeze(0),
            cov_flat.unsqueeze(0),
            sh_flat.unsqueeze(0),
            opa_flat.unsqueeze(0),
        )
    color = color[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(color)
    plt.axis('off')
    plt.tight_layout()
    out_path = os.path.join(render_dir, f'{view_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"  Rendered Splatt3R Gaussian {view_name} to {out_path}")


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


def create_visualizations_vggt_gaussians(vggt_preds_list, gaussian_preds_list, episode_idx, output_base_dir, steps_to_process=None, render_gaussians=False):
    """
    Comprehensive visualization function for VGGT Gaussians outputs.
    Combines VGGT visualizations with Gaussian parameter visualizations and PLY export.
    
    Args:
        vggt_preds_list: List of VGGT prediction dictionaries from npz file (one per step)
        gaussian_preds_list: List of Gaussian prediction dictionaries from npz file (one per step)
        episode_idx: Episode index
        output_base_dir: Base directory to save all visualizations
        steps_to_process: Optional set of step indices to process (None = process 3 evenly spaced)
        render_gaussians: If True, render 2D images from Gaussians using CUDA splatting (requires GPU)
    """
    print(f"\nEpisode {episode_idx}:")
    print(f"  Total steps: {len(vggt_preds_list)}")
    if len(vggt_preds_list) > 0:
        print(f"  Keys in vggt_preds_list[0]: {list(vggt_preds_list[0].keys())}")
        print(f"  Keys in gaussian_preds_list[0]: {list(gaussian_preds_list[0].keys())}")
    
    # Determine which steps to process
    if steps_to_process is None:
        num_steps = len(vggt_preds_list)
        steps_to_check = set(np.linspace(0, num_steps - 1, 3, dtype=int).tolist())
    else:
        steps_to_check = steps_to_process
    
    steps_to_check = sorted(steps_to_check) if isinstance(steps_to_check, set) else sorted(list(steps_to_check))
    
    for step_idx in steps_to_check:
        if step_idx >= len(vggt_preds_list):
            print(f"  Step {step_idx}: OUT OF RANGE")
            continue
        
        vggt_pred = vggt_preds_list[step_idx]
        gaussian_pred = gaussian_preds_list[step_idx]
        
        # Unwrap if needed
        if isinstance(vggt_pred, np.ndarray) and vggt_pred.dtype == object:
            vggt_pred = vggt_pred.item()
        if isinstance(gaussian_pred, np.ndarray) and gaussian_pred.dtype == object:
            gaussian_pred = gaussian_pred.item()
        
        if not isinstance(vggt_pred, dict) or not isinstance(gaussian_pred, dict):
            print(f"  Step {step_idx}: Not a dict, skipping")
            continue
        
        print(f"  Processing step {step_idx}...")
        
        # Visualize VGGT outputs
        _visualize_vggt_single_step(vggt_pred, episode_idx, step_idx, output_base_dir)
        
        # Visualize Gaussian outputs
        _visualize_vggt_gaussians_single_step(vggt_pred, gaussian_pred, episode_idx, step_idx, output_base_dir)
        
        # Optionally render Gaussians into 2D images
        if render_gaussians:
            _render_vggt_gaussians_single_step(gaussian_pred, episode_idx, step_idx, output_base_dir)
        
        # Save PLY file from Gaussian parameters
        ply_output_dir = os.path.join(output_base_dir, f'episode_{episode_idx:06d}', 'combined_ply')
        os.makedirs(ply_output_dir, exist_ok=True)
        ply_path = os.path.join(ply_output_dir, f'step_{step_idx:06d}.ply')
        save_vggt_gaussians_ply_from_npz(gaussian_pred, ply_path)
    
    print(f"  All visualizations saved to {output_base_dir}")


def _visualize_vggt_gaussians_single_step(vggt_pred, gaussian_pred, episode_idx, step_idx, output_base_dir):
    """Visualize Gaussian parameters for a single step"""
    
    # Create directory structure
    step_dir = os.path.join(output_base_dir, f'episode_{episode_idx:06d}', f'step_{step_idx:06d}', 'gaussians')
    os.makedirs(step_dir, exist_ok=True)
    
    # Extract Gaussian arrays (remove batch dimension if present)
    # Shape is [B, S, H, W, ...] where S is sequence length (number of images)
    def extract_array(pred, key):
        """Extract array and remove batch dimension"""
        if key not in pred:
            return None
        arr = pred[key]
        if arr.ndim == 5:  # [B, S, H, W, ...]
            arr = arr[0]  # Remove batch dim: [S, H, W, ...]
        elif arr.ndim == 6:  # [B, S, H, W, ..., ...] (e.g., sh: [B, S, H, W, 3, sh_degree])
            arr = arr[0]  # Remove batch dim: [S, H, W, ..., ...]
        return arr
    
    means = extract_array(gaussian_pred, 'means')
    scales = extract_array(gaussian_pred, 'scales')
    rotations = extract_array(gaussian_pred, 'rotations')
    sh = extract_array(gaussian_pred, 'sh')
    opacities = extract_array(gaussian_pred, 'opacities')
    pts3d = extract_array(gaussian_pred, 'pts3d')
    offsets = extract_array(gaussian_pred, 'offsets')
    
    # Get sequence length
    S = means.shape[0] if means is not None else 1
    
    # Visualize each image in the sequence
    for img_idx in range(S):
        img_dir = os.path.join(step_dir, f'image_{img_idx:02d}')
        os.makedirs(img_dir, exist_ok=True)
        
        # 1. Gaussian means (3D centers)
        if means is not None:
            fig = plt.figure(figsize=(15, 5))
            means_img = means[img_idx]  # [H, W, 3]
            pts_flat = means_img.reshape(-1, 3)
            
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
            ax1.set_title(f'Gaussian Means 3D {img_idx}')
            ax1.view_init(elev=20, azim=45)
            
            # XY projection
            ax2 = fig.add_subplot(132)
            ax2.scatter(pts_flat[:, 0], pts_flat[:, 1], s=0.1, alpha=0.5)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title(f'Gaussian Means XY {img_idx}')
            ax2.set_aspect('equal')
            
            # XZ projection
            ax3 = fig.add_subplot(133)
            ax3.scatter(pts_flat[:, 0], pts_flat[:, 2], s=0.1, alpha=0.5)
            ax3.set_xlabel('X')
            ax3.set_ylabel('Z')
            ax3.set_title(f'Gaussian Means XZ {img_idx}')
            ax3.set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'gaussian_means.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 2. Scales visualization
        if scales is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            scales_img = scales[img_idx]  # [H, W, 3]
            
            for i, ax in enumerate(axes):
                scale_comp = scales_img[:, :, i]
                im = ax.imshow(scale_comp, cmap='viridis', interpolation='nearest')
                ax.set_title(f'Scale Component {i} ({"XYZ"[i]})')
                plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'scales.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. Opacities visualization
        if opacities is not None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            opacities_img = opacities[img_idx]
            if opacities_img.ndim == 3 and opacities_img.shape[-1] == 1:
                opacities_img = opacities_img.squeeze(-1)  # [H, W]
            
            im = ax.imshow(opacities_img, cmap='hot', interpolation='nearest')
            ax.set_title(f'Opacities {img_idx}')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'opacities.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. Spherical Harmonics (RGB from first 3 components)
        if sh is not None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            sh_img = sh[img_idx]  # Should be [H, W, 3, sh_degree] after extraction
            
            # Handle different possible shapes
            if sh_img.ndim == 4:  # [H, W, 3, sh_degree]
                # Extract first 3 components (RGB) from DC (index 0)
                sh_rgb = sh_img[:, :, :, 0]  # [H, W, 3]
            elif sh_img.ndim == 3:  # [H, W, 3] (already extracted)
                sh_rgb = sh_img
            else:
                print(f"WARNING: Unexpected sh shape {sh_img.shape}, skipping visualization")
                plt.close()
                continue
            
            # Normalize to [0, 1] for display
            sh_rgb = (sh_rgb - sh_rgb.min()) / (sh_rgb.max() - sh_rgb.min() + 1e-8)
            sh_rgb = np.clip(sh_rgb, 0, 1)
            
            ax.imshow(sh_rgb)
            ax.set_title(f'Spherical Harmonics RGB (DC) {img_idx}')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'spherical_harmonics_rgb.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 5. Rotations visualization (quaternion magnitude)
        if rotations is not None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            rotations_img = rotations[img_idx]  # [H, W, 4]
            
            # Compute quaternion magnitude (should be ~1 for valid rotations)
            rot_mag = np.linalg.norm(rotations_img, axis=-1)
            
            im = ax.imshow(rot_mag, cmap='viridis', interpolation='nearest')
            ax.set_title(f'Rotation Quaternion Magnitude {img_idx}')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'rotations_magnitude.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 6. Offsets visualization (if available)
        if offsets is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            offsets_img = offsets[img_idx]  # [H, W, 3]
            
            for i, ax in enumerate(axes):
                offset_comp = offsets_img[:, :, i]
                # Use symmetric normalization around 0 for diverging colormap
                vmax = max(abs(offset_comp.max()), abs(offset_comp.min()))
                vmin = -vmax
                im = ax.imshow(offset_comp, cmap='RdBu', interpolation='nearest', vmin=vmin, vmax=vmax)
                ax.set_title(f'Offset Component {i} ({"XYZ"[i]})')
                plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            plt.savefig(os.path.join(img_dir, 'offsets.png'), dpi=150, bbox_inches='tight')
            plt.close()


def _render_vggt_gaussians_single_step(gaussian_pred, episode_idx, step_idx, output_base_dir, near=0.1, far=1000.0):
    """
    Render predicted Gaussians into 2D images using CUDA splatting (if available).
    """
    if render_cuda is None:
        print("  WARNING: render_cuda not available; skipping Gaussian rendering.")
        return
    if normalize_intrinsics is None:
        print("  WARNING: normalize_intrinsics not available; skipping Gaussian rendering.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  WARNING: CUDA device not available; skipping Gaussian rendering.")
        return

    def extract_array(pred, key):
        if key not in pred:
            return None
        arr = pred[key]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = arr.item()
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        elif isinstance(arr, torch.Tensor):
            arr = arr
        else:
            return None
        if arr.ndim >= 1 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    means = extract_array(gaussian_pred, "means")
    covariances = extract_array(gaussian_pred, "covariances")
    sh = extract_array(gaussian_pred, "sh")
    opacities = extract_array(gaussian_pred, "opacities")
    cam_extr = extract_array(gaussian_pred, "camera_extrinsics")
    cam_intr = extract_array(gaussian_pred, "camera_intrinsics")

    if any(x is None for x in [means, covariances, sh, opacities, cam_extr, cam_intr]):
        print("  WARNING: Missing fields for rendering; skipping Gaussian rendering.")
        return

    S, H, W = means.shape[:3]
    render_dir = os.path.join(output_base_dir, f'episode_{episode_idx:06d}', f'step_{step_idx:06d}', 'rendered')
    os.makedirs(render_dir, exist_ok=True)

    for img_idx in range(S):
        means_view = means[img_idx].reshape(-1, 3).to(device).float()
        cov_view = covariances[img_idx].reshape(-1, 3, 3).to(device).float()
        sh_view = sh[img_idx].reshape(-1, sh.shape[-2], sh.shape[-1]).to(device).float()
        op_view = opacities[img_idx].reshape(-1).to(device).float()

        extr_view = cam_extr[img_idx].to(device).float()
        intr_view = cam_intr[img_idx].to(device).float()
        intr_view = normalize_intrinsics(intr_view[None, ...], (H, W))[0]

        bg = torch.tensor([1.0, 1.0, 1.0], device=device).unsqueeze(0)
        with torch.no_grad():
            color = render_cuda(
                extr_view.unsqueeze(0),
                intr_view.unsqueeze(0),
                torch.full((1,), near, device=device),
                torch.full((1,), far, device=device),
                (H, W),
                bg,
                means_view.unsqueeze(0),
                cov_view.unsqueeze(0),
                sh_view.unsqueeze(0),
                op_view.unsqueeze(0),
            )
        color = color[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(color)
        plt.axis('off')
        plt.tight_layout()
        out_path = os.path.join(render_dir, f'view_{img_idx:02d}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"  Rendered Gaussian view {img_idx} to {out_path}")


def save_vggt_gaussians_ply_from_npz(gaussian_pred, save_path):
    """
    Save VGGT Gaussians as PLY file from numpy predictions.
    Works with outputs loaded from npz files.
    Combines all images in the sequence into a single PLY file.
    """
    def extract_array(pred, key):
        """Extract array from pred, handling batch dimensions"""
        if key not in pred:
            return None
        
        arr = pred[key]
        # Remove batch dimension if present: [B, S, H, W, ...] -> [S, H, W, ...]
        if arr.ndim == 5:
            arr = arr[0]  # [S, H, W, ...]
        elif arr.ndim == 6:  # [B, S, H, W, ..., ...] (e.g., sh: [B, S, H, W, 3, sh_degree])
            arr = arr[0]  # [S, H, W, ..., ...]
        return arr
    
    # Extract means
    means = extract_array(gaussian_pred, 'means')
    if means is None:
        print(f"  WARNING: No means found in gaussian_pred, skipping PLY export")
        return
    
    # Extract scales and rotations
    scales = extract_array(gaussian_pred, 'scales')
    rotations = extract_array(gaussian_pred, 'rotations')  # quaternions
    
    # Extract opacities
    opacities = extract_array(gaussian_pred, 'opacities')
    if opacities is not None and opacities.ndim > 2:
        if opacities.shape[-1] == 1:
            opacities = opacities.squeeze(-1)  # [S, H, W]
    
    # Extract spherical harmonics
    sh = extract_array(gaussian_pred, 'sh')
    if sh is not None and sh.ndim > 3:
        # sh is [S, H, W, 3, sh_degree], extract first component (DC)
        sh = sh[:, :, :, :, 0]  # [S, H, W, 3]
    
    # Get sequence length and spatial dimensions
    S, H, W = means.shape[:3]
    
    # Combine all images in sequence
    means_list = []
    scales_list = []
    rotations_list = []
    sh_list = []
    opacities_list = []
    
    for img_idx in range(S):
        means_img = means[img_idx]  # [H, W, 3]
        means_flat = means_img.reshape(-1, 3)  # (H*W, 3)
        means_list.append(means_flat)
        
        if scales is not None:
            scales_img = scales[img_idx]  # [H, W, 3]
            scales_flat = scales_img.reshape(-1, 3)  # (H*W, 3)
            scales_list.append(scales_flat)
        
        if rotations is not None:
            rotations_img = rotations[img_idx]  # [H, W, 4]
            rotations_flat = rotations_img.reshape(-1, 4)  # (H*W, 4)
            rotations_list.append(rotations_flat)
        
        if sh is not None:
            sh_img = sh[img_idx]  # [H, W, 3]
            sh_flat = sh_img.reshape(-1, 3)  # (H*W, 3)
            sh_list.append(sh_flat)
        
        if opacities is not None:
            opacities_img = opacities[img_idx]  # [H, W]
            opacities_flat = opacities_img.reshape(-1)  # (H*W,)
            opacities_list.append(opacities_flat)
    
    # Concatenate all images
    means_combined = np.concatenate(means_list, axis=0)  # (S*H*W, 3)
    
    if scales_list:
        scales_combined = np.concatenate(scales_list, axis=0)  # (S*H*W, 3)
    else:
        scales_combined = np.ones((means_combined.shape[0], 3)) * 0.01  # Default small scale
    
    if rotations_list:
        rotations_combined = np.concatenate(rotations_list, axis=0)  # (S*H*W, 4)
    else:
        # Default identity quaternion [0, 0, 0, 1]
        rotations_combined = np.zeros((means_combined.shape[0], 4))
        rotations_combined[:, 3] = 1.0
    
    if sh_list:
        sh_combined = np.concatenate(sh_list, axis=0)  # (S*H*W, 3)
    else:
        sh_combined = np.ones((means_combined.shape[0], 3)) * 0.5  # Default gray
    
    if opacities_list:
        opacities_combined = np.concatenate(opacities_list, axis=0)  # (S*H*W,)
    else:
        opacities_combined = np.ones(means_combined.shape[0]) * 0.5  # Default opacity
    
    # Normalize quaternions
    quat_norms = np.linalg.norm(rotations_combined, axis=1, keepdims=True)
    quat_norms = np.clip(quat_norms, 1e-8, None)
    rotations_normalized = rotations_combined / quat_norms
    
    # Use first 3 components of SH as RGB (f_dc)
    # Note: reg_dense_sh just reshapes, doesn't convert. SH coefficients are used directly
    # matching Splatt3R's approach (they don't use SH2RGB conversion in PLY export)
    sh_rgb = sh_combined[:, :3] if sh_combined.shape[1] >= 3 else sh_combined
    
    # Construct PLY attributes
    num_gaussians = means_combined.shape[0]
    rest_sh = np.zeros((num_gaussians, 0))  # No rest harmonics for now
    normals = np.zeros((num_gaussians, 3))  # Placeholder normals
    
    # CRITICAL: reg_dense_scales returns scales.exp(), so scales are already in exponential space
    # We need to take log to get back to log-space for PLY format (matching Splatt3R)
    # But wait - let me check: in utils/export.py line 115, they do np.log(scales) where scales come from SVD
    # In our case, scales come from reg_dense_scales which does .exp(), so we need .log() to undo it
    # Actually, the PLY format expects log(scales), and our scales are already exp'd, so we log them
    scales_log = np.log(np.clip(scales_combined, 1e-8, None))
    
    attributes = np.concatenate([
        means_combined,           # x, y, z (3)
        normals,                   # nx, ny, nz (3)
        sh_rgb,                    # f_dc_0, f_dc_1, f_dc_2 (3)
        rest_sh,                   # f_rest (0 for now)
        opacities_combined[:, None],  # opacity (1)
        scales_log,                # log(scale_0), log(scale_1), log(scale_2) (3)
        rotations_normalized       # rot_0, rot_1, rot_2, rot_3 (4)
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
    print(f"  Saved PLY file to {save_path} ({num_gaussians} Gaussians)")
