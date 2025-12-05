"""
Training script for VGGTGaussians model.
Mirrors Splatt3R's run_experiment() function.
"""

import os
import sys
import json

import lightning as L
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import workspace utilities (for config loading)
try:
    import workspace
except ImportError:
    workspace = None
    print("Warning: workspace module not found. Will use Config dataclass directly.")

# Import model and config
from bora_code.vggt_gaussians.main_cursor import VGGTGaussians, Config

# Import dataset utilities
try:
    import data.scannetpp.scannetpp as scannetpp
except ImportError:
    scannetpp = None
    print("Warning: scannetpp dataset module not found. You'll need to provide your own dataset.")


def run_experiment_vggt_gaussians(config):
    """
    Run training experiment for VGGTGaussians, mirroring Splatt3R's run_experiment().
    
    Args:
        config: Either an OmegaConf config (from YAML) or a Config dataclass
    """
    # Set the seed
    seed = getattr(config, 'seed', 0)
    L.seed_everything(seed, workers=True)

    # Set up loggers
    save_dir = getattr(config, 'save_dir', '/media/bora/Extreme Pro/new_proj')
    name = getattr(config, 'name', 'vggt_gaussians')
    os.makedirs(os.path.join(save_dir, name), exist_ok=True)
    
    loggers = []
    
    # CSV logger
    use_csv = getattr(config, 'loggers', {}).get('use_csv_logger', True) if hasattr(config, 'loggers') else True
    if use_csv:
        csv_logger = L.pytorch.loggers.CSVLogger(
            save_dir=save_dir,
            name=name
        )
        loggers.append(csv_logger)
    
    # WandB logger (optional - disabled by default for training-only focus)
    use_wandb = getattr(config, 'loggers', {}).get('use_wandb', False) if hasattr(config, 'loggers') else False
    if use_wandb:
        try:
            import wandb
            import omegaconf
            # Set WandB to offline mode to avoid interactive prompts
            os.environ['WANDB_MODE'] = 'disabled'  # Disable WandB to avoid interactive prompts
            print("WandB is disabled. Set use_wandb: True and configure WandB separately if needed.")
        except ImportError:
            print("Warning: WandB not available, skipping WandB logger")
    else:
        # Explicitly disable WandB to avoid any prompts
        os.environ['WANDB_MODE'] = 'disabled'

    # Set up profiler (optional)
    use_profiler = getattr(config, 'use_profiler', False)
    if use_profiler:
        profiler = L.pytorch.profilers.PyTorchProfiler(
            dirpath=save_dir,
            filename='trace',
            export_to_chrome=True,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(save_dir),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            profile_memory=True,
            with_stack=True
        )
    else:
        profiler = None

    # Model
    print('Loading VGGTGaussians model...')
    model = VGGTGaussians(config)
    
    # Optionally load pretrained checkpoint
    if hasattr(config, 'use_pretrained') and config.use_pretrained:
        if hasattr(config, 'pretrained_path') and config.pretrained_path:
            print(f'Loading pretrained checkpoint from {config.pretrained_path}')
            checkpoint = torch.load(config.pretrained_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            del checkpoint

    # Training Datasets
    print('Building Datasets')
    data_root = getattr(config, 'data', {}).get('root', None) if hasattr(config, 'data') else None
    resolution_raw = getattr(config, 'data', {}).get('resolution', 518) if hasattr(config, 'data') else 518
    
    # Convert resolution to tuple (H, W) - handle both int and tuple/list inputs
    if isinstance(resolution_raw, (int, float)):
        resolution = (int(resolution_raw), int(resolution_raw))
    elif isinstance(resolution_raw, (list, tuple)):
        resolution = tuple(int(x) for x in resolution_raw)
    else:
        resolution = (518, 518)  # fallback
    
    if scannetpp is not None and data_root is not None:
        try:
            train_dataset = scannetpp.get_scannet_dataset(
                data_root,
                'train',
                resolution,
                num_epochs_per_epoch=getattr(config, 'data', {}).get('epochs_per_train_epoch', 1) if hasattr(config, 'data') else 1,
            )
            
            # Check if we have any sequences to train on
            if len(train_dataset) == 0:
                print(f"ERROR: No training sequences found in dataset at {data_root}")
                print("Please check that:")
                print(f"  1. The path {data_root} is correct")
                print(f"  2. The splits file exists: {os.path.join(data_root, 'splits', 'nvs_sem_train.txt')}")
                print(f"  3. At least some sequence folders exist in {os.path.join(data_root, 'data')}")
                return
            
            print(f"Training dataset loaded: {len(train_dataset)} samples")
            
            # Get batch_size from config.data.batch_size (matching Splatt3R pattern)
            batch_size = getattr(config, 'data', {}).get('batch_size', 1) if hasattr(config, 'data') else getattr(config, 'batch_size', 1)
            
            data_loader_train = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=getattr(config, 'data', {}).get('num_workers', 4) if hasattr(config, 'data') else 4,
                pin_memory=True,
            )

            # Validation dataset - optional, skip if it fails
            data_loader_val = None
            try:
                val_dataset = scannetpp.get_scannet_test_dataset(
                    data_root,
                    alpha=0.5,
                    beta=0.5,
                    resolution=resolution,
                    use_every_n_sample=100,
                )
                data_loader_val = torch.utils.data.DataLoader(
                    val_dataset,
                    shuffle=False,
                    batch_size=batch_size,
                    num_workers=getattr(config, 'data', {}).get('num_workers', 4) if hasattr(config, 'data') else 4,
                    pin_memory=True,
                )
                print("Validation dataset loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load validation dataset: {e}")
                print("Continuing with training only (no validation)")
                data_loader_val = None
        except Exception as e:
            print(f"Error loading ScanNet++ dataset: {e}")
            print("You'll need to provide your own dataset or fix the data paths.")
            return
    else:
        print("ScanNet++ dataset not available. You'll need to provide your own dataset.")
        print("Create DataLoaders with batches matching Splatt3R format:")
        print("  batch['context']: list of dicts with 'img' and 'original_img' keys")
        print("  batch['target']: optional list of target view dicts")
        return

    # Training
    print('Training')
    
    # Get devices config
    devices_config = getattr(config, 'devices', 'auto')
    if isinstance(devices_config, str) and devices_config == 'auto':
        devices = 'auto'
    elif isinstance(devices_config, (list, tuple)):
        devices = devices_config
    else:
        devices = 1
    
    # Set up callbacks
    callbacks = [
        L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
    ]
    
    # Add model checkpoint callback
    # Use train/loss if validation is not available, otherwise use val/loss
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(save_dir, name, "checkpoints"),
        filename="vggt_gaussians-{epoch:02d}-{train_loss:.2f}",
        monitor="train/loss",  # Monitor training loss instead of validation
        save_top_k=3,
        mode='min',
        save_last=True,  # Also save the last checkpoint
        every_n_epochs=1,  # Save checkpoint every epoch
    )
    callbacks.append(checkpoint_callback)
    
    trainer = L.Trainer(
        accelerator="gpu",
        benchmark=True,
        callbacks=callbacks,
        check_val_every_n_epoch=1 if data_loader_val is not None else None,  # Skip validation if dataset not available
        default_root_dir=save_dir,
        devices=devices,
        gradient_clip_val=config.opt.get('gradient_clip_val', 0.5) if hasattr(config.opt, 'get') else getattr(config.opt, 'gradient_clip_val', 0.5),
        log_every_n_steps=10,
        logger=loggers,
        max_epochs=config.opt.get('epochs', 20) if hasattr(config.opt, 'get') else getattr(config.opt, 'epochs', 20),
        profiler=profiler,
        strategy="ddp_find_unused_parameters_true" if (isinstance(devices, (list, tuple)) and len(devices) > 1) or (isinstance(devices, int) and devices > 1) else "auto",
    )
    
    # Train the model
    # If validation dataset is None, Lightning will skip validation
    trainer.fit(model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val)

    # Testing (optional, simplified)
    print('Training complete!')
    print(f'Checkpoints saved to: {os.path.join(save_dir, "checkpoints")}')
    
    return model, trainer


if __name__ == "__main__":
    # Option 1: Use workspace.load_config() like Splatt3R (preferred)
    if workspace is not None and len(sys.argv) > 1:
        config_path = sys.argv[1]
        command_line_args = sys.argv[2:] if len(sys.argv) > 2 else []
        config = workspace.load_config(config_path, command_line_args)
        if os.getenv("LOCAL_RANK", '0') == '0':
            config = workspace.create_workspace(config)
        run_experiment_vggt_gaussians(config)
    
    # Option 2: Create Config dataclass directly (fallback for testing)
    else:
        print("Using default Config dataclass (no YAML config provided)")
        print("Usage: python train_cursor.py <config.yaml> [override_args]")
        print("Or modify this script to use a Config() instance directly")
        
        # Example: Create a simple config for testing
        # config = Config(
        #     batch_size=1,
        #     opt={'lr': 0.0001, 'epochs': 1, 'gradient_clip_val': 0.5},
        #     loss={'mse_loss_weight': 1.0, 'lpips_loss_weight': 0.25, 'apply_mask': True, 'average_over_mask': True},
        #     sh_degree=1,
        #     use_offsets=True,
        # )
        # run_experiment_vggt_gaussians(config)

