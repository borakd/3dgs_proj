## 1. Splatt3R Setup

1. Clone /github.com/btsmart/splatt3r
2. Create a virtual environment with uv: `uv venv .venv --python 3.11.12`
3. Activate the venv: `source .venv/bin.activate`
3. Install torch and torchvision: `uv pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121`
4. Install requirements: `uv pip install -r requirements_uv.txt`

## 2. VGGT Setup

1. Deactivate Splatt3R venv (if active): `deactivate`
2. Navigate to desired subdirectory: `cd src/`
3. Clone git@github.com:facebookresearch/vggt.git
4. Navigate into it: `cd vggt/`
5. Create another virtual environment with uv: `uv venv .venv --python 3.11.12`
6. Activate the venv: `source .venv/bin.activate`
7. Install torch and torchvision: `uv pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121`
8. Install requirements: `uv pip install -r requirements_uv.txt`

## 3. MASt3R Setup

1. Just activate the VGGT venv and install scikit-learn. It is the only requirement of MASt3R not already covered by VGGT, so we can simply install it in the VGGT venv: `uv pip install scikit-learn`

## 4. (Optional) Modify line 116 of .venv/bin/activate (same for both venvs) for clarity while venvs are active

Replace: `PS1="(${VIRTUAL_ENV_PROMPT}) ${PS1-}"`

With: `PS1="(desired_venv_name) ${PS1:-}"`

## 5. Downloading the DROID dataset (I currently use the 100 episode version)

Download link (may require you to `uv pip install gsutil`): `gsutil -m cp -r gs://gresearch/robotics/droid_100 $GWM_PATH/datasets`

Put the dataset in the root directory as `/datasets/droid_100/1.0.0/`

## 6. Downloading the Splatt3R checkpoint

Download link: https://huggingface.co/brandonsmart/splatt3r_v1.0/resolve/main/epoch%3D19-step%3D1200.ckpt

Put it in the root directory as well

## 7. Try running demos for Splatt3R, VGGT, and MASt3R

Splatt3R demo: `cd` root directory, activate Splatt3R venv, and run `python demo.py`

VGGT demo: `cd src/vggt/`, activate VGGT venv, and run `python demo.py`

MASt3R demo (can run with VGGT venv): `cd src/mast3r_src/` and run `python demo.py --model_name MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`

Confirm that all 3 demos work.

## TODO: VGGTGaussians setup
