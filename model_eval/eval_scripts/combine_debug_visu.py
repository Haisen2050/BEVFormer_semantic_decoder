#!/usr/bin/env python3
# Combine GT + 6 camera images + predictions from all model folders (simple, uses frame_ids.txt)

import argparse, math
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CAM_ORDER = [
    'CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT',
]
EXTS = ('.png', '.jpg', '.jpeg')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--visu-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--frame-ids', default=None, help='Defaults to <visu-dir>/frame_ids.txt')
    p.add_argument('--cols', type=int, default=4)
    return p.parse_args()

def any_ext(stem: Path):
    for e in EXTS:
        p = stem.with_suffix(e)
        if p.exists(): return p
    return None

def open_rgb(p: Path):
    return Image.open(p).convert('RGB')

def main():
    a = parse_args()
    visu = Path(a.visu_dir)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    # model folders = every subdir except CAM_* (and common camera dir names)
    skip = set(CAM_ORDER) | {'cams','cameras','6cams','camera_images'}
    model_dirs = sorted([d for d in visu.iterdir() if d.is_dir() and d.name not in skip])
    if not model_dirs:
        raise SystemExit(f'No model subfolders in {visu}')

    # frame ids from file (no dependency on scanning GT)
    fid_path = Path(a.frame_ids) if a.frame_ids else (visu/'frame_ids.txt')
    if not fid_path.exists():
        # tolerate common typo "frames_ids.txt"
        alt = visu/'frames_ids.txt'
        if alt.exists(): fid_path = alt
        else: raise SystemExit(f'frame IDs file not found: {fid_path}')
    frame_ids = [ln.strip() for ln in fid_path.read_text().splitlines() if ln.strip()]

    # cameras available?
    cams_present = all((visu/c).is_dir() for c in CAM_ORDER)

    for fid in frame_ids:
        tiles = []

        # 6 cameras
        if cams_present:
            for cam in CAM_ORDER:
                tiles.append((cam, any_ext(visu/cam/fid)))

        # GT: look in first model folder with {fid}_gt.*
        gt = (any_ext(model_dirs[0]/f'{fid}_gt')
              or any_ext(model_dirs[0]/f'{fid}__gt')
              or any_ext(model_dirs[0]/f'{fid}-gt')
              or any_ext(model_dirs[0]/f'{fid}_-gt'))
        tiles.append(('GT', gt))

        # predictions from each model
        for md in model_dirs:
            tiles.append((md.name, any_ext(md/fid)))

        # grid
        n = len(tiles)
        cols = max(1, a.cols)
        rows = math.ceil(n/cols)

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for ax, (title, path) in zip(axes, tiles):
            if path and path.exists():
                ax.imshow(open_rgb(path))
            else:
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center')
            ax.set_title(title)
            ax.axis('off')

        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(out/f'{fid}.png', dpi=100)
        plt.close(fig)

    print(f'Done. Saved {len(frame_ids)} images to {out}')

if __name__ == '__main__':
    main()