#!/usr/bin/env python3

# python model_eval/eval_scripts/collect_cams_from_pred_tokens.py   --pred-dir model_eval/eval_results/model_debug_visu/FPN5Res18     --nusc-root /data/nuscenes     --visu-dir model_eval/eval_results/model_debug_visu     --mode hardlink

import argparse, json, os, re, shutil, sys
from pathlib import Path
from typing import Dict, List

CAM_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
]
HEX32_RE = re.compile(r'([0-9a-fA-F]{32})')
IMG_EXTS = {'.png', '.jpg', '.jpeg'}

def parse_args():
    ap = argparse.ArgumentParser(
        description='Collect the 6 NuScenes camera images for predictions named by sample token.')
    ap.add_argument('--pred-dir', required=True,
                    help='Folder with prediction images; filenames contain a 32-hex token (sample or sample_data).')
    ap.add_argument('--nusc-root', required=True,
                    help='NuScenes root (contains v1.0-*)')
    ap.add_argument('--visu-dir', required=True,
                    help='Destination root; will create CAM_* folders with {token}.jpg')
    ap.add_argument('--mode', choices=['hardlink','copy','symlink'], default='hardlink',
                    help='How to materialize images (default: hardlink)')
    ap.add_argument('--exts', default='.png,.jpg,.jpeg',
                    help='Prediction file extensions to scan (comma-separated)')
    return ap.parse_args()

def materialize(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists(): return
    try:
        if mode == 'hardlink': os.link(src, dst)
        elif mode == 'symlink': os.symlink(src, dst)
        else: shutil.copy2(src, dst)
    except OSError:
        shutil.copy2(src, dst)

def find_versions(nusc_root: Path) -> List[Path]:
    vers = []
    for ver in ('v1.0-trainval', 'v1.0-test', 'v1.0-mini'):
        base = nusc_root / ver
        if (base/'sample.json').exists() and (base/'sample_data.json').exists() \
           and (base/'calibrated_sensor.json').exists() and (base/'sensor.json').exists():
            vers.append(base)
    if not vers:
        raise FileNotFoundError(f'No complete v1.0-* under {nusc_root}')
    return vers

def load_json(p: Path):
    with open(p, 'r') as f: return json.load(f)

def build_maps(nusc_root: Path):
    """
    Build:
      sample_token -> {CAM_*: relative filename}   (key frame preferred)
      sd_token     -> sample_token
    """
    sample_to_camfiles: Dict[str, Dict[str, str]] = {}
    sd_to_sample: Dict[str, str] = {}

    # sensor joins (shared across versions)
    for base in find_versions(nusc_root):
        cs   = load_json(base/'calibrated_sensor.json')
        sens = load_json(base/'sensor.json')
        sdata= load_json(base/'sample_data.json')

        cs2sensor = {r['token']: r['sensor_token'] for r in cs}
        sensor2ch = {r['token']: r['channel']      for r in sens}
        sensor2md = {r['token']: r['modality']     for r in sens}

        for r in sdata:
            sd_tok = r.get('token'); smp_tok = r.get('sample_token')
            if not (isinstance(sd_tok, str) and isinstance(smp_tok, str)):
                continue
            sd_tok_l = sd_tok.lower(); smp_tok_l = smp_tok.lower()
            sd_to_sample.setdefault(sd_tok_l, smp_tok_l)

            # channel + modality via joins
            st = cs2sensor.get(r.get('calibrated_sensor_token'))
            ch = sensor2ch.get(st, '')
            md = sensor2md.get(st, '')
            if md != 'camera' and not str(ch).startswith('CAM_'):
                continue

            rel = r.get('filename') or r.get('file_name')
            if not isinstance(rel, str):
                continue

            d = sample_to_camfiles.setdefault(smp_tok_l, {})
            # prefer key frame if conflicts
            if ch not in d or (r.get('is_key_frame', False)):
                d[ch] = rel

    return sample_to_camfiles, sd_to_sample

def extract_tokens_from_dir(pred_dir: Path, allowed_exts: set) -> List[str]:
    toks, seen = [], set()
    for p in sorted(pred_dir.iterdir()):
        if not p.is_file(): continue
        if p.suffix.lower() not in allowed_exts: continue
        m = HEX32_RE.search(p.name)
        if not m: continue
        t = m.group(1).lower()
        if t not in seen:
            seen.add(t); toks.append(t)
    return toks

def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    nusc_root = Path(args.nusc_root)
    visu_dir = Path(args.visu_dir)
    if not pred_dir.is_dir(): sys.exit(f'[ERR] pred-dir not a directory: {pred_dir}')
    if not nusc_root.is_dir(): sys.exit(f'[ERR] nusc-root not a directory: {nusc_root}')

    allowed_exts = {e.strip().lower() for e in args.exts.split(',') if e.strip()} or IMG_EXTS

    print('[i] Building NuScenes camera lookup (this reads sample_data/calibrated_sensor/sensor)...')
    sample_to_camfiles, sd_to_sample = build_maps(nusc_root)

    tokens = extract_tokens_from_dir(pred_dir, allowed_exts)
    if not tokens:
        sys.exit(f'[ERR] no 32-hex tokens found in {pred_dir}')

    per_cam_found = {c: 0 for c in CAM_ORDER}
    per_cam_miss  = {c: 0 for c in CAM_ORDER}
    miss_tokens = 0

    for tok in tokens:
        # accept either sample token or sample_data token in filenames
        sample_tok = tok if tok in sample_to_camfiles else sd_to_sample.get(tok)
        if not sample_tok:
            miss_tokens += 1
            continue

        camdict = sample_to_camfiles.get(sample_tok, {})
        for cam in CAM_ORDER:
            rel = camdict.get(cam)
            if not rel:
                per_cam_miss[cam] += 1
                continue
            src = (nusc_root / rel).resolve()
            if not src.exists():
                per_cam_miss[cam] += 1
                continue
            ext = src.suffix.lower() or '.jpg'
            dst = visu_dir / cam / f'{sample_tok}{ext}'
            materialize(src, dst, args.mode)
            per_cam_found[cam] += 1

    print('\n=== Summary ===')
    print(f'Prediction tokens scanned: {len(tokens)}')
    print(f'Tokens not resolvable to a sample: {miss_tokens}')
    print(f'Wrote to: {visu_dir} (folders: {", ".join(CAM_ORDER)})')
    for cam in CAM_ORDER:
        print(f'{cam:>16}: found {per_cam_found[cam]:6d} | missing {per_cam_miss[cam]:6d}')

if __name__ == '__main__':
    main()