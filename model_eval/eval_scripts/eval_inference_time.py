#!/usr/bin/env python
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Created by ChatGPT
# ---------------------------------------------
"""
Usage:
    python model_eval/eval_scripts/eval_inference_time.py \
      --config projects/configs/bevformer/bevformer_base_seg_det_150x150.py \
      --decoder-dir work_dirs/bevformer_base_seg_det_150x150 \
      --warmup 10 \
      --batch-size 1 \
      --workers-per-gpu 2 \
      --iters 150 \
      --device cuda:0

Measures average inference latency per batch for each
SegDecoder_<Name>_<Date>.pth in the given directory (full pipeline only).
Includes debug prints to trace execution steps.
Builds the dataloader once, outside the decoder loop.
Outputs all terminal output to model_eval/eval_results/model_inference_speed_output.txt
and saves a bar plot of average latency (descending order) to model_eval/eval_results/model_inference_speed.png
"""

import argparse
from pyexpat import model
import time
import torch
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel, scatter
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
import numpy as np

# Setup logging of stdout and stderr to file
output_dir = Path('model_eval/eval_results')
output_dir.mkdir(parents=True, exist_ok=True)
log_file = output_dir / 'model_inference_speed_output.txt'
log_fh = open(str(log_file), 'w')
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
    def flush(self):
        for f in self.files:
            f.flush()
sys.stdout = Tee(sys.stdout, log_fh)
sys.stderr = Tee(sys.stderr, log_fh)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate full-pipeline inference time')
    parser.add_argument(
        '--config', required=True,
        help='MMDetection3D config file path')
    parser.add_argument(
        '--decoder-dir', dest='decoder_dir', required=True,
        help='Directory with SegDecoder_<Name>_<Date>.pth files')
    parser.add_argument(
        '--warmup', type=int, default=10,
        help='Number of warm-up iterations before timing')
    parser.add_argument(
        '--batch-size', dest='batch_size', type=int, default=1,
        help='Batch size for inference')
    parser.add_argument(
        '--workers-per-gpu', dest='workers_per_gpu', type=int, default=2,
        help='DataLoader workers per GPU')
    parser.add_argument(
        '--iters', type=int, default=None,
        help='Number of batches to time (default: all)')
    parser.add_argument(
        '--device', default='cuda:0',
        help='Device for inference')
    return parser.parse_args()


def build_data_loader(cfg, batch_size, workers_per_gpu):
    print('[Debug] Building data loader')
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False)
    print(f'[Debug] Loader built: {len(dataset)} samples, batch size={batch_size}, workers={workers_per_gpu}')
    return loader, dataset


@torch.no_grad()
def measure_speed(model, data_loader, warmup, iters):
    print('[Debug] Starting measure_speed')
    # Warm-up iterations (not timed)
    for i, data in enumerate(data_loader):
        if i == 0:
            print(f'[Debug] Warm-up: first batch keys={list(data.keys())}')
        if i >= warmup:
            print(f'[Debug] Completed {warmup} warm-up iterations')
            break
        batch = {k: v for k, v in data.items() if k != 'semantic_indices'}
        _ = model(return_loss=False, **batch)

    # Timed iterations
    times = []
    for i, data in enumerate(data_loader):
        if i == 0:
            print(f'[Debug] Timing: first batch keys={list(data.keys())}')
        if iters is not None and i >= iters:
            print(f'[Debug] Completed {iters} timed iterations')
            break
        batch = {k: v for k, v in data.items() if k != 'semantic_indices'}
        torch.cuda.synchronize()
        start = time.time()
        _ = model(return_loss=False, **batch)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        if i % 50 == 0:
            print(f'[Debug] Batch {i} latency: {elapsed*1000:.1f} ms')
    avg = sum(times) / len(times) if times else 0.0
    print(f'[Debug] Average latency computed over {len(times)} batches')
    return avg

@torch.no_grad()
def measure_decoder_speed(model, loader, warmup, iters):

    # Warm-up full forward to cache features
    for i, data in enumerate(loader):
        if i >= warmup:
            break
        batch = {k: v for k, v in data.items() if k != 'semantic_indices'}
        _ = model(return_loss=False, **batch)

    times = []
    for i, data in enumerate(loader):

        if iters and i >= iters:
            break
        bev_former = model.module
        batch = {k: v for k, v in data.items() if k != 'semantic_indices'}
        device = next(model.parameters()).device
        batch_gpu = scatter(batch, [device.index])[0]
        # Extract per-view images list
        dc_img = batch['img'][0]          # DataContainer for GPU 0
        imgs   = dc_img.data[0]           # plain Tensor of shape [B, C, H, W]

        # 2) Move it to the same device as your model
        if not imgs.is_cuda:
            imgs = imgs.to(device)

        # 3) Unwrap img_metas the same way (stays on CPU)
        dc_meta   = batch['img_metas'][0]
        img_metas = dc_meta.data[0]       # list[dict]

        # 4) Now call extract_feat with a Tensor, not a list:
        model.eval()
        with torch.no_grad():
            img_feats = bev_former.extract_feat(
                img=imgs,
                img_metas=img_metas
            )
            bev_feats, seg_preds, bbox_pts = bev_former.simple_test_pts(
                img_feats,          # List[Tensor] from extract_feat
                img_metas,          # List[dict]
                prev_bev=None,      # no history on first frame
                rescale=False       # doesn't affect bev_features
            )
            bs = img_feats[0].shape[0]
            pts_bbox_head = bev_former.pts_bbox_head
            seg_bev = bev_feats.reshape(pts_bbox_head.bev_h, pts_bbox_head.bev_w, bs, -1).permute(2, 3, 1, 0)  # b, c , w, h
            seg_bev = torch.flip(seg_bev, dims=[2])
            seg_bev = pts_bbox_head.feat_cropper(seg_bev)
            # Time decoder only
            torch.cuda.synchronize()
            t0 = time.time()
            _ = pts_bbox_head.seg_decoder(seg_bev)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            times.append(elapsed)
            if i % 50 == 0:
                print(f"[Debug] Decoder-only Batch {i} latency: {elapsed*1000:.1f} ms")

    avg_dec = sum(times) / len(times) if times else 0.0
    print(f"[Debug] Decoder-only average latency over {len(times)} batches: {avg_dec*1000:.1f} ms")
    return avg_dec  # Return average decoder latency


def main():
    args = parse_args()
    print(f'[Debug] Parsed args: {args}')
    device = torch.device(args.device)
    print(f'[Debug] Using device: {device}')
    base_cfg_path = args.config

    # discover checkpoints
    decoder_dir = Path(args.decoder_dir)
    print(f'[Debug] Looking for checkpoints in {decoder_dir}')
    print(f'[Debug] Directory exists: {decoder_dir.exists()}')
    files = []
    try:
        files = list(decoder_dir.iterdir())
    except Exception as e:
        print(f'[Debug] Could not list directory: {e}')
    print(f'[Debug] Files: {[f.name for f in files]}')
    pth_files = sorted(decoder_dir.glob('SegDecoder_*_*.pth'))

    decoder_map = {}
    pattern = re.compile(r'^SegDecoder_(?P<decoder>.+)_\d{8}\.pth$')
    for pth in pth_files:
        m = pattern.match(pth.name)
        if not m:
            print(f'[Debug] Skipping unmatched file: {pth.name}')
            continue
        decoder_map[m.group('decoder')] = str(pth)
    print(f'[Debug] Found decoders: {list(decoder_map.keys())}')
    if not decoder_map:
        raise RuntimeError(f'No valid SegDecoder .pth files in {decoder_dir}')

    # build dataloader once
    base_cfg = Config.fromfile(base_cfg_path)
    loader, dataset = build_data_loader(base_cfg, args.batch_size, args.workers_per_gpu)
    print(f'[Debug] Dataset: {len(dataset)} samples, {len(loader)} batches')

    results = {}
    for name, ckpt in decoder_map.items():
        print(f'[Debug] ---------- Decoder: {name} ----------')
        cfg = Config.fromfile(base_cfg_path)
        cfg.model.pts_bbox_head.seg_encoder.type = name

        print('[Debug] Building model')
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg', {}))
        print(f'[Debug] Loading checkpoint: {ckpt}')
        checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
        if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        model = MMDataParallel(model.to(device), device_ids=[device.index or 0])
        model.eval()

        # Measure full-pipeline speed
        avg_full = measure_speed(model, loader, args.warmup, args.iters)
        # Measure semantic decoder only
        avg_decoder = measure_decoder_speed(model, loader, args.warmup, args.iters)
        results[name] = (avg_full, avg_decoder)
        print(f'Average full-pipeline latency for {name}: {avg_full*1000:.1f} ms/batch')
        print(f'Average decoder-only latency for {name}: {avg_decoder*1000:.1f} ms/batch')

    # Summary
    names = list(results.keys())
    full_vals    = np.array([results[n][0] for n in names]) * 1000
    decoder_vals = np.array([results[n][1] for n in names]) * 1000

    x = np.arange(len(names))
    width = 0.35

    # --- Combined full-pipeline & decoder-only plot (already in your code) ---
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, full_vals,    width, label='Full-Pipeline')
    plt.bar(x + width/2, decoder_vals, width, label='Decoder-Only')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Latency (ms)')
    plt.title('Inference Latency per Decoder')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_dir / 'model_inference_latency_comparison.png'))
    print(f'[Debug] Saved comparison plot to {output_dir / "model_inference_latency_comparison.png"}')

    # --- New decoder-only plot ---
    plt.figure(figsize=(12, 6))
    plt.bar(x, decoder_vals, width, label='Decoder-Only')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Latency (ms)')
    plt.title('Decoder-Only Inference Latency per Decoder')
    plt.tight_layout()
    plt.savefig(str(output_dir / 'model_inference_latency_comparison_decoder.png'))
    print(f'[Debug] Saved decoder-only plot to {output_dir / "model_inference_latency_comparison_decoder.png"}')

if __name__ == '__main__':
    main()

