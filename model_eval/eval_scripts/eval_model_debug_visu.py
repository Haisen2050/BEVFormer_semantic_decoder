#!/usr/bin/env python
"""
Script to evaluate BEVFormer segmentation decoders using debug_test.py.

Usage (from project root):
  python model_eval/eval_scripts/eval_model_debug_visu.py \
    --config projects/configs/bevformer/bevformer_base_seg_det_150x150.py \
    --work-dir work_dirs/bevformer_base_seg_det_150x150 \
    --debug-script debug_test.py \
    [--decoder DecoderName]

For each SegDecoder_<Name>_<Date>.pth checkpoint in a work directory, this script will:
 1. Create a base visualization folder (default: model_eval/eval_results/model_debug_visu) and
    subfolders for each decoder (e.g., Unet2Down1Up, FPN2_256).
 2. Dynamically set the config key `model.pts_bbox_head.seg_encoder.type` via `--cfg-options`.
 3. Invoke debug_test.py to generate visualizations and dump images into each subfolder.
 4. Save the terminal output of each run to `run.log` inside its decoder folder.

Note:
 - debug_test.py does not accept a `--device` flag; device is determined by the config or environment.
 - The script skips missing or invalid checkpoint files and continues without crashing.
 - Use `--decoder NAME` to process only a specific decoder (e.g. FPN2_256).
"""
import os
import glob
import re
import subprocess
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BEVFormer segmentation decoders via debug_test.py"
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to BEVFormer config file'
    )
    parser.add_argument(
        '--work-dir', default='work_dirs/bevformer_base_seg_det_150x150',
        help='Directory containing SegDecoder_*.pth checkpoints'
    )
    parser.add_argument(
        '--output-root', default='model_eval/eval_results/model_debug_visu',
        help='Root directory to save visualizations for each decoder'
    )
    parser.add_argument(
        '--debug-script', default='debug_test.py',
        help='Path to debug_test.py script (relative or absolute)'
    )
    parser.add_argument(
        '--decoder',
        help='(Optional) Name of a single decoder to process (e.g. FPN2_256)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify debug_test.py exists
    if not os.path.isfile(args.debug_script):
        print(f"Error: debug script '{args.debug_script}' not found.", file=sys.stderr)
        sys.exit(1)

    # Pattern to find all SegDecoder checkpoints
    ckpt_pattern = os.path.join(args.work_dir, 'SegDecoder_*_*.pth')
    ckpts = sorted(glob.glob(ckpt_pattern))
    # If user requested a specific decoder, filter early and error if none match
    name_regex = re.compile(r'^SegDecoder_(?P<decoder>.+)_\d{8}\.pth$')
    if args.decoder:
        filtered = []
        for ckpt in ckpts:
            fn = os.path.basename(ckpt)
            m = name_regex.match(fn)
            if m and m.group('decoder') == args.decoder:
                filtered.append(ckpt)
        if not filtered:
            print(f"Error: no checkpoints found for decoder '{args.decoder}'", file=sys.stderr)
            sys.exit(1)
        ckpts = filtered
    if not ckpts:
        print(f'No checkpoints found with pattern: {ckpt_pattern}', file=sys.stderr)
        return

    # Regex to extract decoder name: between prefix and trailing date
    name_regex = re.compile(r'^SegDecoder_(?P<decoder>.+)_\d{8}\.pth$')

    # Create base directory for visualizations
    base_vis_dir = args.output_root
    os.makedirs(base_vis_dir, exist_ok=True)
    print(f"Created base visualization folder: {base_vis_dir}")

    # Process each checkpoint
    for ckpt in ckpts:
        filename = os.path.basename(ckpt)
        match = name_regex.match(filename)
        if not match:
            print(f"Skipping invalid filename: {filename}", file=sys.stderr)
            continue
        decoder_name = match.group('decoder')

        # If a specific decoder is requested, skip others
        if args.decoder and decoder_name != args.decoder:
            continue

        # Normalize folder name
        decoder_folder = decoder_name.replace('UNet', 'Unet')

        # Prepare output subfolder
        out_dir = os.path.join(base_vis_dir, decoder_folder)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nProcessing decoder '{decoder_folder}' -> {out_dir}")

        # Build command to call debug_test.py
        cfg_option = f'model.pts_bbox_head.seg_encoder.type={decoder_name}'
        cmd = [
            sys.executable, args.debug_script,
            '--config', args.config,
            '--checkpoint', ckpt,
            '--show-dir', out_dir,
            '--cfg-options', cfg_option
        ]

        print(f"Command: {' '.join(cmd)}")

        # Execute and log
        log_path = os.path.join(out_dir, 'run.log')
        with open(log_path, 'w') as log_f:
            log_f.write(f"Command: {' '.join(cmd)}\n\n")
            try:
                subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as e:
                log_f.write(f"\nError evaluating {decoder_folder}: {e}\n")
                print(f"Error in {decoder_folder}, see {log_path}", file=sys.stderr)
            except Exception as e:
                log_f.write(f"\nUnexpected error for {decoder_folder}: {e}\n")
                print(f"Unexpected error in {decoder_folder}, see {log_path}", file=sys.stderr)

    print(f"\nDone. Visualizations saved under '{base_vis_dir}'.")

if __name__ == '__main__':
    main()
