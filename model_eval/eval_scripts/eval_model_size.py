#!/usr/bin/env python
"""
Script to evaluate BEVFormer semantic decoder model sizes and summarize parameter counts.

Usage:
    python model_eval/eval_scripts/eval_model_size.py

The script will:
 1. Locate all SegDecoder_<Name>_<Date>.pth files in work_dirs/bevformer_base_seg_det_150x150
 2. Dynamically import the seg_subnet module and extract decoder classes by name
 3. Print a layer-by-layer summary and total/trainable parameter counts to model_summary_output.txt
 4. Plot trainable parameter counts for each decoder and save as semantic_decoder_model_size.png

Ensure you run this from the project root so paths resolve correctly.
"""
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Setup logging directory and tee stdout
log_dir = Path("model_eval/eval_results")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "model_summary_output.txt"
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()
sys.stdout = Tee(sys.stdout, open(log_file, "w"))

# Add project root to PYTHONPATH to allow package imports
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

# Import seg_subnet module dynamically
from projects.mmdet3d_plugin.bevformer.modules import seg_subnet

# Regex to extract decoder name from filename
name_regex = re.compile(r'^SegDecoder_(?P<decoder>.+)_\d{8}\.pth$')

# Model config
IN_CHANNELS = 256
OUT_CHANNELS = 4
DUMMY_INPUT = (2, IN_CHANNELS, 160, 160)

def load_model(pth_path, model_class):
    model = model_class(inC=IN_CHANNELS, outC=OUT_CHANNELS)
    state = torch.load(pth_path, map_location='cpu')
    sd = state.get('state_dict', state)
    model.load_state_dict(sd, strict=False)
    return model


def summarize_model(model, name):
    print(f"\n📋 Model Summary for {name}")
    print("-" * 60)
    summary = {}
    hooks = []
    def hook_fn(m, inp, out):
        cls = m.__class__.__name__
        idx = len(summary) + 1
        key = f"{cls}-{idx}"
        summary[key] = (tuple(out.shape), sum(p.numel() for p in m.parameters() if p.requires_grad))
    for m in model.modules():
        if not isinstance(m, (nn.Sequential, nn.ModuleList)) and m is not model:
            hooks.append(m.register_forward_hook(hook_fn))
    dummy = torch.zeros(*DUMMY_INPUT)
    with torch.no_grad(): model(dummy)
    for h in hooks: h.remove()

    total = trainable = non_trainable = 0
    for k, (shape, params) in summary.items():
        print(f"{k:<30} {shape!s:<20} {params:>10,}")
        total += params
    for p in model.parameters():
        n = p.numel()
        if p.requires_grad: trainable += n
        else: non_trainable += n
    print("-" * 60)
    print(f"Total params:         {total:,}")
    print(f"Trainable params:     {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}\n")
    return {'total': total/1e6, 'trainable': trainable/1e6, 'non_trainable': non_trainable/1e6}

if __name__ == '__main__':
    # Collect checkpoint files
    work_dir = Path("work_dirs/bevformer_base_seg_det_150x150")
    ckpts = sorted(work_dir.glob("SegDecoder_*_*.pth"))
    results = []
    for ckpt in ckpts:
        m = name_regex.match(ckpt.name)
        if not m:
            print(f"Skipping {ckpt.name}: pattern mismatch")
            continue
        name = m.group('decoder')
        cls = getattr(seg_subnet, name, None)
        if cls is None:
            print(f"Class '{name}' not found in seg_subnet, skipping")
            continue
        try:
            model = load_model(ckpt, cls)
            info = summarize_model(model, name)
            results.append((name, info))
        except Exception as e:
            print(f"Error loading {name}: {e}")

    # Plotting
    if not results:
        print("No models to summarize.")
        sys.exit(0)
    results.sort(key=lambda x: x[1]['trainable'], reverse=True)
    names = [n for n, _ in results]
    vals = [v['trainable'] for _, v in results]
    plt.figure(figsize=(10,5))
    bars = plt.bar(names, vals)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x()+bar.get_width()/2, val+0.1, f"{val:.2f}M", ha='center')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Trainable Params (M)")
    plt.title("Semantic Decoder Parameter Counts")
    plt.tight_layout()
    out_plot = log_dir / "semantic_decoder_model_size.png"
    plt.savefig(out_plot, dpi=300)
    print(f"Plot saved to: {out_plot}")

