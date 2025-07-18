import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ✅ Logging setup
log_dir = Path("model_eval")
log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / "model_summary_output.txt"

# 🔁 Redirect stdout to both terminal and file
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = Tee(sys.stdout, open(log_file_path, "w"))

# ✅ Import all decoder classes
from projects.mmdet3d_plugin.bevformer.modules.seg_subnet import (
    UNet2Down1Up,
    UNet3Down2Up,
    UNet4Down3Up,
    Conv1Linear1,
    Linear2,
    FPN1,
    FPN2,
    DeepLabV3Plus,
    PanopticSegFormerDecoder,
)

# 🔍 Map filename keywords to decoder classes
decoder_map = {
    "UNet2Down1Up": UNet2Down1Up,
    "UNet3Down2Up": UNet3Down2Up,
    "UNet4Down3Up": UNet4Down3Up,
    "Conv1Linear1": Conv1Linear1,
    "Linear2": Linear2,
    "FPN1": FPN1,
    "FPN2": FPN2,
    "DeepLabV3Plus": DeepLabV3Plus,
    "PanopticSegFormerDecoder": PanopticSegFormerDecoder,
}

# 🎯 Decoder input/output channels
IN_CHANNELS = 256
OUT_CHANNELS = 4
DUMMY_INPUT_SHAPE = (2, 256, 160, 160)

def match_decoder(filename):
    for key in decoder_map:
        if key in filename:
            return decoder_map[key], key
    return None, None

def load_model(pth_path, model_class):
    model = model_class(inC=IN_CHANNELS, outC=OUT_CHANNELS)
    state_dict = torch.load(pth_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)
    return model

def print_model_summary(model, model_name, input_size=DUMMY_INPUT_SHAPE):
    print(f"\n📋 Model Summary for {model_name}")
    print("-" * 80)
    print(f"{'Layer (type)':<35} {'Output Shape':<25} {'Param #':>10}")
    print("-" * 80)

    summary = {}
    hooks = []

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    def register_hook(module):
        def hook(module, input, output):
            class_name = module.__class__.__name__
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx + 1}"
            output_shape = list(output.shape)
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            summary[m_key] = {
                "output_shape": output_shape,
                "nb_params": params
            }
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    dummy_input = torch.zeros(*input_size)
    with torch.no_grad():
        model(dummy_input)
    for h in hooks:
        h.remove()

    for layer in summary:
        output_shape = str(summary[layer]["output_shape"])
        nb_params = summary[layer]["nb_params"]
        print(f"{layer:<35} {output_shape:<25} {nb_params:>10,}")
        total_params += nb_params

    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            non_trainable_params += p.numel()

    print("-" * 80)
    print(f"Total params:         {trainable_params + non_trainable_params:,}")
    print(f"Trainable params:     {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print("-" * 80)

    return {
        "total": (trainable_params + non_trainable_params) / 1e6,
        "trainable": trainable_params / 1e6,
        "non_trainable": non_trainable_params / 1e6
    }

# 📂 Model files
model_dir = Path("./work_dirs/bevformer_base_seg_det_150x150")
model_paths = sorted(model_dir.glob("SegDecoder*.pth"))

# 📊 Collect model param data
param_data = []

for model_path in model_paths:
    full_name = model_path.stem
    model_class, decoder_name = match_decoder(full_name)
    if model_class is None or decoder_name is None:
        print(f"⚠️  Skipping {full_name}: no decoder class matched")
        continue
    try:
        model = load_model(model_path, model_class)
        param_info = print_model_summary(model, decoder_name)
        param_data.append((decoder_name, param_info))
    except Exception as e:
        print(f"❌ Failed to load {full_name}: {e}")

# 🧮 Sort by trainable params
param_data.sort(key=lambda x: x[1]["trainable"], reverse=True)

if not param_data:
    print("\n⚠️  No decoders were successfully loaded or matched.")
    model_names, trainable_counts = [], []
else:
    print("\n📊 Parameter Count Summary (sorted by trainable params):")
    print("-" * 60)
    print(f"{'Decoder':<30} {'Trainable':>10} {'Non-trainable':>16}")
    print("-" * 60)
    for name, info in param_data:
        print(f"{name:<30} {info['trainable']:10.2f} M {info['non_trainable']:16.2f} M")
    print("-" * 60)
    model_names = [name for name, _ in param_data]
    trainable_counts = [info["trainable"] for _, info in param_data]

# 📈 Plot (trainable only)
plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, trainable_counts)

for bar, count in zip(bars, trainable_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{count:.2f}M", ha='center', va='bottom', fontsize=8)

plt.xticks(rotation=45, ha='right')
plt.ylabel("Trainable Parameters (Millions)")
plt.title("Trainable Parameter Count of Semantic Decoders")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 💾 Save plot
plot_path = log_dir / "semantic_decoder_model_size.png"
plt.savefig(plot_path, dpi=300)
print(f"\n📸 Plot saved to: {plot_path}")
