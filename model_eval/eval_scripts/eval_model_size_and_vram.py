#!/usr/bin/env python3
"""
Evaluate BEVFormer semantic decoder sizes (params) and VRAM usage,
produce plots, and update README section 2 with a markdown table and both PNGs.

Run from project root:
    python model_eval/eval_scripts/eval_model_size_and_vram.py
"""
import re
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ===================== Defaults (no CLI args) =====================
# Project root inferred from .../model_eval/eval_scripts/<this_file>
ROOT = Path(__file__).resolve().parents[2]  # /<repo-root>
WORK_DIR = ROOT / "work_dirs" / "bevformer_base_seg_det_150x150"
CKPT_GLOB = str(WORK_DIR / "SegDecoder_*_*.pth")

OUT_DIR = ROOT / "model_eval" / "eval_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_TXT = OUT_DIR / "model_size_output.txt"  # keep a familiar name
PLOT_SIZE_PNG = OUT_DIR / "semantic_decoder_model_size.png"
PLOT_VRAM_PNG = OUT_DIR / "semantic_decoder_vram_bar.png"

README_PATH = ROOT / "README.md"
SECTION_HEADER = "### 2. Model Size Evaluation"

# Inference/input config
IN_CHANNELS = 256
OUT_CHANNELS = 4
DUMMY_INPUT = (2, IN_CHANNELS, 160, 160)  # (B, C, H, W)
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

# ===================== Import decoders =====================
sys.path.insert(0, str(ROOT))
from projects.mmdet3d_plugin.bevformer.modules import seg_subnet  # noqa: E402


# ===================== Safe stdout tee =====================
class StdoutTee:
    """Context manager that mirrors stdout to extra streams and restores stdout on exit."""
    def __init__(self, *extra_streams):
        self.orig = sys.stdout
        self.streams = (self.orig,) + tuple(extra_streams)

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self.orig
        for s in self.streams:
            try:
                if hasattr(s, "closed") and s.closed:
                    continue
                s.flush()
            except Exception:
                pass

    def write(self, data):
        for s in self.streams:
            try:
                if hasattr(s, "closed") and s.closed:
                    continue
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                if hasattr(s, "closed") and s.closed:
                    continue
                s.flush()
            except Exception:
                pass


# ===================== Utils =====================
CKPT_NAME_RE = re.compile(r"^SegDecoder_(?P<decoder>.+)_\d{8}\.pth$")

def bytes_to_gb(nbytes: int) -> float:
    return nbytes / (1024**3)

def summarize_model(model: nn.Module, name: str) -> Dict[str, float]:
    """Return dict with total/trainable/non_trainable (counts of params)."""
    print(f"\n📋 Model Summary for {name}")
    print("-" * 60)

    summary = {}
    hooks = []

    def hook_fn(m, inp, out):
        cls = m.__class__.__name__
        idx = len(summary) + 1
        key = f"{cls}-{idx}"
        try:
            shape = tuple(out.shape)  # type: ignore
        except Exception:
            shape = ()
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        summary[key] = (shape, params)

    for m in model.modules():
        if not isinstance(m, (nn.Sequential, nn.ModuleList)) and m is not model:
            hooks.append(m.register_forward_hook(hook_fn))

    dummy = torch.zeros(*DUMMY_INPUT, device=DEVICE)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)

    for h in hooks:
        h.remove()

    total = 0
    for k, (shape, params) in summary.items():
        print(f"{k:<30} {str(shape):<20} {params:>10,}")
        total += params

    trainable = 0
    non_trainable = 0
    for p in model.parameters():
        n = p.numel()
        if p.requires_grad:
            trainable += n
        else:
            non_trainable += n

    print("-" * 60)
    print(f"Total params:         {total:,}")
    print(f"Trainable params:     {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}\n")

    return {
        "total": float(total),
        "trainable": float(trainable),
        "non_trainable": float(non_trainable),
    }

def measure_vram_peak(model: nn.Module) -> Optional[float]:
    """Return peak VRAM (GB) for a single forward pass, or None if CUDA not available."""
    if not USE_CUDA:
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    dummy = torch.zeros(*DUMMY_INPUT, device=DEVICE)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)
        torch.cuda.synchronize(DEVICE)
    peak_bytes = torch.cuda.max_memory_allocated(DEVICE)
    return bytes_to_gb(peak_bytes)

def load_model_from_ckpt(ckpt_path: Path, cls: type) -> nn.Module:
    model = cls(inC=IN_CHANNELS, outC=OUT_CHANNELS).to(DEVICE)
    state = torch.load(str(ckpt_path), map_location="cpu")
    sd = state.get("state_dict", state)
    model.load_state_dict(sd, strict=False)
    return model


# ===================== README section replace (robust, with two images) =====================
NEXT_SECTION_RE = re.compile(r"^###\s+", re.MULTILINE)
IMG_SIZE_BLOCK_RE = re.compile(
    r"(?P<div><div[^>]*>\s*.*?semantic_decoder_model_size\.png.*?</div>)",
    re.IGNORECASE | re.DOTALL,
)
IMG_VRAM_BLOCK_RE = re.compile(
    r"(?P<div><div[^>]*>\s*.*?semantic_decoder_vram_bar\.png.*?</div>)",
    re.IGNORECASE | re.DOTALL,
)

def build_section_markdown(table_lines: List[str],
                           keep_size_block: Optional[str],
                           keep_vram_block: Optional[str],
                           include_vram_block: bool) -> List[str]:
    prose = [
        "We benchmarked the **number of trainable parameters** and the **peak VRAM usage (single forward)** for each decoder.",
        "",
    ]
    lines = [SECTION_HEADER, "", *prose, *table_lines, ""]
    # params plot
    if keep_size_block:
        lines += [keep_size_block, ""]
    else:
        lines += [
            '<div align="center">',
            '  <img src="./model_eval/eval_results/semantic_decoder_model_size.png" width="500"/>',
            "</div>",
            "",
        ]
    # vram plot
    if include_vram_block:
        if keep_vram_block:
            lines += [keep_vram_block, ""]
        else:
            lines += [
                '<div align="center">',
                '  <img src="./model_eval/eval_results/semantic_decoder_vram_bar.png" width="500"/>',
                "</div>",
                "",
            ]
    lines += ["---"]
    return [l if l.endswith("\n") else l + "\n" for l in lines]

def replace_full_section(readme_path: Path, table_lines: List[str], have_vram_png: bool):
    text = readme_path.read_text(encoding="utf-8")

    # 1) find section header
    header_pos = text.find(SECTION_HEADER)
    if header_pos == -1:
        print(f"[ERR] Section header '{SECTION_HEADER}' not found in {readme_path}")
        return

    # 2) find the end of this section (next '### ' or EOF)
    match_next = NEXT_SECTION_RE.search(text, header_pos + len(SECTION_HEADER))
    end_pos = match_next.start() if match_next else len(text)

    old_section = text[header_pos:end_pos]

    # 3) try to preserve existing PNG blocks
    keep_size_block = None
    m_size = IMG_SIZE_BLOCK_RE.search(old_section)
    if m_size:
        keep_size_block = m_size.group("div")

    keep_vram_block = None
    m_vram = IMG_VRAM_BLOCK_RE.search(old_section)
    if m_vram:
        keep_vram_block = m_vram.group("div")

    # 4) build new section content
    new_section_lines = build_section_markdown(
        table_lines,
        keep_size_block,
        keep_vram_block,
        include_vram_block=True  # always show the VRAM plot block
    )
    new_text = text[:header_pos] + "".join(new_section_lines) + text[end_pos:]

    readme_path.write_text(new_text, encoding="utf-8")
    print(f"[OK ] Replaced entire section '{SECTION_HEADER}'")


# ===================== Main =====================
def main():
    with open(LOG_TXT, "w") as lf, StdoutTee(lf):
        print(f"[INFO] Project root : {ROOT}")
        print(f"[INFO] Output dir   : {OUT_DIR}")
        print(f"[INFO] Device       : {'CUDA' if USE_CUDA else 'CPU'}")

        ckpts = sorted(Path(p) for p in glob.glob(CKPT_GLOB))
        if not ckpts:
            print("[ERR ] No decoder checkpoints found:")
            print(f"       {CKPT_GLOB}")
            return

        results: List[Tuple[str, Dict[str, float], Optional[float]]] = []
        for ckpt in ckpts:
            m = CKPT_NAME_RE.match(ckpt.name)
            if not m:
                print(f"[WARN] Skipping {ckpt.name}: pattern mismatch")
                continue
            name = m.group("decoder")
            cls = getattr(seg_subnet, name, None)
            if cls is None:
                print(f"[WARN] Class '{name}' not found in seg_subnet, skipping")
                continue

            try:
                model = load_model_from_ckpt(ckpt, cls)
                sizes = summarize_model(model, name)
                vram_gb = measure_vram_peak(model)
                if vram_gb is None:
                    print(f"[INFO] Peak VRAM: N/A (CUDA not available).")
                else:
                    print(f"[OK ] Peak VRAM (single forward): {vram_gb:.3f} GB")

                results.append((name, sizes, vram_gb))
                del model
                if USE_CUDA:
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"[ERR ] Error evaluating {name}: {e}")

        if not results:
            print("[ERR] No models evaluated.")
            return

        # ===================== PLOTS =====================
        # Sort by trainable params (desc) for size plot
        results_size = sorted(results, key=lambda x: x[1]["trainable"], reverse=True)
        names_size = [n for n, _, _ in results_size]
        trainable_m = [r[1]["trainable"] / 1e6 for r in results_size]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(names_size, trainable_m)
        for bar, val in zip(bars, trainable_m):
            plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}M", ha="center", va="bottom")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Trainable Params (Millions)")
        plt.title("Semantic Decoder Parameter Counts")
        plt.tight_layout()
        plt.savefig(PLOT_SIZE_PNG, dpi=300)
        print(f"Plot saved to: {PLOT_SIZE_PNG}")

        # VRAM plot (desc). If no CUDA/values, make a placeholder chart so the README image exists.
        have_vram = [(n, v) for n, _, v in results if v is not None]
        if have_vram:
            have_vram.sort(key=lambda x: x[1], reverse=True)
            names_vram = [n for n, _ in have_vram]
            vram_vals = [v for _, v in have_vram]

            plt.figure(figsize=(10, 5))
            bars = plt.bar(names_vram, vram_vals)
            for bar, val in zip(bars, vram_vals):
                plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f} GB", ha="center", va="bottom")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Peak VRAM (GB) — single forward")
            plt.title("Semantic Decoder Peak VRAM Usage")
            plt.tight_layout()
            plt.savefig(PLOT_VRAM_PNG, dpi=300)
            print(f"VRAM plot saved to: {PLOT_VRAM_PNG}")
            have_vram_png = True
        else:
            # Placeholder chart so the README can still show an image
            plt.figure(figsize=(10, 3))
            plt.axis('off')
            plt.text(0.5, 0.5,
                     "VRAM comparison unavailable\n(CUDA not available or no measurements)",
                     ha='center', va='center')
            plt.tight_layout()
            plt.savefig(PLOT_VRAM_PNG, dpi=300)
            print(f"[INFO] Wrote placeholder VRAM PNG: {PLOT_VRAM_PNG}")
            have_vram_png = True  # we just created a placeholder

        # ===================== README TABLE (desc by trainable) =====================
        header = [
            "| Decoder Architecture     | Trainable (M) | Total (M) | Peak VRAM (GB) |",
            "|--------------------------|---------------|-----------|----------------|",
        ]
        body = []
        for name, sizes, vram_gb in results_size:
            train_m = sizes["trainable"] / 1e6
            total_m = sizes["total"] / 1e6
            vram_txt = f"{vram_gb:.3f}" if vram_gb is not None else "N/A"
            body.append(f"| {name:<24} | {train_m:>11.2f} | {total_m:>9.2f} | {vram_txt:>14} |")

        table_lines = header + body

        # Replace the entire section robustly (preserve existing image blocks if present, insert both)
        replace_full_section(README_PATH, table_lines, have_vram_png)
        print("[DONE] Model size & VRAM section replaced with both PNGs.\n")


if __name__ == "__main__":
    main()
