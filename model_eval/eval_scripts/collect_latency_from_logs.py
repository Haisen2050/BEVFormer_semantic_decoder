#!/usr/bin/env python3
import os
import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/haisen/BEVFormer_segmentation_detection"
IN_TXT = os.path.join(ROOT, "model_eval", "eval_results", "model_inference_speed_output.txt")

OUT_DIR = os.path.join(ROOT, "model_eval", "eval_results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FULL_PNG    = os.path.join(OUT_DIR, "model_inference_latency_comparison.png")
OUT_DECODER_PNG = os.path.join(OUT_DIR, "model_inference_latency_comparison_decoder.png")

print(f"[INFO] Input  : {IN_TXT}")
print(f"[INFO] OutDir : {OUT_DIR}")

if not os.path.isfile(IN_TXT):
    print("[ERR ] Summary file not found. Exiting.")
    sys.exit(1)

# Lines look like:
# Average full-pipeline latency for UNet4Res18: 586.4 ms/batch
# Average decoder-only latency for UNet4Res18: 6.3 ms/batch
rx_full = re.compile(
    r"Average\s+full[-\s]?pipeline\s+latency\s+for\s+(.+?):\s*([0-9]*\.?[0-9]+)\s*ms/batch",
    re.IGNORECASE
)
rx_dec = re.compile(
    r"Average\s+decoder[-\s]?only\s+latency\s+for\s+(.+?):\s*([0-9]*\.?[0-9]+)\s*ms/batch",
    re.IGNORECASE
)

full = {}    # model -> ms
decoder = {} # model -> ms

with open(IN_TXT, "r", errors="ignore") as f:
    for ln, line in enumerate(f, 1):
        m = rx_full.search(line)
        if m:
            model = m.group(1).strip().replace("SegDecoder_", "")
            val = float(m.group(2))
            full[model] = val   # last occurrence wins
            print(f"[FULL]    {model:<20} {val:>8.3f}  ({ln})")
            continue
        m = rx_dec.search(line)
        if m:
            model = m.group(1).strip().replace("SegDecoder_", "")
            val = float(m.group(2))
            decoder[model] = val
            print(f"[DECODER] {model:<20} {val:>8.3f}  ({ln})")

if not full and not decoder:
    print("[ERR ] No latency lines matched in the summary. Exiting.")
    sys.exit(2)

def plot_bar(data_dict, title, out_png):
    items = sorted(data_dict.items(), key=lambda kv: kv[1], reverse=True)  # large -> small
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms/batch)")
    ax.set_title(title)

    if values:
        ax.set_ylim(0, max(values) * 1.08)
        # annotate values
        for i, v in enumerate(values):
            ax.text(i, v * 1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK ] Saved {out_png}")

if full:
    print("\n[INFO] Plotting full-pipeline latencies (desc)…")
    plot_bar(full, "Average Full-Pipeline Latency (ms/batch)", OUT_FULL_PNG)
else:
    print("[WARN] No full-pipeline entries found; skipping full plot.")

if decoder:
    print("\n[INFO] Plotting decoder-only latencies (desc)…")
    plot_bar(decoder, "Average Decoder-Only Latency (ms/batch)", OUT_DECODER_PNG)
else:
    print("[WARN] No decoder-only entries found; skipping decoder plot.")
