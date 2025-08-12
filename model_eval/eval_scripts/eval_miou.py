#!/usr/bin/env python3
import os
import re
import glob
import matplotlib.pyplot as plt

# 📂 Paths & prefixes
LOG_DIR    = '/home/haisen/BEVFormer_segmentation_detection/work_dirs/bevformer_base_seg_det_150x150'
LOG_PREFIX = 'SegDecoder'

# 🔧 Determine project folders
current_folder        = os.path.dirname(os.path.abspath(__file__))
# where to save PNG and TXT
output_folder         = os.path.abspath(
    os.path.join(current_folder, "..", "eval_results")
)
os.makedirs(output_folder, exist_ok=True)

# file paths
OUTPUT_PNG  = os.path.join(output_folder, "semantic_decoder_miou.png")
OUTPUT_TXT  = os.path.join(output_folder, "model_miou_output.txt")

# 🧪 Regex patterns for epochs and mIoU rows
pat_epoch = re.compile(r'Epoch\s*\[?\s*(\d+)\s*\]?')
pat_row   = re.compile(r'^\|\s*\d+\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-9]*\.[0-9]+)\s*\|')

# 🔍 Find all log files
log_paths = glob.glob(os.path.join(LOG_DIR, LOG_PREFIX + '*.log'))
if not log_paths:
    raise FileNotFoundError(f'No files matching {LOG_PREFIX}*.log in {LOG_DIR}')

all_data = {}
for lp in log_paths:
    base = os.path.splitext(os.path.basename(lp))[0]
    # clean up to friendly model name
    name = re.sub(r'[:_]\d{6,}$', '', base).replace("SegDecoder_", "")

    epoch_scores = {}
    current_epoch = None
    with open(lp, 'r') as f:
        for line in f:
            me = pat_epoch.search(line)
            if me:
                current_epoch = int(me.group(1))
            mr = pat_row.match(line)
            if mr and current_epoch is not None:
                epoch_scores[current_epoch] = float(mr.group(1))

    if not epoch_scores:
        print(f'⚠️  No mIoU in {base}, skipping.')
        continue

    epochs = sorted(epoch_scores)
    all_data[name] = {
        'epochs': epochs,
        'miou':   [epoch_scores[e] for e in epochs]
    }

# 📄 Write out results to TXT
with open(OUTPUT_TXT, 'w') as outf:
    for name, data in all_data.items():
        outf.write(f"Model: {name}\n")
        for e, m in zip(data['epochs'], data['miou']):
            outf.write(f"  Epoch {e:3d}: mIoU = {m:.4f}\n")
        outf.write("\n")
print(f'✅ Saved model mIoU results to {OUTPUT_TXT}')

# 📊 Now plot
plt.figure(figsize=(12, 6))

# sort by peak mIoU descending
sorted_items = sorted(
    all_data.items(),
    key=lambda item: max(item[1]['miou']),
    reverse=True
)

for name, data in sorted_items:
    plt.plot(data['epochs'], data['miou'], marker='o', linestyle='-', label=name)

plt.title('Validation mIoU over Epochs')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title='Decoder')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(OUTPUT_PNG)
print(f'✅ Saved combined mIoU plot to {OUTPUT_PNG}')
