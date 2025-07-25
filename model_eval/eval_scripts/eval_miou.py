#!/usr/bin/env python3
import os
import re
import glob
import matplotlib.pyplot as plt

LOG_DIR    = '/home/haisen/BEVFormer_segmentation_detection/work_dirs/bevformer_base_seg_det_150x150'
LOG_PREFIX = 'SegDecoder'

# 📁 Set up clean output path
current_folder = os.path.dirname(os.path.abspath(__file__))
eval_results_folder = os.path.join(current_folder, "..", "eval_results")
os.makedirs(eval_results_folder, exist_ok=True)
OUTPUT_PNG = os.path.abspath(os.path.join(eval_results_folder, "semantic_decoder_miou.png"))

# 🧪 Regex patterns
pat_epoch = re.compile(r'Epoch\s*\[?\s*(\d+)\s*\]?')
pat_row   = re.compile(r'^\|\s*\d+\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-9]*\.[0-9]+)\s*\|')

# 🔍 Collect logs
log_paths = glob.glob(os.path.join(LOG_DIR, LOG_PREFIX + '*.log'))
if not log_paths:
    raise FileNotFoundError(f'No files matching {LOG_PREFIX}*.log in {LOG_DIR}')

all_data = {}
for lp in log_paths:
    full_name = os.path.splitext(os.path.basename(lp))[0]

    # 🧼 Remove date suffix and SegDecoder_ prefix for display
    name = re.sub(r'[:_]\d{6,}$', '', full_name)
    name = name.replace("SegDecoder_", "")

    epoch_scores = {}
    current_epoch = None
    with open(lp) as f:
        for line in f:
            me = pat_epoch.search(line)
            if me:
                current_epoch = int(me.group(1))
            mr = pat_row.match(line)
            if mr and current_epoch is not None:
                epoch_scores[current_epoch] = float(mr.group(1))

    if not epoch_scores:
        print(f'⚠️  No mIoU in {full_name}, skipping.')
        continue

    epochs = sorted(epoch_scores)
    all_data[name] = {
        'epochs': epochs,
        'miou':   [epoch_scores[e] for e in epochs]
    }

# 📊 Plot with sorted legend (by max mIoU)
plt.figure(figsize=(12, 6))  # Wider to accommodate legend outside

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

# Move legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title='Decoder')

# Adjust layout to fit legend
plt.tight_layout(rect=[0, 0, 0.85, 1])

plt.savefig(OUTPUT_PNG)
print(f'✅ Saved combined mIoU plot to {OUTPUT_PNG}')
