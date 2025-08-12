#!/usr/bin/env python3
import os
import re
import glob
import matplotlib.pyplot as plt

LOG_DIR    = '/home/haisen/BEVFormer_segmentation_detection/work_dirs/bevformer_base_seg_det_150x150'
LOG_PREFIX = 'SegDecoder'
OUTPUT_PNG = 'all_decoders_miou.png'

pat_epoch = re.compile(r'Epoch\s*\[?\s*(\d+)\s*\]?')
pat_row   = re.compile(r'^\|\s*\d+\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([0-9]*\.[0-9]+)\s*\|')

log_paths = glob.glob(os.path.join(LOG_DIR, LOG_PREFIX + '*.log'))
if not log_paths:
    raise FileNotFoundError(f'No files matching {LOG_PREFIX}*.log in {LOG_DIR}')

all_data = {}
for lp in log_paths:
    full_name = os.path.splitext(os.path.basename(lp))[0]
    # keep SegEncoder prefix but strip off trailing date
    name = re.sub(r'[:_]\d{6,}$', '', full_name)

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

plt.figure(figsize=(10, 6))
for name, data in all_data.items():
    plt.plot(data['epochs'], data['miou'], marker='o', linestyle='-', label=name)

plt.title('Validation mIoU over Epochs')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.grid(True)
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.savefig(OUTPUT_PNG)
print(f'✅ Saved combined mIoU plot to {OUTPUT_PNG}')