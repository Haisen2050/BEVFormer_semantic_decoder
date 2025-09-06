#!/usr/bin/env python3
import os, re, glob, sys, csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ---------- Defaults (no CLI args) ----------
ROOT = '/home/haisen/BEVFormer_segmentation_detection'
PRIMARY = os.path.join(ROOT, 'model_eval', 'eval_results', 'model_debug_visu', '*', 'run.log')
FALLBACK = os.path.join(ROOT, 'work_dirs', 'bevformer_base_seg_det_150x150', 'SegDecoder_*', 'run.log')

OUT_DIR = os.path.join(ROOT, 'model_eval', 'eval_results')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_TXT  = os.path.join(OUT_DIR, 'semantic_decoder_miou_summary.txt')
OUT_CSV  = os.path.join(OUT_DIR, 'semantic_decoder_miou_summary.csv')
OUT_PNG  = os.path.join(OUT_DIR, 'semantic_decoder_miou_bar.png')

README_PATH = os.path.join(ROOT, 'README.md')
SECTION_HEADER = "### 1. mIoU Evaluation"

print(f"[INFO] Project root : {ROOT}")
print(f"[INFO] Output dir   : {OUT_DIR}")

# ---------- Regex ----------
# Match a table data row anywhere in the line:
#   |   600   | 0.4314 | 0.292 | 0.4317 | 0.385 |
row_re = re.compile(
    r"\|\s*(\d+)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|"
)

# README helpers
TABLE_HEADER_REGEX = re.compile(
    r"^\|\s*Decoder\s*Architecture\s*\|\s*mIoU\s*\|\s*$",
    re.IGNORECASE
)
TABLE_SEP_REGEX = re.compile(r"^\|[-\s|]+$")
IMG_LINE_REGEX = re.compile(r'img\s+src=.*semantic_decoder_miou_bar\.png', re.IGNORECASE)

# ---------- Collect logs ----------
logs = glob.glob(PRIMARY)
if logs:
    print(f"[INFO] Found {len(logs)} log(s) in model_debug_visu:")
    for p in logs: print(f"       - {p}")
else:
    print("[WARN] No logs in model_debug_visu; trying work_dirs fallback…")
    logs = glob.glob(FALLBACK)
    if logs:
        print(f"[INFO] Found {len(logs)} log(s) in work_dirs:")
        for p in logs: print(f"       - {p}")
    else:
        print("[ERR ] No run.log files found.")
        print(f"       Checked:\n         {PRIMARY}\n         {FALLBACK}")
        sys.exit(1)

results = []  # list of dicts: model, val_num, divider, crossing, boundary, miou

for lp in logs:
    model = os.path.basename(os.path.dirname(lp))  # folder name under model_debug_visu or SegDecoder_*
    model = model.replace('SegDecoder_', '')       # normalize if coming from work_dirs
    print(f"\n[INFO] Parsing {model}: {lp}")

    found_rows = []
    with open(lp, 'r', errors='ignore') as f:
        for i, line in enumerate(f, 1):
            m = row_re.search(line)
            if m:
                valnum = int(m.group(1))
                divider = float(m.group(2))
                crossing = float(m.group(3))
                boundary = float(m.group(4))
                miou = float(m.group(5))
                found_rows.append((i, valnum, divider, crossing, boundary, miou))
                print(f"  [DBG] line {i}: N={valnum} D={divider:.4f} C={crossing:.4f} B={boundary:.4f} mIoU={miou:.4f}")

    if not found_rows:
        print("  [WARN] No mIoU table rows found in this log.")
        continue

    # Use the last (most recent) table in the log
    _, valnum, divider, crossing, boundary, miou = found_rows[-1]
    results.append({
        'model': model,
        'val_num': valnum,
        'divider': divider,
        'crossing': crossing,
        'boundary': boundary,
        'miou': miou
    })
    print(f"  [OK ] Final for {model}: mIoU={miou:.4f} (N={valnum})")

if not results:
    print("[ERR] No mIoU rows parsed from any log. Exiting.")
    sys.exit(2)

# ---------- Sort by mIoU desc ----------
results.sort(key=lambda d: d['miou'], reverse=True)

# ---------- Write TXT ----------
with open(OUT_TXT, 'w') as wf:
    for r in results:
        wf.write(
            f"Model: {r['model']}\n"
            f"  Validation num: {r['val_num']}\n"
            f"  Divider:        {r['divider']:.4f}\n"
            f"  Pred Crossing:  {r['crossing']:.4f}\n"
            f"  Boundary:       {r['boundary']:.4f}\n"
            f"  mIoU:           {r['miou']:.4f}\n\n"
        )
print(f"[OK ] Wrote TXT summary: {OUT_TXT}")

# ---------- Write CSV ----------
with open(OUT_CSV, 'w', newline='') as cf:
    writer = csv.writer(cf)
    writer.writerow(['Model', 'Validation num', 'Divider', 'Pred Crossing', 'Boundary', 'mIoU'])
    for r in results:
        writer.writerow([r['model'], r['val_num'], r['divider'], r['crossing'], r['boundary'], r['miou']])
print(f"[OK ] Wrote CSV summary: {OUT_CSV}")

# ---------- Plot bar of final mIoU per model (desc, largest left) ----------
labels = [r['model'] for r in results]
values = [r['miou'] for r in results]

plt.figure(figsize=(12, 6))
plt.bar(range(len(values)), values)
plt.xticks(range(len(values)), labels, rotation=30, ha='right')
plt.ylabel('mIoU')
plt.title('Final mIoU per Semantic Decoder (from logs)')

ax = plt.gca()
ax.set_ylim(0, max(0.5, max(values) + 0.02))  # headroom above max
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"[OK ] Saved PNG plot: {OUT_PNG}")

# ---------- Build markdown table lines (desc by mIoU) ----------
def fmt_float(x: float) -> str:
    return f"{x:.4f}".rstrip('0').rstrip('.') if '.' in f"{x:.4f}" else f"{x:.4f}"

table_lines = [
    "| Decoder Architecture     | mIoU |",
    "|--------------------------|------|",
]
for r in results:
    table_lines.append(f"| {r['model']:<24} | {fmt_float(r['miou']):>4} |")

# ---------- Update README: insert/replace table under SECTION_HEADER ----------
def update_readme_with_table(readme_path: str, section_header: str, new_table_lines: list):
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 1) Find the section header line
    try:
        sec_idx = next(i for i, line in enumerate(lines) if line.strip() == section_header)
    except StopIteration:
        print(f"[ERR] Section header '{section_header}' not found in {readme_path}")
        sys.exit(3)

    # 2) Within this section, try to find an existing mIoU table header
    start = None
    img_block_start = None
    img_block_end = None

    # Find image block bounds (<div ...> ... </div>) that contains our PNG
    for i in range(sec_idx + 1, len(lines)):
        if '<div' in lines[i]:
            # possible start of image block
            img_block_start = i
        if IMG_LINE_REGEX.search(lines[i] or ""):
            # we are inside the image block
            pass
        if '</div>' in lines[i] and img_block_start is not None:
            # potential end; verify block had our image
            block_text = ''.join(lines[img_block_start:i+1])
            if IMG_LINE_REGEX.search(block_text):
                img_block_end = i
                break

    # Find an existing mIoU table between header and image block (or before next section)
    scan_end = img_block_start if img_block_start is not None else len(lines)
    for i in range(sec_idx + 1, scan_end):
        if TABLE_HEADER_REGEX.match(lines[i]):
            start = i
            break

    if start is not None:
        # Replace existing table
        sep_idx = start + 1
        if sep_idx >= len(lines) or not TABLE_SEP_REGEX.match(lines[sep_idx]):
            print("[ERR] Existing table separator not found.")
            sys.exit(4)

        end = sep_idx + 1
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1

        new_block = [l + "\n" for l in new_table_lines]
        new_lines = lines[:start] + new_block + lines[end:]
        with open(readme_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print(f"[OK ] Replaced mIoU table in section '{section_header}'")
        return

    # No table found: insert table just above the image block (if present), otherwise right after header text
    insert_at = img_block_start if img_block_start is not None else (sec_idx + 1)
    insertion = [l + "\n" for l in new_table_lines] + ["\n"]
    new_lines = lines[:insert_at] + insertion + lines[insert_at:]

    with open(readme_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"[OK ] Inserted mIoU table in section '{section_header}'")

update_readme_with_table(README_PATH, SECTION_HEADER, table_lines)
print("[DONE] mIoU section updated.")
