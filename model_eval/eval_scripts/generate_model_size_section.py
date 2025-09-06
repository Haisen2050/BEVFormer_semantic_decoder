#!/usr/bin/env python3
import re
import sys
from typing import Dict, List
import matplotlib.pyplot as plt

# ---------- Regex patterns ----------
SEC_PATTERN = re.compile(r"^\s*[📋]?\s*Model Summary for\s+(.+?)\s*$", re.IGNORECASE)
TRAIN_PATTERN = re.compile(r"^\s*Trainable params:\s*([\d,]+)\s*$", re.IGNORECASE)

# README table detection for Section 2
TABLE_HEADER_REGEX = re.compile(
    r"^\|\s*Decoder\s+Architecture\s*\|\s*Trainable\s+Parameters\s*\|\s*$",
    re.IGNORECASE,
)
TABLE_SEP_REGEX = re.compile(r"^\|[-\s|]+$")

# ---------- Default paths ----------
LOG_PATH = "model_eval/eval_results/model_size_output.txt"
README_PATH = "README.md"
SECTION_HEADER = "### 2. Model Size Evaluation"
PNG_PATH = "model_eval/eval_results/semantic_decoder_model_size.png"


def _to_int(num_str: str) -> int:
    return int(num_str.replace(",", "").strip())


def parse_trainable_params(log_path: str) -> Dict[str, int]:
    models: Dict[str, int] = {}
    cur = None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")

                m_sec = SEC_PATTERN.match(line)
                if m_sec:
                    cur = m_sec.group(1).strip()
                    continue

                if cur is None:
                    continue

                m_tr = TRAIN_PATTERN.match(line)
                if m_tr:
                    models[cur] = _to_int(m_tr.group(1))
                    cur = None
    except FileNotFoundError:
        sys.exit(f"Error: File not found: {log_path}")

    if not models:
        sys.exit(f"Warning: No trainable parameters found in {log_path}")

    return models


def fmt_millions(n: int) -> str:
    val = n / 1_000_000.0
    s = f"{val:.2f}"
    s = s.rstrip("0").rstrip(".")
    return f"{s} M"


def generate_table_lines(models: Dict[str, int]) -> List[str]:
    # Table stays ascending for easier reading
    lines = [
        "| Decoder Architecture     | Trainable Parameters |",
        "|--------------------------|----------------------|",
    ]
    for name, tp in sorted(models.items(), key=lambda kv: kv[1]):
        lines.append(f"| {name:<24} | {fmt_millions(tp):>20} |")
    return lines


def save_bar_chart(models: Dict[str, int], png_path: str):
    # Sort descending for bar chart (largest leftmost)
    items = sorted(models.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    vals_m = [v / 1_000_000.0 for v in vals]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, vals_m)
    plt.title("Decoder Model Size (Trainable Params)")
    plt.ylabel("Trainable Parameters (Millions)")
    plt.xticks(rotation=25, ha="right")

    for rect, v in zip(bars, vals_m):
        plt.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height(),
            f"{v:.2f}M",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()


def replace_table_in_readme(readme_path: str, section_header: str, new_table_lines: List[str]):
    with open(readme_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    try:
        sec_idx = next(i for i, line in enumerate(lines) if line.strip() == section_header)
    except StopIteration:
        sys.exit(f"Section header '{section_header}' not found in {readme_path}")

    start = None
    for i in range(sec_idx + 1, len(lines)):
        if TABLE_HEADER_REGEX.match(lines[i]):
            start = i
            break
    if start is None:
        sys.exit("Existing table header not found under the specified section.")

    sep_idx = start + 1
    if sep_idx >= len(lines) or not TABLE_SEP_REGEX.match(lines[sep_idx]):
        sys.exit("Existing table separator not found.")

    end = sep_idx + 1
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1

    new_block = [l + "\n" for l in new_table_lines]
    new_lines = lines[:start] + new_block + lines[end:]

    with open(readme_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def main():
    models = parse_trainable_params(LOG_PATH)
    table_lines = generate_table_lines(models)
    save_bar_chart(models, PNG_PATH)
    replace_table_in_readme(README_PATH, SECTION_HEADER, table_lines)

    print(f"✅ Updated table under section '{SECTION_HEADER}' in {README_PATH}")
    print(f"🖼️ Saved plot to: {PNG_PATH}")


if __name__ == "__main__":
    main()
