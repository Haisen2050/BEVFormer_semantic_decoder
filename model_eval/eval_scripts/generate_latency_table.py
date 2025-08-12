import re
import argparse
import sys

# Patterns for full-pipeline and decoder-only latencies
FULL_PATTERN = re.compile(
    r"Average full-pipeline latency for (\S+):\s*([\d\.]+)\s*ms/batch",
    re.IGNORECASE
)
DECODER_PATTERN = re.compile(
    r"Average decoder-only latency for (\S+):\s*([\d\.]+)\s*ms/batch",
    re.IGNORECASE
)

# Regex to identify the existing table header and separator lines
TABLE_HEADER_REGEX = re.compile(r"^\|\s*Decoder\s*\|.*$")
TABLE_SEP_REGEX = re.compile(r"^\|[-\s|]+$")


def generate_table_lines(input_path: str):
    """
    Parses the latency log and returns table lines for insertion.
    """
    metrics = {}
    try:
        with open(input_path, 'r') as f:
            for line in f:
                m_full = FULL_PATTERN.search(line)
                m_dec = DECODER_PATTERN.search(line)
                if m_full:
                    model, full_ms = m_full.groups()
                    metrics.setdefault(model, {})['full_ms'] = float(full_ms)
                if m_dec:
                    model, dec_ms = m_dec.groups()
                    metrics.setdefault(model, {})['dec_ms'] = float(dec_ms)
    except FileNotFoundError:
        sys.exit(f"Error: File not found: {input_path}")
    if not metrics:
        sys.exit(f"Warning: No latency entries found in {input_path}")

    # Build lines
    lines = []
    lines.append("| Decoder                 | Full Time (ms) | FPS Full | Decoder-Only (ms) | FPS Dec |")
    lines.append("|-------------------------|----------------|----------|-------------------|---------|")
    for model, vals in sorted(metrics.items()):
        full_ms = vals.get('full_ms', 0.0)
        dec_ms = vals.get('dec_ms', 0.0)
        fps_full = round(1000.0 / full_ms, 2) if full_ms else 0.0
        fps_dec = round(1000.0 / dec_ms, 2) if dec_ms else 0.0
        lines.append(
            f"| {model:<23} | {full_ms:<14.2f} | {fps_full:<8.2f} | {dec_ms:<17.2f} | {fps_dec:<7.2f} |"
        )
    return lines


def inject_into_readme(table_lines, readme_path, section_header):
    """
    Reads README, replaces only the existing markdown table under section_header, and injects decoder-only plot image, preserving all other content.
    """
    with open(readme_path, 'r') as f:
        lines = f.readlines()

    # Locate section header
    try:
        sec_idx = next(i for i, line in enumerate(lines) if line.strip() == section_header)
    except StopIteration:
        sys.exit(f"Section header '{section_header}' not found in {readme_path}")

    # Find table header
    start = None
    for i in range(sec_idx + 1, len(lines)):
        if TABLE_HEADER_REGEX.match(lines[i]):
            start = i
            break
    if start is None:
        sys.exit("Existing table header not found.")

    # Find table separator (next line)
    sep_idx = start + 1
    if sep_idx >= len(lines) or not TABLE_SEP_REGEX.match(lines[sep_idx]):
        sys.exit("Existing table separator not found.")

    # Find end of table body
    end = sep_idx + 1
    while end < len(lines) and lines[end].lstrip().startswith('|'):
        end += 1

    # Insert new table lines and decoder-only image
    image_line = "![Decoder-only latency comparison](model_eval/eval_results/model_inference_latency_comparison_decoder.png)"
    new_block = [l + '\n' for l in table_lines] + ['\n', image_line + '\n']

    # Replace old table with new block
    new_lines = lines[:start] + new_block + lines[end:]

    with open(readme_path, 'w') as f:
        f.writelines(new_lines)

    print(f"Replaced latency table and injected decoder-only plot in section '{section_header}' of {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Update README with latency table and plot.")
    parser.add_argument(
        "-i", "--input",
        default="model_eval/eval_results/model_inference_speed_output.txt",
        help="Path to the latency log file"
    )
    parser.add_argument(
        "-r", "--readme",
        default="README.md",
        help="Path to the README to update"
    )
    parser.add_argument(
        "-s", "--section",
        default="### 3. Inference Speed Evaluation",
        help="Section header under which to inject content"
    )
    args = parser.parse_args()

    table_lines = generate_table_lines(args.input)
    inject_into_readme(table_lines, args.readme, args.section)

if __name__ == "__main__":
    main()
