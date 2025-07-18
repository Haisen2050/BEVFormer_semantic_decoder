import torch
import time
from models import build_decoder  # Adjust if your models are loaded differently

# Dummy input matching your decoder input shape
B, C, H, W = 1, 256, 128, 128  # Adjust as needed for your actual input
dummy_input = torch.randn(B, C, H, W).cuda()

# Decoders to test (names must match your build_decoder keys)
decoder_names = [
    "SegDecoder_Conv1Linear1_20250704",
    "SegDecoder_DeepLabV3Plus_20250716",
    "SegDecoder_FPN1_20250713",
    "SegDecoder_FPN2_20250724",
    "SegDecoder_FPN2_256_20250727",
    "SegDecoder_UNet2Down1Up_20250722",
    "SegDecoder_UNet3Down2Up_20250709",
]

print("\n📦 Memory Usage Benchmark for SegDecoder Models")
print("=======================================================")
print(f"{'Decoder':40} {'Peak Memory (MB)':>20}")
print("-------------------------------------------------------")

for name in decoder_names:
    model = build_decoder(name).cuda().eval()

    # Warm-up
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)

    # Memory profiling
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = model(dummy_input)

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # bytes to MB

    print(f"{name:40} {peak_memory:>20.2f}")

print("=======================================================")