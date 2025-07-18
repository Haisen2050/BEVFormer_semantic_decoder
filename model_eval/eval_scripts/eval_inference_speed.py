import torch
import time
from pathlib import Path

# 🔁 Import all decoder classes
from decoders.unet2down1up import SegDecoder_UNet2Down1Up_20250722
from decoders.unet3down2up import SegDecoder_UNet3Down2Up_20250709
from decoders.fpn1 import SegDecoder_FPN1_20250713
from decoders.fpn2 import SegDecoder_FPN2_20250724
from decoders.conv1linear1 import SegDecoder_Conv1Linear1_20250704
from decoders.deeplabv3plus import SegDecoder_DeepLabV3Plus_20250716

# 🔁 Map model names to classes
decoder_class_map = {
    "SegDecoder_UNet2Down1Up_20250722": SegDecoder_UNet2Down1Up_20250722,
    "SegDecoder_UNet3Down2Up_20250709": SegDecoder_UNet3Down2Up_20250709,
    "SegDecoder_FPN1_20250713": SegDecoder_FPN1_20250713,
    "SegDecoder_FPN2_20250724": SegDecoder_FPN2_20250724,
    "SegDecoder_Conv1Linear1_20250704": SegDecoder_Conv1Linear1_20250704,
    "SegDecoder_DeepLabV3Plus_20250716": SegDecoder_DeepLabV3Plus_20250716,
}

def load_model(pth_path, model_class):
    model = model_class()
    state_dict = torch.load(pth_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def measure_inference_time(model, input_tensor, runs=100, warmup=10):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        if input_tensor.is_cuda:
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            _ = model(input_tensor)
        if input_tensor.is_cuda:
            torch.cuda.synchronize()
        end = time.time()
    avg_time = (end - start) / runs
    return avg_time * 1000  # ms

# 🧪 Input
input_tensor = torch.randn(1, 256, 128, 128).cuda()

# 📂 Directory
model_dir = Path("./work_dirs/bevformer_base_seg_det_150x150")
model_paths = sorted(model_dir.glob("SegDecoder*.pth"))

print(f"\n📊 Inference Speed Benchmark for SegDecoder Models\n{'=' * 50}")
for model_path in model_paths:
    name = model_path.stem
    print(f"\n🔍 {name}")
    try:
        model_class = decoder_class_map[name]
        model = load_model(model_path, model_class).cuda()
        avg_ms = measure_inference_time(model, input_tensor)
        print(f"🕒 Average Inference Time: {avg_ms:.2f} ms")
        print(f"⚡ {1000 / avg_ms:.2f} FPS (frames per second)")
    except Exception as e:
        print(f"❌ Failed for {name}: {e}")