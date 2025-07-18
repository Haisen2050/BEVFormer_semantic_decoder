import os
import torch
from torchview import draw_graph

# Import all decoder classes from your refactored seg_subnet
from projects.mmdet3d_plugin.bevformer.modules.seg_subnet import (
    UNet2Down1Up,
    UNet3Down2Up,
    UNet4Down3Up,
    Conv1Linear1,
    FPN1,
    FPN2,
    DeepLabV3Plus,
    PanopticSegFormerDecoder,
)

# Create output directory
current_dir = os.path.dirname(__file__)
output_dir = os.path.join(current_dir, "model_graphs")
os.makedirs(output_dir, exist_ok=True)

# Input shape
example_input = torch.zeros((1, 256, 200, 400))

# Define each model and corresponding instantiation args
models_to_plot = [
    ("UNet2Down1Up", UNet2Down1Up, dict(inC=256, outC=4)),
    ("UNet3Down2Up", UNet3Down2Up, dict(inC=256, outC=4)),
    ("UNet4Down3Up", UNet4Down3Up, dict(inC=256, outC=4)),
    ("Conv1Linear1", Conv1Linear1, dict(in_channel=256, outC=4, num_deconv_filters=(256, 128, 64), num_deconv_kernels=(4, 4, 4), use_dcn=False)),
    ("FPN1", FPN1, dict(inC=256, outC=4)),
    ("FPN2", FPN2, dict(inC=256, outC=4)),
    ("DeepLabV3Plus", DeepLabV3Plus, dict(inC=256, outC=4)),
    ("PanopticSegFormerDecoder", PanopticSegFormerDecoder, dict(inC=256, outC=4)),
]

for name, cls, kwargs in models_to_plot:
    print(f"🔧 Processing {name} ...")
    try:
        model = cls(**kwargs)
        # PanopticSegFormerDecoder expects a list of feature maps
        input_data = [example_input] * 4 if name == "PanopticSegFormerDecoder" else example_input

        graph = draw_graph(
            model,
            input_data=input_data,
            expand_nested=True,
            save_graph=False,
            show_shapes=True,
            graph_name=name
        )

        graph.visual_graph.render(
            filename=name,
            format="png",
            directory=output_dir,
            cleanup=True
        )
        print(f"✅ Saved: {output_dir}/{name}.png")

    except Exception as e:
        print(f"❌ Failed: {name} → {str(e)}")