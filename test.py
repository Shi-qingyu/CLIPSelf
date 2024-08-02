import sys
sys.path.append("src")

from open_clip.factory import create_model_and_transforms
import torch

model, preprocess_train, preprocess_val = create_model_and_transforms(
    model_name="ViT-bigG-14",
    pretrained="eva",
    cache_dir="checkpoints/open_clip_pytorch_model.bin",
    precision="fp16",
    device="cuda"
)

with torch.no_grad():
    image = torch.randn((2, 3, 896, 896), dtype=torch.float16, device="cuda")
    normed_boxes = [torch.tensor([[0,   0,    0.5, 0.33],
                                  [0.5, 0,    0.5, 0.33],
                                  [0,   0.33, 0.5, 0.66],
                                  [0.5, 0.33, 1,   0.66]], dtype=torch.float16, device="cuda")]
    print(model.visual.extract_roi_features(image, normed_boxes).shape)