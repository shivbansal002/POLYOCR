import sys
import os
import torch

# Add the root directory (OCR-1) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.craft_architecture import CRAFT

model = CRAFT(pretrained=True)  # Use pretrained weights
input_tensor = torch.randn(1, 3, 768, 768)  # Dummy input
output, _ = model(input_tensor)

print("Output shape:", output.shape)
print("CRAFT imported successfully.")