import os, textwrap
from rec_multilang import get_text

folder = "test_images"
imgs = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

print(f"Found {len(imgs)} image(s) in {folder}\n")

for name in imgs:
    path = os.path.join(folder, name)
    try:
        txt = get_text(path)
        print(f"{name:<25} →  {txt}")
    except Exception as e:
        print(f"{name:<25} →  ERROR: {e}")
