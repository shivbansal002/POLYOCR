# rec_en.py  ── English OCR (detection + recognition)
from paddleocr import PaddleOCR
import cv2, numpy as np

# ── 1.  create a single PaddleOCR instance  ────────────────────────────
#        det=True   → finds the text boxes for us
#        rec=True   → recognises the text inside those boxes
#        cls=False  → we skip the angle-classifier (faster)
ocr_en = PaddleOCR(lang="en",
                   det=True,
                   rec=True,
                   cls=False,
                   use_textline_orientation=False)

# ── 2.  tiny helper: accept ndarray or file-path and return a BGR image ─
def _img(x):
    return x if isinstance(x, np.ndarray) else cv2.imread(str(x))


# ── 3.  main entry-point ------------------------------------------------
def get_text(crop_img: str | np.ndarray) -> str:
    """
    Feed an *entire* image (not just a crop) and return all recognised
    words concatenated with spaces.

    • Accepts either a path or an already-loaded ndarray.
    • Handles any nested structure that PaddleOCR might return.
    """
    img = _img(crop_img)
    if img is None:
        raise FileNotFoundError(f"Image not found: {crop_img}")

    # ───────── OCR call (detection+recognition in one go) ─────────
    res = ocr_en.ocr(img, det=True, rec=True, cls=False)
    #           ↑↑↑ make sure *det=True* here!

    words: list[str] = []

    # Flatten PaddleOCR's nested return structure
    def _collect(node):
        if isinstance(node, (list, tuple)):
            # Leaf: (text, confidence)
            if len(node) == 2 and isinstance(node[0], str):
                words.append(node[0])
            else:
                for child in node:
                    _collect(child)

    _collect(res)
    return " ".join(words)


# ── 4.  quick CLI test  -------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rec_en.py <image_path>")
    else:
        print(get_text(sys.argv[1]))
