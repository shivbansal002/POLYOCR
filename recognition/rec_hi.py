# rec_hi.py
from paddleocr import PaddleOCR
import cv2, numpy as np

ocr_hi = PaddleOCR(lang="hi", det=False, cls=False, use_textline_orientation=False)

def _img(x):
    return x if isinstance(x, np.ndarray) else cv2.imread(str(x))

def get_text(crop_img: str | np.ndarray) -> str:
    img = _img(crop_img)
    if img is None:
        raise FileNotFoundError(f"Image not found: {crop_img}")

    res = ocr_hi.ocr(img, det=False, rec=True, cls=False)
    words = []

    def _collect(node):
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and isinstance(node[0], str):
                words.append(node[0])
            else:
                for child in node:
                    _collect(child)

    _collect(res)
    return " ".join(words)

if __name__ == "__main__":
    import sys
    print(get_text(sys.argv[1]))
