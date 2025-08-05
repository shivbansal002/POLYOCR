from paddleocr import PaddleOCR
import cv2, numpy as np
ocr_ta = PaddleOCR(lang="ta", det=False, cls=False, use_textline_orientation=False)

def _img(x): return x if isinstance(x, np.ndarray) else cv2.imread(str(x))
def get_text(crop_img) -> str:
    img = _img(crop_img)
    res = ocr_ta.ocr(img, det=False, rec=True, cls=False)
    return "".join([b[1][0] for b in res[0]]) if res else ""

if __name__ == "__main__":
    import sys; print(get_text(sys.argv[1]))
