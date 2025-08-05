import os, cv2, numpy as np
from paddleocr import PaddleOCR
import scipy.io as sio

# Initialize PaddleOCR English recognizer
ocr_en = PaddleOCR(lang="en", det=False, cls=False, use_textline_orientation=False)

def _img(x):
    return x if isinstance(x, np.ndarray) else cv2.imread(str(x))


def get_text(crop_img:str|np.ndarray) -> str:
    """
    Robust helper – flattens any PaddleOCR return shape and
    concatenates the recognised words.
    """
    img = crop_img if isinstance(crop_img, np.ndarray) else cv2.imread(str(crop_img))
    if img is None:
        raise FileNotFoundError(crop_img)

    res = ocr_en.ocr(img, det=False, rec=True, cls=False)
    words = []

    # --- flatten recursively ---
    def _collect(node):
        if isinstance(node, (list, tuple)):
            #  ( text , conf )   or   [text , conf]
            if len(node) == 2 and isinstance(node[0], str):
                words.append(node[0])
            else:
                for child in node:
                    _collect(child)

    _collect(res)
    return " ".join(words)



if __name__ == "__main__":
    # Load IIIT5K testdata.mat file
    mat = sio.loadmat("dataset/IIIT5K/testdata.mat")
    samples = mat["testdata"][0]

    # Pick 1 sample (e.g., the first one)
    sample = samples[0]
    img_name = sample["ImgName"][0]
    gt_text = sample["GroundTruth"][0]

    # Compose full path to the image
    img_path = os.path.join("dataset/IIIT5K", img_name)

    # Run recognition
    pred_text = get_text(img_path)

    # Print everything
    print(f"Image: {img_name}")
    print(f"Ground Truth: {gt_text}")
    print(f"OCR Prediction: {pred_text}")


