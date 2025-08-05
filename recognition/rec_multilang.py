"""
Universal multi-language OCR helper
----------------------------------
• One DB text-detector (English model – works for any script)
• Seven recognition models      en / hi / zh / ja / ko / ta / te
• Tiny Unicode-range heuristic to choose the right recogniser
"""

from __future__ import annotations
import os, unicodedata, cv2, numpy as np
from paddleocr import PaddleOCR

# ───────────────────────── detector (boxes only)
detector = PaddleOCR(lang="en", det=True,  rec=False, cls=False)

# ───────────────────────── per-script recognisers
TAG = dict(
    en="latin",         # English / French …
    hi="devanagari",    # Hindi
    zh="ch",            # Simplified Chinese
    ja="japan",         # Japanese
    ko="korean",        # Korean
    ta="ta",            # Tamil
    te="te",            # Telugu
)
recogniser = {
    code: PaddleOCR(lang=tag, det=False, rec=True, cls=False)
    for code, tag in TAG.items()
}

# ───────────────────────── quick script guess
def guess_lang(text: str) -> str:
    if not text:
        return "en"
    name = unicodedata.name(text[0], "")
    if   "DEVANAGARI" in name:                return "hi"
    elif "CJK UNIFIED" in name:               return "zh"
    elif "HIRAGANA" in name or "KATAKANA" in name: return "ja"
    elif "HANGUL"   in name:                  return "ko"
    elif "TAMIL"    in name:                  return "ta"
    elif "TELUGU"   in name:                  return "te"
    else:                                     return "en"

# ───────────────────────── MAIN API
def get_text(img_in: str | np.ndarray) -> str:
    """
    Detect text lines in *img_in* and recognise them with the proper script
    recogniser. Returns all lines joined by '\n'.  Empty string if nothing.
    """
    # ---------- load image ----------
    img = img_in if isinstance(img_in, np.ndarray) else cv2.imread(str(img_in))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_in}")

    # ---------- detect ----------
    det = detector.ocr(img, det=True, rec=False, cls=False)
    if not det or not det[0]:
        return ""

    lines_out: list[str] = []

    # ---------- recognise each box ----------
    for poly in det[0][::-1]:                      # polygon only
        pts = np.array(poly).astype(int)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        if x1 <= x0 or y1 <= y0:
            continue                         # bad box
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        best_txt, best_conf = "", -1.0

        # ---------- run EVERY recogniser once ----------
        for lang, ocr in recogniser.items():
            res = ocr.ocr(crop, det=False, rec=True, cls=False)
            if not res or not res[0]:
                continue
            txt, conf = res[0][0]
            if conf > best_conf:
                best_txt, best_conf = txt, conf

        if best_txt.strip():
            lines_out.append(best_txt)

    return "\n".join(lines_out)


# ───────────────────────── CLI smoke-test
if __name__ == "__main__":
    import sys, textwrap
    if len(sys.argv) < 2:
        print("Usage: python rec_multilang.py <image>")
        sys.exit(1)
    out = get_text(sys.argv[1])
    print(textwrap.dedent(out))
