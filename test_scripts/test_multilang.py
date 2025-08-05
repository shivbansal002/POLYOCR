# rec_multilang.py  ── one-file multilingual recogniser
from __future__ import annotations
import unicodedata, cv2, numpy as np
from paddleocr import PaddleOCR

# ── 1. detector (English PP-OCRv3 detector works for all scripts) ──────────
detector = PaddleOCR(lang="en", det=True, rec=False, cls=False)

# ── 2. one recogniser per target script ────────────────────────────────────
TAG = dict(            # language-code ➜ PaddleOCR tag
    en="latin",        # English / French (Latin)
    hi="devanagari",   # Hindi
    zh="ch",           # Simplified Chinese
    ja="japan",        # Japanese
    ko="korean",       # Korean
    ta="ta",           # Tamil
    te="te",           # Telugu
)
recogniser = {code: PaddleOCR(lang=tag, det=False, rec=True, cls=False)
              for code, tag in TAG.items()}

# ── 3. quick Unicode-range heuristic to guess script ───────────────────────
def guess_lang(text: str) -> str:
    if not text:
        return "en"
    name = unicodedata.name(text[0], "")
    if "DEVANAGARI" in name:        return "hi"
    if "CJK UNIFIED" in name:       return "zh"
    if "HIRAGANA" in name or "KATAKANA" in name: return "ja"
    if "HANGUL"   in name:          return "ko"
    if "TAMIL"    in name:          return "ta"
    if "TELUGU"   in name:          return "te"
    return "en"

# ── 4. public helper ───────────────────────────────────────────────────────
def get_text(img_in: str | np.ndarray) -> str:
    """Detect lines, pick recogniser by script, return full text."""
    img = img_in if isinstance(img_in, np.ndarray) else cv2.imread(str(img_in))
    if img is None:
        raise FileNotFoundError(img_in)

    det = detector.ocr(img, det=True, rec=False, cls=False)
    if not det or not det[0]:
        return ""

    lines: list[str] = []
    for box in det[0]:
        # box[0]  → 4 corner points [[x,y],...]
        pts = np.asarray(box[0]).astype(int)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        crop = img[y_min:y_max, x_min:x_max]

        # quick draft with Latin model to guess script
        draft = recogniser["en"].ocr(crop, det=False, rec=True, cls=False)
        draft_text = draft[0][0][0] if draft else ""
        lang = guess_lang(draft_text)

        res = recogniser[lang].ocr(crop, det=False, rec=True, cls=False)
        final_text = res[0][0][0] if res else ""
        lines.append(final_text)

    return "\n".join(lines)

# ── 5. CLI smoke-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, textwrap
    print(textwrap.dedent(get_text(sys.argv[1])))

