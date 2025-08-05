from paddleocr import PaddleOCR
import cv2

# ── Initialize CRNN models per language ──
ocr_models = {
    'en': PaddleOCR(use_textline_orientation=False, lang='en'),
    'hi': PaddleOCR(use_textline_orientation=False, lang='devanagari'),  # 🔁 fixed
    'fr': PaddleOCR(use_textline_orientation=False, lang='french'),
    'ko': PaddleOCR(use_textline_orientation=False, lang='korean'),
}


# ── OCR per language ──
def recognize_crop(img_path, lang_code):
    ocr = ocr_models[lang_code]
    result = ocr.ocr(img_path, cls=False)

    # 🔍 Debug: show raw OCR output
    print(f"🔍 Raw result for {lang_code}:", result)

    # ✅ Safety check
    if isinstance(result, list) and isinstance(result[0], list):
        try:
            text = ''.join([item[1][0] for item in result[0] if len(item) > 1])
            return text
        except Exception as e:
            print(f"❌ Error parsing result for {lang_code}:", e)
    else:
        print(f"⚠️ Unexpected result format for {lang_code}: {type(result)}")

    return ''

# ── Test one image per language ──
# (replace 'test_en.jpg' etc. with actual image paths)
test_images = {
    'en': 'test_en.jpg',
    'hi': 'test_hi.jpg',
    'fr': 'test_fr.jpg',
    'ko': 'test_ko.jpg',
}

for lang_code, img_path in test_images.items():
    print(f"\n🔤 Language: {lang_code.upper()}")
    text = recognize_crop(img_path, lang_code)
    print(f"📜 Detected Text:\n{text}")
