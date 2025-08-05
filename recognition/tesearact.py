import pytesseract # Corrected: Changed 'Pytesseract' to 'pytesseract'
from PIL import Image
import numpy as np 
# from preprocessing.models.open_cvdenoising import opencv_denoising # This import is not used in the class methods provided, can be removed if not needed elsewhere
# from langdetect import detect, DetectorFactory # These are commented out in your original, keeping them commented
# from langdetect.lang_detect_exception import LangDetectException # These are commented out in your original, keeping them commented
import cv2
import os
import fasttext

# --- REMOVED THIS LINE: import PytesseractRecognizer ---
# The class PytesseractRecognizer is defined below, so it does not need to be imported from itself.

class PytesseractRecognizer:
    def __init__(self, lang='eng+hin'):
        """
        Initializes the recognizer with the specified language(s).
        Example: lang='eng+hin'
        """
        self.lang = lang

    def preprocess_image(self, image_path):
        """
        Converts image to grayscale and applies thresholding (optional but can improve OCR).
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(thresh)

    def recognize_text(self, image_path):
        """
        Performs text recognition using pytesseract on a given image path.
        Returns the recognized text as a string.
        """
        processed_image = self.preprocess_image(image_path)
        text = pytesseract.image_to_string(processed_image, lang=self.lang)
        return text.strip()

    def recognize_from_folder(self, folder_path):
        """
        Runs OCR on all image files in a given folder and prints the results.
        """
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(folder_path, filename)
                print(f"\n--- Recognizing: {filename} ---")
                print(self.recognize_text(file_path))

    def recognize_with_details(self, image_path):
        image = self.preprocess_image(image_path)
        data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)

        results = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            if text and conf > 0:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                
                # --- IMPORTANT: Fix the path for fasttext model ---
                # This path was hardcoded to a local Windows path.
                # It has been updated to a generic Google Drive path.
                # You NEED to update this to your fasttext model's EXACT path in Google Drive.
                fasttext_model_path = "/content/drive/MyDrive/PolyOCR_Models/lid.176.bin" # <--- UPDATE THIS PATH
                
                if not os.path.exists(fasttext_model_path):
                    print(f"Warning: FastText language detection model not found at {fasttext_model_path}. Language detection will be skipped.")
                    lang_detected = 'unknown'
                else:
                    model = fasttext.load_model(fasttext_model_path)
                    allowed_lang={'en','hi'} # Your allowed languages

                    label, score = model.predict(text.strip(), k=1)
                    lang_det = label[0].replace("__label__", "")
                    if lang_det in allowed_lang:
                        lang_detected = lang_det
                    else:
                        lang_detected = 'unknown'

                results.append({
                    "det_text": text,
                    "bounding_box_coords": [x, y, w, h],
                    "confidence": conf,
                    # "language_used_for_OCR": self.lang, # This line was commented out in your original, keeping it that way
                    "language_detected": lang_detected
                })

        return results

    def recognize_from_folder_with_details(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(folder_path, filename)
                print(f"\n--- Recognizing: {filename} ---")
                details = self.recognize_with_details(file_path)
                for item in details:
                    print(item)

# The commented out sections below are for local testing and not part of the class methods.
# img_path = r"C:\Users\jayv5\Downloads\th (2).jpg"
# image = Image.open(img_path).convert("RGB")
# ocr_data = pytesseract.image_to_data(image, lang='hin+eng', output_type=pytesseract.Output.DICT)

# img = np.asarray(image)
# image = opencv_denoising(img) # This line depends on opencv_denoising being imported and functional
