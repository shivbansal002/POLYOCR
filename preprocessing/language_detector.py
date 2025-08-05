import fasttext
import os
import requests

# Path to the pre-trained fastText language detection model
# This model is specifically for language identification.
MODEL_PATH = "model/lid.176.bin"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

class LanguageDetector:
    """
    A class to handle fastText language detection.
    Downloads the model if it's not present and provides a detection method.
    """
    _model = None

    def __init__(self):
        if not os.path.exists('model'):
            os.makedirs('model')
        if not os.path.exists(MODEL_PATH):
            print("Downloading fastText language detection model...")
            self.download_model()
        if LanguageDetector._model is None:
            try:
                LanguageDetector._model = fasttext.load_model(MODEL_PATH)
            except Exception as e:
                print(f"Error loading fastText model: {e}")
                raise

    def download_model(self):
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("fastText model downloaded successfully.")

    def detect(self, text):
        """
        Detects the language of the given text.
        Returns the language code (e.g., 'en', 'hi').
        """
        if not text.strip():
            return None

        # fastText returns a list of tuples, e.g., (('__label__en',), [0.99])
        try:
            predictions = self._model.predict(text.replace('\n', ' '))
            label = predictions[0][0].replace('__label__', '')
            return label
        except Exception as e:
            print(f"Language detection failed for text: '{text}'. Error: {e}")
            return None

