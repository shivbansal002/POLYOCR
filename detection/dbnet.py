# dbnet.py

# --- FIX: Corrected import from 'PaddleOCR' to 'paddleocr' ---
from paddleocr import TextDetection
from PIL import Image
from torchvision import transforms
# from paddleocr import PaddleOCR # This line was commented out in your original, keeping it that way
import scipy # This import is not used in the provided code, can be removed if not needed elsewhere


class DBNet:
    def __init__(self, model_dir: str = None):
        """
        Initialize the DBNet model.
        
        Args:
            model_dir (str): Path to the directory containing the DBNet model files.
                             For PaddleOCR's TextDetection, this is usually None or points to a pre-trained model path.
                             If model_dir is None, PaddleOCR will download default models.
        """
        self.model_dir = model_dir
        # TextDetection from paddleocr expects `rec_model_dir` or `det_model_dir` if you're loading custom models.
        # If model_dir is for a detection model, you might need to specify `det_model_dir`.
        # For simplicity, let's assume `model_dir` is passed to `det_model_dir` if provided,
        # otherwise, PaddleOCR will use its default detection model.
        if self.model_dir:
            print(f"Initializing PaddleOCR TextDetection with det_model_dir: {self.model_dir}")
            self.ocr = TextDetection(det_model_dir=self.model_dir)
        else:
            print("Initializing PaddleOCR TextDetection with default model (no custom det_model_dir provided).")
            self.ocr = TextDetection() # Will download default detection model if not present

    def detect(self, image_path: str):
        """
        Detect text in an image using the DBNet model (via PaddleOCR's TextDetection).
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            list: Detected text boxes and their corresponding scores.
                  Format: [ [[box_coords], (text, confidence)], ... ]
                  For TextDetection, it's usually just box_coords.
        """
        # The TextDetection class's predict method takes an image path or numpy array.
        result = self.ocr.predict(image_path)
        # The result from TextDetection.predict is typically a list of bounding boxes.
        # Example format: [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ... ]
        return result
