
from paddleocr import TextDetection
from PIL import Image
from torchvision import transforms
# from paddleocr import PaddleOCR
import scipy


# Access a specific variable
# your_array = data['variable_name']
# print(your_array.shape)
# print(your_array)

# ocr = PaddleOCR(det_algorithm='EAST')  # 'DB', 'SAST', 'PSE', 'EAST'
# result = ocr.predict(r'g', det=True, rec=False)
# print(result)



class DBNet:
    def __init__(self, model_dir: str = None):
        """
        Initialize the DBNet model.
        
        Args:
            model_dir (str): Path to the directory containing the DBNet model files.
        """
        self.model_dir = model_dir
        self.ocr = TextDetection(model_dir= self.model_dir)

    def detect(self, image_path: str):
        """
        Detect text in an image using the DBNet model.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            list: Detected text boxes and their corresponding scores.
        """
        # im_path= Image.open(image_path)
        # im_path = transforms.ToTensor()(im_path).unsqueeze(0)
        # imag_path = im_path.numpy().squeeze(0)

        result = self.ocr.predict(image_path)
        return result
    


 
