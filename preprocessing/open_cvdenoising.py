import cv2
import numpy as np
from PIL import Image # Added PIL import for flexibility

def opencv_denoising(img_input):
    """
    Performs a series of OpenCV-based denoising and enhancement steps on an image.
    
    Args:
        img_input (PIL.Image or np.ndarray): The input image. Can be PIL Image (RGB or L)
                                             or OpenCV NumPy array (BGR or grayscale).
    Returns:
        np.ndarray: The processed image as an OpenCV NumPy array (BGR or grayscale,
                    depending on original input and processing steps).
    """
    # Convert PIL Image to OpenCV numpy array if needed
    if isinstance(img_input, Image.Image):
        # Convert to RGB first for consistency, then BGR for OpenCV
        img = np.array(img_input.convert('RGB')) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy() # Work on a copy to avoid modifying original input
    else:
        raise TypeError("Input to opencv_denoising must be a PIL Image or NumPy array.")

    # Detect if original image was grayscale to decide final output format
    is_grayscale_input = (len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1))
    
    # Ensure img is 3-channel (BGR) for some operations if it's grayscale
    if is_grayscale_input:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Step 1: Salt & pepper noise removal (Median Blur)
    img = cv2.medianBlur(img, 3)

    # Step 2: Gaussian noise removal (Non-local Means Denoising)
    # Ensure img is 8-bit unsigned integer (CV_8U) for fastNlMeansDenoising
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) # Assuming input might be float [0,1] or similar

    # Use fastNlMeansDenoisingColored for 3-channel images
    new_img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    # Step 3: Blur detection (nested function for internal use)
    def is_blurry(img_check, threshold=100):
        # Convert to grayscale for Laplacian if not already
        if len(img_check.shape) == 3:
            gray_check = cv2.cvtColor(img_check, cv2.COLOR_BGR2GRAY)
        else:
            gray_check = img_check
        return cv2.Laplacian(gray_check, cv2.CV_64F).var() < threshold

    # Step 4: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(new_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    new_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) # Update new_img with CLAHE applied

    # Step 5: Smart Resize (nested function for internal use)
    def smart_resize(img_resize, max_size=640, min_size=240):
        h, w = img_resize.shape[:2]
        if max(h, w) >= max_size:
            return img_resize
        scale = min(max_size / max(h, w), min_size / min(h, w))
        if scale < 1.0: # Don't downscale
            return img_resize
        elif scale > 2.0: # Limit over-enlargement to 2x
            scale = 2.0
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img_resize, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    new_img = smart_resize(new_img) # Apply smart resize

    # Step 6: Apply sharpening only if blurry (after resize)
    if is_blurry(new_img):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        new_img = cv2.filter2D(new_img, -1, kernel)

    # Final conversion to grayscale if original input was grayscale
    if is_grayscale_input and len(new_img.shape) == 3:
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    
    return new_img # Returns a NumPy array (BGR or grayscale)
