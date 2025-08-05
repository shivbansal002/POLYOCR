import cv2
import os
import numpy as np
from PIL import Image
from PIL import ImageFilter
import io


#folder containing images
# image_folder=os.path.join('images','IIIT5K','test')

# #output folder for processed images
# output_dir='outputs'
# os.makedirs(output_dir,exist_ok=True)

# #supported image extensions
# valid_extensions=('.jpg','.jpeg','.png','.bmp','.tiff')

# #list all image files in the folder with valid extensions
# image_files=[f for f in os.listdir(image_folder)if f.lower().endswith(valid_extensions)]
# print(f"Found {len(image_files)}images.")

# for idx, image_name in enumerate(image_files):
#     image_path=os.path.join(image_folder,image_name)
#     print(f"Processing:{image_path}")

#     #load image
#     image=cv2.imread(image_path)
#     if image is None:
#         print(f"Warning:Unable to load image{image_path}.Skipping.")
#         continue

#     #resize to 256*256
#     resized=cv2.resize(image,(256,256))

#     #convert to grayscale
#     gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

#     #Denoising
#     denoised=cv2.fastNlMeansDenoising(gray,None,h=30,templateWindowSize=7,searchWindowSize=21)

#     #Adaptive threholding(binarization)
#     binary=cv2.adaptiveThreshold(
#         denoised,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11,
#         2
#     )

#     #Save processed image
#     cv2.imwrite(os.path.join(output_dir,f"processed_{idx}_resized.jpg"),resized)
#     cv2.imwrite(os.path.join(output_dir,f"processed_{idx}_gray.jpg"),gray)
#     cv2.imwrite(os.path.join(output_dir,f"processed_{idx}_denoised.jpg"),denoised)
#     cv2.imwrite(os.path.join(output_dir,f"processed_{idx}_binary.jpg"),binary)

#     #to display
#     cv2.imshow("Final Preprocessed Image",binary)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     print("Processing Complete!")
    
def add_gaussian_noise(img, mean=0, sigma=0.1):
    img = np.array(img).astype(np.float32) / 255.0  # normalize
    noise = np.random.normal(mean, sigma, img.shape)
    noisy_img = np.clip(img + noise, 0, 1)
    noisy_img = noisy_img.squeeze()  # remove single-dimensional entries
    if noisy_img.ndim == 3 and noisy_img.shape[-1] == 1:
        noisy_img = noisy_img.squeeze(-1)

    if noisy_img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape: {noisy_img.shape}")

    return Image.fromarray((noisy_img * 255).astype(np.uint8))
import numpy as np
from PIL import Image

def add_salt_pepper_noise(img, amount=0.2, salt_vs_pepper=0.5):
    if isinstance(img, Image.Image):
        img = np.array(img)

    noisy = img.copy()
    h, w = noisy.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt noise (white pixels)
    num_salt = int(salt_vs_pepper * num_pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in (h, w)]
    noisy[coords[0], coords[1]] = 255

    # Pepper noise (black pixels)
    num_pepper = num_pixels - num_salt
    coords = [np.random.randint(0, i - 1, num_pepper) for i in (h, w)]
    noisy[coords[0], coords[1]] = 0
    if noisy.ndim == 3 and noisy.shape[-1] == 1:
        noisy = noisy.squeeze(-1)
    if noisy.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape: {noisy.shape}")

    return Image.fromarray(noisy.astype(np.uint8))


def add_light_blur(img, radius=1):
    noisy_img = img.filter(ImageFilter.GaussianBlur(radius))
    
    if noisy_img.mode != 'L':
        noisy_img = noisy_img.convert('L')  # Ensure grayscale
    noisy_img = np.array(noisy_img)
    if noisy_img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape: {noisy_img.size}")
    noisy_img = Image.fromarray((noisy_img ).astype(np.uint8))
    return noisy_img.copy()



def add_jpeg_artifacts(img, quality=5):
    
    if not isinstance(img, Image.Image):
        raise ValueError("Input must be a PIL.Image")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_img = Image.open(buffer).convert("L")  # Convert forces decoding
    if compressed_img.mode != 'L':
        compressed_img = compressed_img.convert('L')    
    compressed_img = np.array(compressed_img)
    if compressed_img.ndim != 2:    
        raise ValueError(f"Expected 2D grayscale image, got shape: {compressed_img.size}")
    compressed_img = Image.fromarray((compressed_img).astype(np.uint8))
    return compressed_img.copy()
    


def add_gaussian_blur_opencv(img, ksize=3):
    """Add standard Gaussian blur using OpenCV."""
    if isinstance(img, Image.Image):  # Convert from PIL to OpenCV
        img = np.array(img)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return Image.fromarray(blurred)

def add_motion_blur(img, kernel_size=5, angle='horizontal'):
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Create a horizontal or vertical motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    if angle == 'horizontal':
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    elif angle == 'vertical':
        kernel[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    else:
        raise ValueError("angle must be 'horizontal' or 'vertical'")

    kernel = kernel / kernel_size
    blurred = cv2.filter2D(img, -1, kernel)
    return Image.fromarray(blurred)
import random

def add_random_blur(img):
    if random.random() < 0.5:
        return add_gaussian_blur_opencv(img, ksize=random.choice([5, 7]))
    else:
        return add_motion_blur(img, kernel_size=random.choice([7, 9]), angle=random.choice(['horizontal', 'vertical']))


