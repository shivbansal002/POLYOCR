import os
import cv2
import numpy as np
from PIL import Image
from preprocess import add_gaussian_noise, add_salt_pepper_noise, add_light_blur, add_jpeg_artifacts, add_gaussian_blur_opencv, add_motion_blur
from models.open_cvdenoising import opencv_denoising
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from models.dncnn import DnCNN
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),        
        )  
        
        #Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 

class denoisindataset(Dataset):
        def __init__(self, img_folder, transform=None):
            self.img_folder = img_folder
            self.transform = transform
            self.img_files = [os.path.join(self.img_folder,f) for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')]


        def __len__(self):
            return len(self.img_files)

        def __getitem__(self, idx):
            img_path = self.img_files[idx]
            img = Image.open(img_path).convert('L')
            clean_img = self.transform(img)
            # Apply transformations
              
            # Add noise

            noisy_img = add_salt_pepper_noise(clean_img, amount=0.05, salt_vs_pepper=0.5)
            noisy_img = add_gaussian_noise(noisy_img, mean=0, sigma=0.1)
            noisy_img = add_light_blur(noisy_img, radius=1)
            noisy_img = add_jpeg_artifacts(noisy_img, quality=30)
            # noisy_img = add_gaussian_blur_opencv(noisy_img)
            # noisy_img = add_motion_blur(noisy_img)
            # Convert to numpy array
            noisy_img = np.array(noisy_img)
            clean_img = np.array(clean_img)

            if noisy_img.ndim != 2:
                raise ValueError(f"Noisy image has invalid shape: {noisy_img.shape}")
            if clean_img.ndim != 2:
                raise ValueError(f"Clean image has invalid shape: {clean_img.shape}")


            return noisy_img,clean_img
        
# def main():
#     def ensure_grayscale_2d(img):
#         if isinstance(img, Image.Image):
#             img = img.convert('L')  # Ensure grayscale mode
#             img = np.array(img)

#         if img.ndim == 3 and img.shape[-1] == 1:
#             img = img.squeeze(-1)

#         if img.ndim != 2:
#             raise ValueError(f"Image must be 2D, got shape: {img.shape}")

#         return img
def main():   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    state_dict = torch.load(r"C:\Users\jayv5\OneDrive\study\iiti soc\OCR\preprocessing\datasets\autoencoder.pth", map_location = device)
    model = Autoencoder().to(device)
    model.load_state_dict(state_dict)

    transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])
    

    img_folder=r"C:\Users\jayv5\OneDrive\study\iiti soc\OCR\preprocessing\datasets\mixed_dataset"
    dataset= denoisindataset(img_folder,transform=transform)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True,num_workers=6)
    psnr1 = []
    psnr2 = []
    ssim = []
    for batch_idx,(noisy_img, clean_img) in enumerate(dataloader):
        # print(noisy_img.shape,clean_img.shape)
        noisy_img = noisy_img.numpy()
        clean_img = clean_img.numpy()
        # print(f"Batch {batch_idx} - Noisy Image Shape: {noisy_img.shape}, Clean Image Shape: {clean_img.shape}")
        for i in range(noisy_img.shape[0]):
            noisy_imag = transforms.ToTensor()(noisy_img[i]).to(device)
            noisy_imag = noisy_imag.unsqueeze(0)
            
            denoised_img=noisy_imag - model.forward(noisy_imag)
            clean_imag = clean_img[i]
            noisy_im = noisy_img[i]
            denoised_img = denoised_img.squeeze(0)  # Remove batch dimension
            denoised_imag = denoised_img.cpu().detach().numpy()
            # print(f"noisy_img is None: {noisy_im is None}")
            # print(f"clean_imag is None: {clean_imag is None}")
            # print(f"denoised_img is None: {denoised_img is None}")

            # print(f"noisy_img shape: {noisy_img.shape}, clean_imag shape: {clean_imag.shape}, denoised_img shape: {denoised_img.shape}")
            # noisy_img = ensure_grayscale_2d(noisy_img)
            # clean_imag = ensure_grayscale_2d(clean_imag)
            # denoised_img = ensure_grayscale_2d(denoised_img)
            denoised_imag = denoised_imag.squeeze(0)  # Remove single-dimensional entries
            # print(f"noisy_im shape: {noisy_im.shape}, clean_imag shape: {clean_imag.shape}, denoised_imag shape: {denoised_imag.shape}" )
            # print(f"Type: {clean_imag.dtype}")
            # print(f"type: {noisy_im.dtype} ")
            # print(f"Type: {denoised_imag.dtype}")
        
            denoised_imag = (denoised_imag * 255).astype(np.uint8)

            psnr2.append(peak_signal_noise_ratio(clean_imag, denoised_imag))
            psnr1.append(peak_signal_noise_ratio(clean_imag, noisy_im))
            ssim.append(structural_similarity(clean_imag, denoised_imag))
        print(f"btach{batch_idx} PSNR_denoised: {np.mean(psnr1)}, PSNR_noised: {np.mean(psnr2)}, SSIM: {np.mean(ssim)}")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # Optional, for compatibility
    main()      
