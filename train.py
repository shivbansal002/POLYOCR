import os
import torch 
from models.DnCNN import DnCNN
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from preprocessing.noise_adding import add_gaussian_noise, add_salt_pepper_noise, add_light_blur, add_jpeg_artifacts
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from datasets import load_dataset # Import the datasets library

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Global paths and configurations for training ---
# CRITICAL: Replace this with your desired online dataset name if you want to use a different one.
ONLINE_DATASET_NAME = "HuggingFaceDatasets/wikipedia_images" 
OUTPUT_BASE_DIR = "/content/drive/MyDrive/PolyOCR_Train_Outputs/" # Base folder for all training runs
RUN_NAME = datetime.now().strftime("DnCNN_Run_Online_%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, RUN_NAME)
CHECKPOINT_DIR = os.path.join(RUN_OUTPUT_DIR, "checkpoints")
LOGS_DIR = os.path.join(RUN_OUTPUT_DIR, "logs")

# Hyperparameters
NUM_EPOCHS = 200 # Increased for a more thorough training
BATCH_SIZE = 32 # Increased batch size for faster training
LEARNING_RATE = 1e-4
SAVE_CHECKPOINT_EVERY = 10 # Save a checkpoint every N epochs

# Create output directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(LOGS_DIR)

class DenoisingDataset(Dataset):
        def __init__(self, dataset_name, transform):
            # Load the dataset from Hugging Face
            self.dataset = load_dataset(dataset_name, split='train')
            self.transform = transform
            
        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            try:
                # Get the image and convert to PIL Image
                # Assuming the dataset has an 'image' feature
                image_data = self.dataset[idx]['image']
                # The image might be a dict or a PIL Image directly
                if isinstance(image_data, dict):
                    # Handle cases where the image is a nested object
                    if 'path' in image_data:
                        c_image = Image.open(image_data['path']).convert('L')
                    elif 'bytes' in image_data:
                        from io import BytesIO
                        c_image = Image.open(BytesIO(image_data['bytes'])).convert('L')
                    else:
                        raise ValueError("Unsupported image data format.")
                else: # Assume it's a PIL Image
                    c_image = image_data.convert('L')
                
                # Apply noise functions
                noisy_img = add_salt_pepper_noise(c_image, amount=0.05, salt_vs_pepper=0.5)
                noisy_img = add_gaussian_noise(noisy_img, mean=0, sigma=0.1)
                noisy_img = add_light_blur(noisy_img, radius=1)
                noisy_img = add_jpeg_artifacts(noisy_img, quality=30) 

                c_image = self.transform(c_image)
                n_image = self.transform(noisy_img)
                if c_image.shape[-2] < 2 or c_image.shape[-1] < 2:
                    return None 
                return n_image, c_image
            except Exception as e:
                print(f"Error loading or processing item {idx} from dataset: {e}")
                return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([]) 
    return torch.utils.data.default_collate(batch)

def main():
    print(f"Using device: {device}")
    
    # Check if a checkpoint exists to resume training
    start_epoch = 0
    model = DnCNN().to(device)
    
    # Check for latest checkpoint
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')])
    if checkpoints:
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        print(f"Found latest checkpoint: {latest_checkpoint_path}. Resuming training.")
        state_dict = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        start_epoch = int(checkpoints[-1].split('epoch')[1].split('.')[0])
        print(f"Resuming from epoch {start_epoch}.")
    else:
        print("No checkpoints found. Starting training from scratch.")
        pretrained_model_path = "/content/drive/MyDrive/PolyOCR_Train_Outputs/Run_YYYYMMDD_HHMMSS/dncnn_finetunedfinal3.pth"
        if os.path.exists(pretrained_model_path):
            print(f"Loading pre-trained model from: {pretrained_model_path}")
            state_dict = torch.load(pretrained_model_path, map_location=device)
            model.load_state_dict(state_dict)


    transform = transforms.Compose(
        [
            transforms.Resize((256, 512)),
            transforms.ToTensor()
        ]
    )
    
    # Use the online dataset name here
    dataset = DenoisingDataset(ONLINE_DATASET_NAME, transform)
    
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    dataloader_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    dataloader_val = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    model.train()

    print(f"Starting training for {NUM_EPOCHS - start_epoch} epochs...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        total_train_loss = 0.0
        
        # Training Loop
        for batch_idx, (n_image, c_image) in enumerate(dataloader_train):
            if n_image.nelement() == 0: continue
            
            n_image = n_image.to(device)
            c_image = c_image.to(device)

            optimizer.zero_grad()
            residual_image = model(n_image)
            denoised_image = n_image - residual_image

            loss = criterion(denoised_image, c_image)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(dataloader_train)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(dataloader_train)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        print(f"End of Epoch {epoch+1}. Avg Train Loss: {avg_train_loss:.4f}")

        # Validation Loop
        model.eval()
        total_val_loss = 0.0
        psnr_scores_denoised = []
        psnr_scores_noisy = []
        ssim_scores = []
        with torch.no_grad():
            for n_test_image, c_test_image in dataloader_val:
                if n_test_image.nelement() == 0: continue
                
                n_test_image = n_test_image.to(device)
                c_test_image = c_test_image.to(device)

                residual_image = model(n_test_image)
                denoised_image = n_test_image - residual_image
                loss_val = criterion(denoised_image, c_test_image)
                total_val_loss += loss_val.item()

                output_np = denoised_image.cpu().numpy().squeeze(1)
                c_image_np = c_test_image.cpu().numpy().squeeze(1)
                n_image_np = n_test_image.cpu().numpy().squeeze(1)

                for i in range(output_np.shape[0]):
                    current_c_img = c_image_np[i] if c_image_np[i].ndim == 2 else c_image_np[i][0]
                    current_output_img = output_np[i] if output_np[i].ndim == 2 else output_np[i][0]
                    current_n_img = n_image_np[i] if n_image_np[i].ndim == 2 else n_image_np[i][0]
                    psnr_scores_denoised.append(peak_signal_noise_ratio(current_c_img, current_output_img, data_range=1.0))
                    psnr_scores_noisy.append(peak_signal_noise_ratio(current_c_img, current_n_img, data_range=1.0))
                    ssim_scores.append(structural_similarity(current_c_img, current_output_img, data_range=1.0))
        
        avg_val_loss = total_val_loss / len(dataloader_val)
        avg_psnr_denoised = np.mean(psnr_scores_denoised)
        avg_psnr_noisy = np.mean(psnr_scores_noisy)
        avg_ssim = np.mean(ssim_scores)

        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('PSNR/denoised', avg_psnr_denoised, epoch)
        writer.add_scalar('PSNR/noisy', avg_psnr_noisy, epoch)
        writer.add_scalar('SSIM/denoised', avg_ssim, epoch)

        print(f"Validation for Epoch {epoch+1}: Avg Loss: {avg_val_loss:.4f}, "
              f"Avg PSNR (Denoised): {avg_psnr_denoised:.2f}, Avg SSIM: {avg_ssim:.4f}")

        # Save checkpoint
        if (epoch + 1) % SAVE_CHECKPOINT_EVERY == 0 or (epoch + 1) == NUM_EPOCHS:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved for Epoch {epoch+1} to {checkpoint_path}")

    # Save final model state
    final_model_path = os.path.join(RUN_OUTPUT_DIR, "dncnn_finetuned_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    writer.close()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
