import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot  as plt


           




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
    
def fun():


    if torch.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu') 
        
    print('using device:', device)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081))   
        ])
    
    train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    def add_noise(img, noise_factor=0.7, device=device):
        noisy_img = img + noise_factor * torch.randn(*img.shape).to(device)
        noisy_img = torch.clip(noisy_img, 0., 1.)
        return noisy_img


    img, _ = next(iter(train_loader))

    plt.figure(figsize=(8,4))
    for i, noise_factor in enumerate(np.linspace(0.1, 1, 10)):
        plt.subplot(2, 5, 1+i)
        plt.imshow(add_noise(img[0], noise_factor=noise_factor, device='cpu').squeeze(), cmap='gray')
        plt.title(f'Noise factor: {noise_factor:.2f}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
        
    dummy_input = torch.randn(1, 1, 28, 28)
    dummy_model = Autoencoder()
    dummy_model(dummy_input).shape
        
    def calc_shape_up_conv(in_shape, kernel_size, stride, padding, output_padding):
        return (in_shape - 1) * stride - 2 * padding + kernel_size + output_padding

    def calc_shape_down_conv(in_shape, kernel_size, stride, padding):
        return (in_shape +2 * padding - kernel_size) // stride +1

    print('Shape after first conv:', calc_shape_down_conv(28, 3, 2, 1))
    print('Shape after second conv:', calc_shape_down_conv(14, 3, 2, 1))

    print('Shape after first up-conv:', calc_shape_up_conv(7, 3, 2, 1, 1))
    print('Shape after second up-conv:', calc_shape_up_conv(14, 3, 2, 1, 1))

        
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    EPOCHS = 15

    def visualize_outputs(model, data_loader, device, title=None):
        model.eval()
        with torch.no_grad():
            images, _= next(iter(data_loader))
            images = images.to(device)
            noise_factor = np.random.rand()
            noisy_images = add_noise(images, noise_factor=noise_factor)
            outputs = model(noisy_images)
            
            #Plot original, noisy and denoised images
            plt.figure(figsize=(9, 3))
            for i in range(5):
                
                #Original
                plt.subplot(3, 5, i+1)
                plt.imshow(images[i].cpu().squeeze(), cmap='gray')
                plt.axis('off')
                
                #noisy
                plt.subplot(3, 5, i+6)
                plt.imshow(noisy_images[i].cpu().squeeze(), cmap='gray')
                plt.axis('off')
                
                #Denoised
                plt.subplot(3, 5, i+11)
                plt.imshow(outputs[i].cpu().squeeze(), cmap='gray')
                plt.axis('off')            
                
            if title:
                plt.suptitle(title)
            plt.show()
                    
    train_losses =[]
    test_losses = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, _ in train_loader:
            images = images.to(device)
            noise_factor = np.random.rand()
            noisy_images =add_noise(images, noise_factor)
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader) 
        train_losses.append(avg_train_loss)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                noise_factor = np.random.rand()
                noisy_images = add_noise(images, noise_factor)
                outputs = model(noisy_images)
                loss = criterion(outputs, images)
                
                test_loss += loss.item()
                
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f'Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss} | Test_Loss: {avg_test_loss}')
        
        if epoch == 0 or (epoch + 1) % 5 == 0:
            visualize_outputs(model, train_loader, device, title=f'Epoch {epoch+1}')
        
        #plot losses 
        plt.figure()
        plt.plot(train_losses, label='Train Loss')        
        plt.plot(test_losses, label='Test Loss') 
        plt.legend()
        plt.title('Losses')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.show()
                        
                        
    visualize_outputs(model, test_loader, device, title='Test data')

    #save model state_dict                      
    torch.save(model.state_dict(), 'models/autoencoder_0.pth')

    #Load model
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load('models/autoencoder_0.pth'))

    pass
if __name__ == "__main__":
    fun()