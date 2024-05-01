# We start by importing the libraries we'll use today
import numpy as np
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ToTensor
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import time

trainingdata = torchvision.datasets.FashionMNIST('./FashionMNIST/',train=True,download=True,transform=torchvision.transforms.ToTensor())
testdata = torchvision.datasets.FashionMNIST('./FashionMNIST/',train=False,download=True,transform=torchvision.transforms.ToTensor())

print(len(trainingdata), len(testdata))
image, label = trainingdata[0]
print(image.shape, label)
print(image.squeeze().shape)

trainDataLoader = torch.utils.data.DataLoader(trainingdata,batch_size=64,shuffle=True)
testDataLoader = torch.utils.data.DataLoader(testdata,batch_size=64,shuffle=False)

print(len(trainDataLoader))
print(len(testDataLoader))

print(len(trainDataLoader) * 64) # batch_size from above
print(len(testDataLoader) * 64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for hw report purpose
def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


# generatorr
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 256*7*7)  # Map from noise vector to 256x7x7
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=1, padding=2),  # 28x28 -> 28x28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        x = self.conv_layers(x)
        return x

# discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.3),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),
            nn.Dropout2d(0.3)
        )
        self.fc = nn.Linear(128*7*7, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 128*7*7)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCEWithLogitsLoss()
lr = 1e-4
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# for graph purpose to obtain the result
discriminator_losses = []
generator_losses = []

real_probabilities = []
fake_probabilities = []

num_epochs = 50
for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(trainDataLoader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        #train discriminator
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # real image and loss
        output_real = discriminator(real_images)
        loss_real = criterion(output_real, real_labels)
        avg_real_prob = torch.sigmoid(output_real).mean().item()
        

        # create fake images, and then computet the loss
        noise = torch.randn(batch_size, 100).to(device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())
        avg_fake_prob = torch.sigmoid(output_fake).mean().item()
        loss_fake = criterion(output_fake, fake_labels)

        d_loss = loss_real + loss_fake

        
        d_loss.backward()
        optimizer_D.step()

        #train generator
        generator.zero_grad()
        output = discriminator(fake_images)
        g_loss = criterion(output, real_labels)
        g_loss.backward()
        optimizer_G.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(trainDataLoader)}], "
                  f"Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}")

    discriminator_losses.append(d_loss.item())
    generator_losses.append(g_loss.item())
    real_probabilities.append(avg_real_prob)
    fake_probabilities.append(avg_fake_prob)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
    save_checkpoint(generator, optimizer_G, epoch, g_loss.item(), checkpoint_path)

    # generate fake images using trained model
    # save the images to the image folder
    save_dir = "images"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        fake_samples = generator(torch.randn(64, 100).to(device))
        fake_grid = torchvision.utils.make_grid(fake_samples.cpu(), nrow=8, normalize=True)

    image_path = os.path.join(save_dir, f"generated_image{epoch+1}.png")
    torchvision.utils.save_image(fake_grid, image_path)

    print(f"Generated image saved at: {image_path}")

os.makedirs("images", exist_ok=True)
with torch.no_grad():
    # eval mode to begin with
    generator.eval()  
    noise = torch.randn(64, 100).to(device) 
    fake_images = generator(noise).to(device)
    fake_grid = torchvision.utils.make_grid(fake_images.cpu(), nrow=8, normalize=True)
plt.imshow(fake_grid.permute(1, 2, 0)) 
plt.savefig("images/generated_images.png")  
plt.show()
plt.close()

plt.plot(discriminator_losses, label="Discriminator Loss")
plt.plot(generator_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Loss in {num_epochs} epochs")
plt.legend()
plt.grid(True)

report_dir = "report"
os.makedirs(report_dir, exist_ok=True)
plt.savefig(os.path.join(report_dir, "loss_curves.png"))
plt.show()
plt.close()

plt.plot(real_probabilities, label="Average Real Probability")
plt.plot(fake_probabilities, label="Average Fake Probability")
plt.xlabel("Epoch")
plt.ylabel("Probability")
plt.title(f"Discriminator Real and Fake image detection accuracy in {num_epochs} epochs")
plt.legend()
plt.grid(True)

report_dir = "report"
os.makedirs(report_dir, exist_ok=True)
plt.savefig(os.path.join(report_dir, "discriminator_performance.png"))
plt.show()
plt.close()