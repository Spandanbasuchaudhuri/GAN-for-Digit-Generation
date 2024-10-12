import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 32
lr = 0.0002
epochs = 100
noise_dim = 100

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

mnist_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.init_size = 7
        self.fc = nn.Linear(noise_dim, 128 * self.init_size ** 2)
        
        self.block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc(x).view(x.size(0), 128, self.init_size, self.init_size)
        return self.block(out)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)

generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

optim_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

def train_GAN(generator, discriminator, data_loader, epochs):
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            optim_D.zero_grad()
            real_outputs = discriminator(real_images)
            real_loss = criterion(real_outputs, real_labels)

            z = torch.randn(batch_size, noise_dim).to(device)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)

            D_loss = real_loss + fake_loss
            D_loss.backward()
            optim_D.step()

            optim_G.zero_grad()
            fake_outputs = discriminator(fake_images)
            G_loss = criterion(fake_outputs, real_labels)
            G_loss.backward()
            optim_G.step()

        print(f'Epoch [{epoch+1}/{epochs}] | D Loss: {D_loss.item():.4f} | G Loss: {G_loss.item():.4f}')

        if (epoch+1) % 10 == 0:
            visualize_comparison(real_images, fake_images)

def visualize_comparison(real_images, fake_images):
    real_images = (real_images + 1) / 2
    fake_images = (fake_images + 1) / 2

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axs[0, i].imshow(real_images[i].squeeze().cpu().detach().numpy(), cmap='gray')
        axs[0, i].set_title('Real')
        axs[0, i].axis('off')

        axs[1, i].imshow(fake_images[i].squeeze().cpu().detach().numpy(), cmap='gray')
        axs[1, i].set_title('Fake')
        axs[1, i].axis('off')

    plt.show()

train_GAN(generator, discriminator, data_loader, epochs)
