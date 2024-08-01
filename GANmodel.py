import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST  # 假设我们使用MNIST数据集

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, img_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # 输入层: latent_dim
            nn.Linear(latent_dim, 128 * (img_size // 4) * (img_size // 4)),
            nn.BatchNorm1d(128 * (img_size // 4) * (img_size // 4)),
            nn.LeakyReLU(0.2, inplace=True),
            # 转换层
            nn.View(-1, 128, img_size // 4, img_size // 4),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),
            nn.Tanh()  # 输出层使用Tanh确保输出值在[-1, 1]之间
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, img_channels, img_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 输入层
            nn.Conv2d(img_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 转换层
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出层
            nn.Flatten(),
            nn.Linear(128 * (img_size // 4) * (img_size // 4), 1),
            nn.Sigmoid()  # 输出层使用Sigmoid，输出一个介于0和1之间的值
        )

    def forward(self, img):
        return self.model(img)


class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def train(self, dataloader, num_epochs, latent_dim, batch_size):
        # 优化器
        optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            for real_images, _ in dataloader:
                real_images = real_images.to(device)
                batch_size = real_images.size(0)

                # 训练判别器
                labels_real = torch.ones(batch_size, 1).to(device)
                labels_fake = torch.zeros(batch_size, 1).to(device)

                outputs = self.discriminator(real_images)
                loss_D_real = criterion(outputs, labels_real)

                z = torch.randn(batch_size, latent_dim).to(device)
                fake_images = self.generator(z)
                outputs = self.discriminator(fake_images.detach())  # detach以避免在反向传播时影响生成器
                loss_D_fake = criterion(outputs, labels_fake)

                loss_D = (loss_D_real + loss_D_fake) / 2
                self.discriminator.zero_grad()
                loss_D.backward()
                optimizer_D.step()

                # 训练生成器
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_images = self.generator(z)
                outputs = self.discriminator(fake_images)
                loss_G = criterion(outputs, labels_real)

                self.generator.zero_grad()
                loss_G.backward()
                optimizer_G.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}')


# 实例化GAN
latent_dim = 64
img_size = 28
img_channels = 1

generator = Generator(latent_dim, img_channels, img_size).to(device)
discriminator = Discriminator(img_channels, img_size).to(device)
gan = GAN(generator, discriminator)

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练GAN
gan.train(dataloader, num_epochs=50, latent_dim=latent_dim, batch_size=64)

# 注意：这里只是一个简单的训练流程示例。
# 在实际应用中，您可能需要添加更多的功能，比如保存模型、加载模型、可视化结果等。
# 此外，GAN的训练通常很不稳定，可能需要调整超参数、网络结构或使用更复杂的技巧（如特征匹配、历史平均等）来稳定训练过程。
