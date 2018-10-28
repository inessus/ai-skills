import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from datetime import datetime


#Device configureation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start = datetime.now()
loopstart = start

# Hyper-parameters
latent_size = 64 #
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = 'sample'

# Create a directory if no exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels，把值标准化到[0,1]之间，均值为0.5, 方差0.5
                        std=(0.5, 0.5, 0.5))])

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)
# Data Loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                         batch_size=batch_size,
                                         shuffle=True)

# Discriminator  判别模型
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())   # [0, 1]之间的数据，正好使用Sigmoid

# Generator  # 生成模型
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()) #(-1, 1)之间的数据

# Device setting
D = D.to(device)
G = G.to(device)

#Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


# Tanh值域 向 Sigmoid值域 转换
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# 重置优化梯度
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    
# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):  # 整批数据通过的次数
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device) #原始分类标签丢弃，只取图片数据
        
        # Create the labels which are later used as input for teh BCE loss
        real_labels = torch.ones(batch_size, 1).to(device) # 原始图片都是真
        fake_labels = torch.zeros(batch_size, 1).to(device) # 生成图片都是假
        
        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        
        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels) # D学习真的图片 从目不识丁到辨真
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        
        z = torch.randn(batch_size, latent_size).to(device) #这个假造的好自然，肯定有改进
        fake_images = G(z)                            
        outputs = D(fake_images)                      # D学习假的图片 从目不识丁到辨假
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        #Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()                             # 目标相同，一起搞
        d_optimizer.step()                            # D优化
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        
        # Compute loss with fake image
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We trainer G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        
        g_loss = criterion(outputs, real_labels) # 本来是假的数据，我当成真的最小化误差，也就是当成真的最大化误差，这个解释我都觉得不妥
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')