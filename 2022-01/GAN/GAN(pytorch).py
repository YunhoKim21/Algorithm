from doctest import OutputChecker
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
from tensorflow.keras.datasets import mnist
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

class Discriminator(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.img_size = args['img_size'] #1x28x28

        if 'device' in args:
            self.device = args['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_size)), 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class Generator(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.nz = args['nz']
        self.img_size = args['img_size']
        if 'device' in args:
            self.device = args['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = nn.Sequential(
            nn.Linear(self.nz, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256), 
            nn.LeakyReLU(), 
            nn.Linear(256, 512), 
            nn.LeakyReLU(), 
            nn.Linear(512, int(np.prod(self.img_size)))
        )
    
    def get_noise(self):
        return torch.randn(128, self.nz).to(self.device)

    def forward(self, x):
        img = self.model(x)
        return img.view(img.size(0), *self.img_size)

    def Generate(self):
        return self.forward(self.get_noise())

class GAN:
    def __init__(self, train_data, args) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data = train_data
        self.G = Generator(args).to(self.device)
        self.D = Discriminator(args).to(self.device)
        self.batch_size = 10

    def evaluate(self):
        sample_data = random.sample(self.train_data, self.batch_size)
        values = list(map(lambda x : np.log(self.D(x[0])), sample_data))
        print(values)
        '''
        
        sample_data = list(map(lambda x : np.log(self.D(x)), sample_data))
        print(sample_data)'''

    def fit(self, data, epochs = 10, k = 5):
        for epoch in range(epochs):
            for i in range(k):
                pass
        
###hyper params###
BATCH_SIZE = 128
IMG_SIZE = 28
lr = 0.0002
beta1 = 0.5
EPOCHS = 1000
NZ = 100

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    g_args = {'nz':NZ, 'img_size':(1, IMG_SIZE, IMG_SIZE)}
    G = Generator(g_args).to(device)
    
    d_args = {'img_size':(1, IMG_SIZE, IMG_SIZE)}
    D = Discriminator(d_args).to(device)

    mnist_train = dsets.MNIST(root = 'MNIST_data/', train = True, transform = transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    optimizer_D = optim.Adam(D.parameters(), lr = lr, betas = (beta1, 0.999))
    optimizer_G = optim.Adam(G.parameters(), lr = lr, betas = (beta1, 0.999))

    _count = 0
    for epoch in tqdm(range(EPOCHS)):
        for data, i in data_loader:
            data = data.to(device)
            _count += 1
            D.zero_grad()

            label = torch.full((BATCH_SIZE, ), real_label, dtype=torch.float, device=device)
            
            output = D(data).view(-1)

            D_cost_real = criterion(output, label)
            D_cost_real.backward()
            
            noise = torch.randn(BATCH_SIZE, NZ, device=device)
            fake = G(noise)
            label_fake = torch.full((BATCH_SIZE, ), fake_label, dtype=torch.float, device=device)
            output_fake = D(fake).view(-1)
            D_cost_fake = criterion(output_fake, label_fake)
            D_cost_fake.backward(retain_graph = True)

            D_cost = D_cost_real + D_cost_fake
            optimizer_D.step()


            G.zero_grad()
            label.fill_(real_label)
            output = D(fake).view(-1)
            G_cost = criterion(output, label)
            G_cost.backward()
            optimizer_G.step()
            #print(D_cost, G_cost)

        if (epoch+1) % 20 == 0:
            images = G.Generate().cpu().detach().numpy().reshape(-1, 28, 28)
            size = 10
            fig = plt.figure(figsize = (size, size))
            for i in range(size ** 2):
                plt.subplot(size, size, i + 1)
                plt.imshow(images[i], cmap = 'gray')
                plt.axis('off')
            plt.savefig('torch_gan_epoch={}.png'.format(epoch))


if __name__ == '__main__':
    main()