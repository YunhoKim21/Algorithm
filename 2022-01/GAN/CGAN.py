import torch.nn as nn
import torch 
import numpy as np
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.class_size = args['class_size'] #10
        self.img_size = args['img_size'] #1x28x28

        if 'device' in args:
            self.device = args['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_size)) + self.class_size, 512),
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
        self.class_size = args['class_size'] #10
        self.img_size = args['img_size']
        if 'device' in args:
            self.device = args['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = nn.Sequential(
            nn.Linear(self.nz + self.class_size, 128),
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


###hyper params###
BATCH_SIZE = 128
IMG_SIZE = 28
lr = 0.0002
beta1 = 0.5
EPOCHS = 1000
NZ = 100

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    g_args = {'nz':NZ, 'img_size':(1, IMG_SIZE, IMG_SIZE), 'class_size':10}
    G = Generator(g_args).to(device)
    
    d_args = {'img_size':(1, IMG_SIZE, IMG_SIZE), 'class_size':10}
    D = Discriminator(d_args).to(device)

    mnist_train = dsets.MNIST(root = 'MNIST_data/', train = True, transform = transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset=mnist_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    optimizer_D = optim.Adam(D.parameters(), lr = lr, betas = (beta1, 0.999))
    optimizer_G = optim.Adam(G.parameters(), lr = lr, betas = (beta1, 0.999))

    count = 0

    img_fake = None

    for epoch in tqdm(range(1, EPOCHS + 1)):
        for data, label in data_loader:

            D.zero_grad()

            data = data.to(device)
            label = label.to(device)
            
            label = F.one_hot(label, num_classes=10)
            data = data.view(data.size(0), -1)
            final_data = torch.cat([data, label], dim=1)

            D_cost_real = criterion(D(final_data), torch.full((BATCH_SIZE, 1), 1, dtype=torch.float).to(device))
            D_cost_real.backward()

            z = torch.randn(BATCH_SIZE, NZ, device=device)
            y = torch.randint(0, 10, (BATCH_SIZE, )).to(device)
            y = F.one_hot(y, num_classes=10)
            fake = G(torch.cat([z, y], dim=1))
            img_fake = fake
            fake = torch.cat([fake.view(fake.size(0), -1), y], dim = 1)
            D_cost_fake = criterion(D(fake), torch.full((BATCH_SIZE, 1), 0, dtype=torch.float, device=device))
            D_cost_fake.backward(retain_graph = True)

            D_cost = D_cost_real + D_cost_fake
            optimizer_D.step()
            
            G.zero_grad()
            G_cost = criterion(D(fake), torch.full((BATCH_SIZE, 1), 1, dtype=torch.float, device=device))
            G_cost.backward()

            optimizer_G.step()
            count += 1

        if epoch % 20 == 0:
            images = img_fake.cpu().detach().numpy().reshape(-1, 28, 28)
            size = 10
            fig = plt.figure(figsize = (size, size))
            for i in range(size ** 2):
                plt.subplot(size, size, i + 1)
                plt.imshow(images[i], cmap = 'gray')
                plt.axis('off')
            plt.savefig('torch_cgan_epoch={}.png'.format(epoch))



if __name__ == '__main__':
    test()