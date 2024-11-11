from matplotlib import pyplot as plt
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as torch_dataset
from torchvision.utils import make_grid
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions.uniform import Uniform

img_size = 640
data_dir = "data/anime-faces/data"
data_transforms = T.Compose([
    T.Resize(img_size),
    T.CenterCrop(img_size),
    T.ToTensor,
    T.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])
anime_dataset = torch_dataset.ImageFolder(root=data_dir, transform=data_transforms)
dataloader = DataLoader(
        dataset=anime_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )
img_batch = next(iter(dataloader))[0]
combine_img = make_grid(img_batch[:32], normalize=True, padding=2).permute(1,2,0)
plt.figure(figsize=(15, 15))
plt.imshow(combine_img)
plt.show()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.normal_(m.bias.data, 0)

def Conv(n_input, n_output, k_size = 4, stride = 2, padding = 0, bn = False):
    return nn.Sequential(
        nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.2, inplace=False)
    )

def Deconv(n_input, n_output, k_size = 4, stride = 2, padding = 1):
    return nn.Sequential(
        nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.ReLU(inplace=True)
    )

class Generator(nn.Module):
    def __init__(self, z=100, nc=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            Deconv(z, nc * 8, 4, 1, 0),
            Deconv(nc * 8, nc * 4, 4, 2, 1),
            Deconv(nc * 4, nc * 2, 4, 2, 1),
            Deconv(nc * 2, nc, 4, 2, 1),
            nn.ConvTranspose2d(nc,3,4,2,1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.net(input)

class Discriminator(nn.Module):
    def __init__(self, nc=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                3, nc,
                kernel_size=4,
                stride=2,
                padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(nc, nc * 2, 4, 2, 1),
            Conv(nc * 2, nc * 4, 4, 2, 1),
            Conv(nc * 4, nc * 8, 4, 2, 1),
            nn.Conv2d(nc * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, input):
        return self.net(input)

device = torch.device("cuda")
gen_model = Generator()
dis_model = Discriminator()

gen_model.apply(weights_init)
dis_model.apply(weights_init)
dis_model.to(device)
gen_model.to(device)
print("init model")

real_label = 1.
fake_label = 0.
lr = 0.0002
beta1 = 0.5

criterion = nn.BCELoss()
optim_D = optim.Adam(dis_model.parameters(), lr=lr, betas=(beta1, 0.999))
optim_G = optim.Adam(gen_model.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 1
epoch_nb = 30
fixed_noise = torch.randn(32, 100, 1,1, device=device)

for epoch in range(epoch_nb):
    for i, data in enumerate(dataloader):
        # Train Discriminator

        ## Train with real image
        dis_model.zero_grad()
        real_img = data[0].to(device)
        bz = real_img.size(0)

        #  label smoothing
        label= Uniform(0.9, 1.0).sample((bz,)).to(device)

        output = dis_model(real_img).view(-1)
        error_real = criterion(output, label)
        error_real.backward()
        D_x = output.mean().item()

        ## Train with fake image
        noise = torch.randn(bz, 100, 1, 1, device=device)
        fake_img = gen_model(noise)
        label = Uniform(0., 0.05).sample((bz,)).to(device)

        output = dis_model(fake_img.detach()).view(-1)
        error_fake = criterion(output, label)
        error_fake.backward()
        D_G_z1 = output.mean().item()
        error_D= error_fake + error_real
        optim_D.step()

        ## Train Generator
        gen_model.zero_grad()
        label = Uniform(0.9, 1.0).sample((bz,)).to(device)
        output = dis_model(fake_img).view(-1)
        error_G = criterion(output, label)
        error_G.backward()
        optim_G.step()
        D_G_z2 = output.mean().item()

        if i % 300 == 0:
            print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                % (epoch, epoch_nb, i, len(dataloader),
                    error_D.item(), error_G.item(), D_x, D_G_z1, D_G_z2))

        if epoch>1:
            if( iters%1000 == 0 ) or ((epoch == epoch_nb - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake_img = gen_model(noise).detach().cpu()
                fake_img = make_grid(fake_img, padding=2, normalize=True)
                img_list.append(fake_img)
                plt.figure(figsize=(10,10))
                plt.imshow(img_list[-1].permute(1,2,0))
                plt.show()
        iters+=1


























