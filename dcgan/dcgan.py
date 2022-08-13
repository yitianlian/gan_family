import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torchvision.utils as vutils
import random

manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

num_epoch = 4
learning_rate = 0.0002
z_dimension = 100
nc = 3  #num of channels
ndf = 64  #判别网络中的初始feature数
ngf = 64  #生成网络中的初始feature数
batchSize = 64
beta = 0.5

# data_loader
img_size = 64
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
data_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='data', download=True, transform=transform),
    batch_size=batchSize,
    shuffle=True,
    num_workers=0,
    drop_last=True)

# data_loader = datasets.CIFAR10(root='data',download=True,transform=transform)


##Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            ##input is nc*64*64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            ## state size ndf*32*32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 卷积核大小为4，步长为2，填补1
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            ##state size (ndf*2)*16*16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            ##state size (ndf*4)*8*8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            ##state size (ndf*8)*4*4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        return x.view(-1, 1).squeeze(1)


##Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(

            ##input is z_dimension
            nn.ConvTranspose2d(z_dimension, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            ##state size (ngf*8)*4*4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            ##state size (ngf*4)*8*8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            ##state size (ngf*2)*16*16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            ##state size ngf*32*32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.gen(x)
        return x


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


D = Discriminator()
D.apply(weights_init)
G = Generator()
G.apply(weights_init)
G.cuda()
D.cuda()
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(
    D.parameters(), lr=learning_rate, betas=(beta, 0.999))
g_optimizer = torch.optim.Adam(
    G.parameters(), lr=learning_rate, betas=(beta, 0.999))


#print network architercture
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


print('---------- Networks architecture -------------')
print_network(G)
print_network(D)
print('-----------------------------------------------')

# r_label = torch.ones(batchSize, 1).cuda()
# f_label = torch.zeros(batchSize, 1).cuda()
fixed_noise = torch.randn(batchSize, z_dimension, 1, 1).cuda()
print('training start!!')
##start training
for epoch in range(num_epoch):
    for iter, data in enumerate(data_loader, 0):
        x_ = data[0].cuda()
        bacthSize = x_.size(0)
        r_label = torch.full((batchSize, ), 1).cuda()
        f_label = torch.full((batchSize, ), 0).cuda()
        ##train Disceiminator
        D.zero_grad()

        ##compute the loss of real_image
        real_out = D(x_)
        d_loss_real = criterion(real_out.float(), r_label.float())
        d_loss_real.backward()
        D_x = real_out.mean().item()

        ##compute the loss of fake_image
        noise = torch.randn(batchSize, z_dimension, 1, 1).cuda()
        fake_img = G(noise)
        fake_out = D(fake_img.detach())
        d_loss_fake = criterion(fake_out.float(), f_label.float())
        d_loss_fake.backward()
        D_G_z1 = fake_out.mean().item()

        ##bp and optimize
        d_loss = d_loss_fake + d_loss_real
        d_optimizer.step()

        ##train Generotor
        G.zero_grad()
        output = D(fake_img)
        g_loss = criterion(output.float(), r_label.float())

        ##bp and optimize
        g_loss.backward()
        D_G_z2 = output.mean().item()
        g_optimizer.step()

        if ((iter + 1) % 100) == 0:
            vutils.save_image(x_, './real_samples.png', normalize=True)
            fake = G(fixed_noise)
            vutils.save_image(
                fake.detach(),
                './fake_samples_epoch_%03d.png' % epoch,
                normalize=True)
            print(
                '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, num_epoch, iter, len(data_loader), d_loss.item(),
                   g_loss.item(), D_x, D_G_z1, D_G_z2))
    if (epoch+1)%5 == 0:
        addressG = '_G' + str(epoch) + '.pkl'
        addressD = '_D' + str(epoch) + '.pkl'
        torch.save(G.state_dict(), addressG)
        torch.save(D.state_dict(), addressD)
print("Training finish!... save training results")
torch.save(G.state_dict(), '_G.pkl')
torch.save(D.state_dict(), '_D.pkl')
print("well done!")