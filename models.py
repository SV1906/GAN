import torch.nn

# class Discriminator(torch.nn.Module):
#     def __init__(self, input_channels=3):
#         super(Discriminator, self).__init__()
        
#         self.conv1 = torch.nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1, bias = False)
#         self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias = False)
#         self.bn2 = torch.nn.BatchNorm2d(256)
#         self.conv3 = torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1,bias = False)
#         self.bn3 = torch.nn.BatchNorm2d(512)
#         self.conv4 = torch.nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1,bias = False)
#         self.bn4 = torch.nn.BatchNorm2d(1024)
#         self.conv5 = torch.nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1,bias = False)

#     def forward(self, x):
#         x = torch.nn.functional.leaky_relu(self.conv1(x), 0.2, inplace = True)
#         x = torch.nn.functional.leaky_relu(self.bn2(self.conv2(x)), 0.2)
#         x = torch.nn.functional.leaky_relu(self.bn3(self.conv3(x)), 0.2)
#         x = torch.nn.functional.leaky_relu(self.bn4(self.conv4(x)), 0.2)
#         x = torch.sigmoid(self.conv5(x))
#         return x

class Discriminator(torch.nn.Module):
    def __init__(self,input_channels=3):
        super().__init__()
        # super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        nc = 3 
        ndf = 128
        self.main = torch.nn.Sequential(
            # input is (nc) x 64 x 64

            torch.nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            torch.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            torch.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            torch.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            torch.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
# # Create the Discriminator
# netD = Discriminator(ngpu).to(device)
# # Handle multi-gpu if desired
# if (device.type == 'cuda') and (ngpu > 1):
#     netD = nn.DataParallel(netD, list(range(ngpu)))
# netD.apply(weights_init)
# # Print the model
# print(netD)
# class Generator(torch.nn.Module):
#     def __init__(self, noise_dim, output_channels=3):
#         super(Generator, self).__init__()    
#         self.noise_dim = noise_dim
        
#         self.conv1 = torch.nn.ConvTranspose2d(noise_dim, 1024, kernel_size=4, stride=1, padding=0)
#         self.bn1 = torch.nn.BatchNorm2d(1024)
#         self.conv2 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
#         self.bn2 = torch.nn.BatchNorm2d(512)
#         self.conv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
#         self.bn3 = torch.nn.BatchNorm2d(256)
#         self.conv4 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.bn4 = torch.nn.BatchNorm2d(128)
#         self.conv5 = torch.nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)

#     def forward(self, x):
#         x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
#         x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
#         x = torch.nn.functional.relu(self.bn3(self.conv3(x)))
#         x = torch.nn.functional.relu(self.bn4(self.conv4(x)))
#         x = torch.tanh(self.conv5(x))
#         return x
class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super().__init__()
        # self.ngpu = ngpu
        nz = noise_dim 
        nc = output_channels
        ngf = 128
        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            torch.nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)
# netG = Generator(ngpu).to(device)
# # Handle multi-gpu if desired
# if (device.type == 'cuda') and (ngpu > 1):
#     netG = nn.DataParallel(netG, list(range(ngpu)))
# netG.apply(weights_init)
# # Print the model
# print(netG)