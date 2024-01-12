""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from thop import profile, clever_format

from unet_misc import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def hohonet_counter():
    print('testing UNet')

    device = torch.device('cuda')

    test_full = False

    net = UNet(3,2).to(device)
    
    # testing
    layers = net

    if(test_full):
        inputs = [torch.randn(1, 3, 512, 1024).to(device)]
    else:
        inputs = [torch.randn(1, 3, 256, 512).to(device)]

    with torch.no_grad():
        flops, params = profile(layers, inputs)
    ##print(f'input :', [v.shape for v in inputs])
    print(f'flops : {flops/(10**9):.2f} G')
    print(f'params: {params/(10**6):.2f} M')

    import time
    fps = []
    with torch.no_grad():
        out = layers(inputs[0])
        print('out shape',out.shape)
        for _ in range(50):
            eps_time = time.time()
            layers(inputs[0])
            torch.cuda.synchronize()
            eps_time = time.time() - eps_time
            fps.append(eps_time)
    print(f'fps   : {1 / (sum(fps) / len(fps)):.2f}')


if __name__ == "__main__":
    hohonet_counter()
    #model = UNet(3,3)
    #model.initialize_weights()
    #x = torch.randn(2,3,256,512)
    #y = model(x)
    
    #print(y.shape)