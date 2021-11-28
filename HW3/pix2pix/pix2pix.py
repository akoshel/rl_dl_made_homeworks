from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_challels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_challels, out_channels, kernel_size=3, padding=1),
        # nn.BatchNorm2d(out_channels),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        # nn.BatchNorm2d(out_channels),
        nn.InstanceNorm2d(out_channels, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )

class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_down1 = conv_layer(3, 64)
        self.conv_down2 = conv_layer(64, 128)
        self.conv_down3 = conv_layer(128, 256)
        self.conv_down4 = conv_layer(256, 512)
        self.conv_down5 = conv_layer(512, 1024)
        
        self.conv_up1 = conv_layer(512 + 1024, 512)
        self.conv_up2 = conv_layer(256 + 512, 256)
        self.conv_up3 = conv_layer(128 + 256, 128)
        self.conv_up4 = conv_layer(64 + 128, 64)
        # self.conv_up4 = conv_layer(64, 3)
        self.conv_up5 = nn.Conv2d(64, 3, kernel_size=1)
        
        self.down = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        
    def forward(self, x):
        out1 = self.conv_down1(x)
        out1 = F.dropout(out1)
        x = self.down(out1)
        
        out2 = self.conv_down2(x)
        out2 = F.dropout(out2)
        x = self.down(out2)
        
        out3 = self.conv_down3(x)
        out3 = F.dropout(out3)
        x = self.down(out3)

        out4 = self.conv_down4(x)
        out4 = F.dropout(out4)
        x = self.down(out4)
        
        x = self.conv_down5(x)
        
        x = self.up(x)    
        x = torch.cat([x, out4], dim=1)
        x = self.conv_up1(x)
        
        x = self.up(x)        
        x = torch.cat([x, out3], dim=1)
        x = self.conv_up2(x)
        
        x = self.up(x)        
        x = torch.cat([x, out2], dim=1)
        x = self.conv_up3(x)

        x = self.up(x)        
        x = torch.cat([x, out1], dim=1)
        x = self.conv_up4(x)
        
        x = self.conv_up5(x)
        x = torch.sigmoid(x)
        return x
   
        

class Discriminator(nn.Module):
    def __init__(self, is_single=False):
        super().__init__()
        self.is_single = is_single
        self.conv_down1 = conv_layer(3 if is_single else 6, 64)
        self.conv_down2 = conv_layer(64, 128)
        self.conv_down3 = conv_layer(128, 256)
        self.conv_down4 = conv_layer(256, 512)
        self.down = nn.MaxPool2d(kernel_size=2)
        
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)
        
    def forward(self, condition, x=None):
        if self.is_single:
            x = condition
        else:
            x = torch.cat([condition, x], dim=1)
        
        x = self.conv_down1(x)
        x = self.down(x)
        x = F.dropout(x)
        
        x = self.conv_down2(x)
        x = self.down(x)
        x = F.dropout(x)
        
        x = self.conv_down3(x)
        x = self.down(x)
        
        x = self.conv_down4(x)
        x = self.down(x) 
        
        x = self.pooling(x)
        x = x.squeeze(-1).squeeze(-1) 
        x = self.fc(x)

        return x


class Pix2Pix:

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = GeneratorUNet().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.criterion_l1 = torch.nn.SmoothL1Loss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_discrimator = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.patch = (1, 256 // 2 ** 4, 256 ** 4) # 256x256 image size

    def train_step(self, batch: List[torch.Tensor]) -> Tuple[float, float]:
        real_seg, real_origin = batch
        real_seg = real_seg.to(self.device)
        real_origin = real_origin.to(self.device)
        self.optimizer_discrimator.zero_grad()
        img_fake = self.generator(real_seg)
        logits_fake = self.discriminator(real_seg, img_fake)
        logits_real = self.discriminator(real_seg, real_origin)
        D_fake_loss = self.criterion_ce(logits_fake, torch.zeros(logits_fake.shape[0]).long().to(self.device))
        D_real_loss = self.criterion_ce(logits_real, torch.ones(logits_fake.shape[0]).long().to(self.device))
        D_loss = (D_fake_loss + D_real_loss) * 0.5    
        D_loss.backward()
        self.optimizer_discrimator.step()

        self.optimizer_generator.zero_grad()
        img_fake = self.generator(real_seg)
        logits_fake = self.discriminator(real_origin, img_fake)
        G_cross_entropy = self.criterion_ce(logits_fake, torch.ones(logits_fake.shape[0]).long().to(self.device))
        G_l1 = self.criterion_l1(img_fake, real_origin)
        G_loss = G_cross_entropy + 100 * G_l1
        
        G_loss.backward()
        self.optimizer_generator.step()
        
        return D_loss.item(), G_loss.item()

    def generate(self, batch: torch.Tensor) -> None:
        with torch.no_grad():
            pred = self.generator(batch.to(self.device))
        return pred.cpu()
