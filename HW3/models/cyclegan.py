from typing import List, Tuple
import torch
from torch.optim import Adam
from .pix2pix import GeneratorUNet, Discriminator

class CycleGan:

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator1 = GeneratorUNet().to(self.device)
        self.generator2 = GeneratorUNet().to(self.device)
        self.discriminator1 = Discriminator(is_cycle=True)
        self.discriminator2 = Discriminator(is_cycle=True)
        self.optimizer_g1 = Adam(self.generator1.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_g2 = Adam(self.generator2.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d1 = Adam(self.discriminator1.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d2 = Adam(self.discriminator2.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion_l1 = torch.nn.SmoothL1Loss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()

    def train_step(self, batch: List[torch.Tensor]) -> Tuple[float, float, float, float]:
        real_seg, real_origin = batch
        real_seg = real_seg.to(self.device)
        real_origin = real_origin.to(self.device)
        self.optimizer_d1.zero_grad()
        self.optimizer_d2.zero_grad()
        fake_origin = self.generator1(real_seg)
        fake_seg = self.generator2(real_origin)
        seg_logits_fake = self.discriminator1(fake_seg)
        seg_logits_real = self.discriminator1(real_seg)
        orig_logits_fake = self.discriminator2(fake_origin)
        orig_logits_real = self.discriminator2(real_origin)

        d1_fake_loss = self.criterion_ce(seg_logits_fake, torch.zeros(seg_logits_fake.shape[0]).long().to(self.device))
        d1_real_loss = self.criterion_ce(seg_logits_real, torch.ones(seg_logits_real.shape[0]).long().to(self.device))
        loss_d1 = (d1_fake_loss + d1_real_loss) * 0.5
        loss_d1.backward()

        d2_fake_loss = self.criterion_ce(orig_logits_fake, torch.zeros(orig_logits_fake.shape[0]).long().to(self.device))
        d2_real_loss = self.criterion_ce(orig_logits_real, torch.ones(orig_logits_real.shape[0]).long().to(self.device))
        loss_d2 = (d2_fake_loss + d2_real_loss) * 0.5
        loss_d2.backward()
        
        self.optimizer_d1.step()
        self.optimizer_d2.step()

        self.optimizer_g1.zero_grad()
        self.optimizer_g2.zero_grad()

        fake_orig = self.generator1(real_seg)
        fake_seg = self.generator2(real_origin)
        fake_cycle_orig = self.generator1(fake_seg)
        fake_cycle_seg = self.generator2(fake_orig)
        loss_g1_l1 = 0.5 * (self.criterion_l1(real_seg, fake_cycle_seg) + self.criterion_l1(real_origin, fake_cycle_orig))
        loss_g2_l1 = 0.5 * (self.criterion_l1(real_seg, fake_seg) + self.criterion_l1(real_origin, fake_orig))
        seg_logits_fake = self.discriminator1(fake_seg)
        orig_logits_fake = self.discriminator2(fake_orig)
        ce_g1 = self.criterion_ce(orig_logits_fake, torch.ones(orig_logits_fake.shape[0]).long().to(self.device))
        ce_g2 = self.criterion_ce(seg_logits_fake, torch.ones(seg_logits_fake.shape[0]).long().to(self.device))
        
        loss_g1 = loss_g1_l1 + ce_g1
        loss_g2 = loss_g2_l1 + ce_g2

        loss_g1.backward()
        loss_g2.backward()

        self.optimizer_g1.step()
        self.optimizer_g2.step()
        
        return loss_d1.item(), loss_d2.item(), loss_g1.item(), loss_g2.item()
    
    def generate(self, batch: torch.Tensor) -> None:
        with torch.no_grad():
            pred = self.generator1(batch.to(self.device))
        return pred.cpu()