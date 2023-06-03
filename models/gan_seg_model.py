import torch
from torch import nn
import random

class GanSegModel(nn.Module):

    def __init__(self,
        MODEL_DICT: dict,
        model_g: dict,
        model_d: dict,
        model_s: dict,
        compute_identity=True,
        compute_identity_seg=True,
        phase="train",
        inference: str=None,
        **kwargs):
        super().__init__()
        self.segmentor: nn.Module = None
        self.generator: nn.Module = None
        self.discriminator: nn.Module = None
        if phase == "train" or inference == "S":
            self.segmentor = MODEL_DICT[model_s.pop("name")](**model_s)
        if phase == "train" or inference == "G":
            self.generator = MODEL_DICT[model_g.pop("name")](**model_g)
        if phase == "train":
            self.discriminator = MODEL_DICT[model_d.pop("name")](**model_d)
        self.compute_identity = compute_identity
        self.compute_identity_seg = compute_identity_seg
        self.inference = False

    def eval(self):
        if self.generator is not None:
            self.generator.eval()
        if self.discriminator is not None:
            self.discriminator.eval()
        if self.segmentor is not None:
            self.segmentor.eval()
        self.inference = True

    def train(self, *params):
        self.generator.train()
        self.discriminator.train()
        self.segmentor.train()
        self.inference = False

    def forward(self, input, _=None, complete=False):
        if complete:
            if not isinstance(input, tuple):
                input = input, _
            real_A, real_B = input
            fake_B, idt_B, pred_fake_B, pred_real_B = self.forward_GD(input)
            pred_fake_B, fake_B_seg, real_B_seg, idt_B_seg = self.forward_GS(real_B, fake_B, idt_B)
            return fake_B_seg
        else:
            if self.segmentor is not None:
                up_shape = (8*int(input.shape[2] / 2), 8*int(input.shape[3] / 2))
                return self.segmentor(torch.nn.functional.interpolate(input, size=up_shape, mode="bilinear"))
            else:
                return self.generator(input)


    def forward_GD(self, input: tuple[torch.Tensor], tune_D = True) -> tuple[torch.Tensor]:
        real_A, real_B = input
        fake_B = self.generator(real_A)
        if self.compute_identity_seg or self.compute_identity:
            idt_B = self.generator(real_B)
        else:
            idt_B = [None]
        
        self.discriminator.requires_grad_(tune_D)
        pred_fake_B = self.discriminator(fake_B.detach())
        pred_real_B = self.discriminator(real_B)
        return fake_B, idt_B, pred_fake_B, pred_real_B

    def forward_GS(self, real_B, fake_B, idt_B) -> tuple[torch.Tensor]:
        """Calculate GAN and NCE loss for the generator"""
        # First, G(A) should fake the discriminator
        self.discriminator.requires_grad_(False)
        pred_fake_B = self.discriminator(fake_B)

        up_shape = (min(1216,8*int(real_B.shape[2] / 2)), min(1216,8*int(real_B.shape[3] / 2)))
        real_B_seg = self.segmentor(torch.nn.functional.interpolate(real_B, size=up_shape, mode="bilinear"))
        if self.compute_identity_seg:
            idt_B_seg = self.segmentor(torch.nn.functional.interpolate(idt_B, size=up_shape, mode="bilinear"))
        else:
            idt_B_seg = [None]
        fake_B_seg = self.segmentor(torch.nn.functional.interpolate(fake_B, size=up_shape, mode="bilinear"))
        return pred_fake_B, fake_B_seg, real_B_seg, idt_B_seg

    def apply(self, init_func):
        self.generator.apply(init_func)
        self.discriminator.apply(init_func)
        self.segmentor.apply(init_func)

    def control_point_brightness_augmentation(self, t: torch.Tensor):
        if not self.inference and random.random()>0.5:
            c = torch.rand([*t.shape[:2],4,4], dtype=t.dtype, device=t.device)*0.8+0.6
            C = torch.nn.functional.interpolate(c, size=t.shape[2:], mode="bicubic")
            t = torch.clip(t*C, 0,1)
        return t
