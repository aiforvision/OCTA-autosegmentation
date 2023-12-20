import torch
from torch import nn
from models.base_model_abc import BaseModelABC
from models.model_interface_abc import Output
from monai.data import decollate_batch
from typing import Any, Callable, Tuple
from utils.losses import get_loss_function_by_name
from utils.decorators import overrides
from utils.visualizer import Visualizer
from utils.enums import Phase

class GanSegModel(BaseModelABC):

    def __init__(self,
        MODEL_DICT: dict,
        model_g: dict,
        model_d: dict,
        model_s: dict,
        compute_identity=True,
        compute_identity_seg=True,
        phase: Phase=Phase.TRAIN,
        inference: str=None,
        **kwargs):
        super().__init__(optimizer_mapping={
            "optimizer_G": ["generator"],
            "optimizer_D": ["discriminator"],
            "optimizer_S": ["segmentor"]
        }, optimizer_configs={
            "optimizer_S": {"betas": (0.9, 0.999)}
        }, **kwargs)
        self.segmentor: nn.Module = None
        self.generator: nn.Module = None
        self.discriminator: nn.Module = None
        self.optimizer_G: torch.optim.Optimizer
        self.optimizer_D: torch.optim.Optimizer
        self.optimizer_S: torch.optim.Optimizer
        if phase == Phase.TRAIN or inference == "S":
            self.segmentor = MODEL_DICT[model_s.pop("name")](**model_s)
        if phase == Phase.TRAIN or inference == "G":
            self.generator = MODEL_DICT[model_g.pop("name")](**model_g)
        if phase == Phase.TRAIN:
            self.discriminator = MODEL_DICT[model_d.pop("name")](**model_d)
        self.compute_identity = compute_identity
        self.compute_identity_seg = compute_identity_seg
        self.criterionIdt = torch.nn.L1Loss()

    @overrides(BaseModelABC)
    def initialize_model_and_optimizer(self, init_mini_batch: dict, init_weights: Callable, config: dict, args, scaler, phase: Phase=Phase.TRAIN):
        if phase != Phase.TEST:
            self.loss_name_dg = config[Phase.TRAIN]["loss_dg"]
            self.loss_name_s = config[Phase.TRAIN]["loss_s"]
            self.dg_loss = get_loss_function_by_name(self.loss_name_dg, config)
            self.s_loss = get_loss_function_by_name(self.loss_name_s, config)
        super().initialize_model_and_optimizer(init_mini_batch, init_weights,config,args,scaler,phase)

    def forward(self, input: torch.Tensor):
        if self.segmentor is not None:
            up_shape = (8*int(input.shape[2] / 2), 8*int(input.shape[3] / 2))
            return self.segmentor(torch.nn.functional.interpolate(input, size=up_shape, mode="bilinear"))
        else:
            return self.generator(input)
    
    def inference(self, mini_batch: dict[str, Any],
                post_transformations: dict[str, Callable],
                device: torch.device = "cpu",
                phase: Phase = Phase.TEST
        ) -> Tuple[Output, dict[str, torch.Tensor]]:
        assert phase==Phase.VALIDATION or phase==Phase.TEST, "This forward function only supports val and test. Use perform_step for training"
        input: torch.Tensor = mini_batch["image"].to(device=device, non_blocking=True)
        pred = self.forward(input)
        losses = dict()
        outputs: Output = { "prediction": [post_transformations["prediction"](i) for i in decollate_batch(pred[0:1,0:1])]}
        if self.segmentor is not None and phase == Phase.VALIDATION:
            labels: torch.Tensor = mini_batch["label"].to(device=device, non_blocking=True)
            outputs["label"] = [post_transformations["label"](i) for i in decollate_batch(labels[0:1,0:1])]
            losses[self.loss_name_s] = self.s_loss(pred, labels)
        return outputs, losses

    def forward_GD(self, input: tuple[torch.Tensor], tune_D = True) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        real_A, real_B = input
        fake_B: torch.Tensor = self.generator(real_A)
        if self.compute_identity_seg or self.compute_identity:
            idt_B = self.generator(real_B)
        else:
            idt_B = [None]
        
        self.discriminator.requires_grad_(tune_D)
        pred_fake_B = self.discriminator(fake_B.detach())
        pred_real_B = self.discriminator(real_B)
        return fake_B, idt_B, pred_fake_B, pred_real_B

    def forward_GS(self, real_B, fake_B, idt_B) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
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
    
    @overrides(BaseModelABC)
    def perform_training_step(self,
        mini_batch: dict[str, Any],
        scaler: torch.cuda.amp.grad_scaler.GradScaler,
        post_transformations: dict[str, Callable],
        device: torch.device = "cpu"
    ) -> Tuple[Output, dict[str, float]]:
        real_A: torch.Tensor = mini_batch["real_A"].to(device, non_blocking=True)
        real_B: torch.Tensor = mini_batch["real_B"].to(device, non_blocking=True)
        real_A_seg: torch.Tensor = mini_batch["real_A_seg"].to(device, non_blocking=True)
        self.optimizer_D.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            fake_B, idt_B, pred_fake_B, pred_real_B = self.forward_GD((real_A, real_B))
            loss_D_fake = self.dg_loss(pred_fake_B, False)
            loss_D_real = self.dg_loss(pred_real_B, True)
            loss_D = 0.5*(loss_D_fake + loss_D_real)

        scaler.scale(loss_D).backward()
        scaler.step(self.optimizer_D)

        self.optimizer_G.zero_grad(set_to_none=True)
        self.optimizer_S.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred_fake_B, fake_B_seg, real_B_seg, idt_B_seg  = self.forward_GS(real_B, fake_B, idt_B)
            real_B_seg[real_B_seg<=0.5]=0
            real_B_seg[real_B_seg>0.5]=1
            loss_G = self.dg_loss(pred_fake_B, True)
            if self.compute_identity:
                loss_G_idt = self.criterionIdt(idt_B, real_B)
            else:
                loss_G_idt = torch.tensor(0)

            loss_G += loss_G_idt

            loss_S = self.s_loss(fake_B_seg, real_A_seg)
            if self.compute_identity_seg:
                loss_S_idt = self.s_loss(idt_B_seg, real_B_seg)
                loss_SS = 0.5*(loss_S + loss_S_idt)
            else:
                loss_S_idt = torch.tensor(0)
                loss_SS = loss_S

            loss_GS = loss_G + loss_SS

        scaler.scale(loss_GS).backward()
        scaler.step(self.optimizer_G)
        scaler.step(self.optimizer_S)
        scaler.update()

        outputs: Output = {
            "prediction": [post_transformations["prediction"](i) for i in decollate_batch(fake_B_seg[0:1, 0:1])],
            "label": [post_transformations["label"](i) for i in decollate_batch(real_A_seg[0:1, 0:1])],
            "fake_B": fake_B[0:1,0:1].detach(),
            "idt_B": idt_B[0:1,0:1].detach(),
            "real_B_seg": real_B_seg
        }
        losses = {
            "S": loss_S.item(),
            "D_fake": loss_D_fake.item(),
            "D_real": loss_D_real.item(),
            "G": loss_G.item(),
            "G_idt": loss_G_idt.item(),
            "S_idt": loss_S_idt.item()
        }
        return outputs, losses

    @overrides(BaseModelABC)
    def plot_sample(self, visualizer: Visualizer, mini_batch: dict[str, Any], outputs: Output, *, suffix: str = ""):
        if "fake_B" in outputs:
            return visualizer.plot_gan_seg_sample(
                 mini_batch["real_A"][0],
                outputs["fake_B"][0],
                outputs["prediction"][0],
                mini_batch["real_B"][0],
                outputs["idt_B"][0],
                outputs["real_B_seg"][0],
                path_A=mini_batch["real_A_path"][0],
                path_B=mini_batch["real_B_path"][0],
                suffix=suffix
            )
        else:
            return visualizer.plot_sample(
                mini_batch["image"][0],
                outputs["prediction"][0],
                outputs["label"][0],
                path=mini_batch["image_path"][0],
                suffix=suffix
            )
