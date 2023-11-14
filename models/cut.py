import numpy as np
import torch
import torch.nn as nn
from models.base_model_abc import BaseModelABC, Output
from utils.enums import Phase
from typing import Any, Callable, Tuple
from utils.decorators import overrides
from utils.losses import get_loss_function_by_name
from monai.data import decollate_batch
from utils.visualizer import Visualizer


class CUTModel(BaseModelABC):
    """ 
    Adapted from https://github.com/taesungp/contrastive-unpaired-translation

    This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self,
            phase: Phase,
            MODEL_DICT: dict,
            inference:str,
            netG_config: dict,
            netD_config: dict,
            netF_config: dict,
            nce_layers: str,
            nce_idt:float,
            lambda_NCE:float,
            lambda_GAN: float,
            flip_equivariance: bool,
            num_patches: int,
            *args, **kwargs) -> None:
        super().__init__({
            "optimizer_G": ["netG"],
            "optimizer_D": ["netD"],
            "optimizer_F": ["netF"]
        }, *args, **kwargs)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.nce_layers = [int(i) for i in nce_layers.split(',')]
        self.lambda_NCE = lambda_NCE
        self.lambda_GAN = lambda_GAN
        self.nce_idt = nce_idt
        self.flip_equivariance = flip_equivariance
        self.num_patches = num_patches

        self.netG: nn.Module = None
        self.netD: nn.Module = None
        self.netF: nn.Module = None
        self.optimizer_G: torch.optim.Optimizer = None
        self.optimizer_D: torch.optim.Optimizer = None
        self.optimizer_F: torch.optim.Optimizer = None

        # define networks (both generator and discriminator)
        self.netG = MODEL_DICT[netG_config.pop("name")](**netG_config)
        if phase == Phase.TRAIN:
            self.netD = MODEL_DICT[netD_config.pop("name")](**netD_config)
            self.netF = MODEL_DICT[netF_config.pop("name")](**netF_config)

    @overrides(BaseModelABC)
    def initialize_model_and_optimizer(self, init_mini_batch: dict, init_weights: Callable, config: dict, args, scaler, phase: Phase=Phase.TRAIN):
        self.loss_name_criterionGAN = config[Phase.TRAIN]["loss_criterionGAN"]
        self.criterionGAN = get_loss_function_by_name(self.loss_name_criterionGAN, config)

        self.loss_name_criterionNCE = config[Phase.TRAIN]["loss_criterionNCE"]
        self.criterionNCE = []
        for _ in self.nce_layers:
            self.criterionNCE.append(get_loss_function_by_name(self.loss_name_criterionNCE, config))

        # Initialize netF
        with torch.cuda.amp.autocast():
            feat_k = self.netG(init_mini_batch["image"].to(config["General"]["device"], non_blocking=True), self.nce_layers, encode_only=True)
            feat_k_pool, sample_ids = self.netF(feat_k, self.num_patches, None)

        super().initialize_model_and_optimizer(init_mini_batch,init_weights,config,args,scaler,phase)

    @overrides(BaseModelABC)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.netG(input)
    
    @overrides(BaseModelABC)
    def inference(self,
                mini_batch: dict[str, Any],
                post_transformations: dict[str, Callable],
                device: torch.device = "cpu",
                phase: Phase = Phase.TEST
        ) -> Tuple[Output, dict[str, torch.Tensor]]:
        """
        Computes a full forward pass given a mini_batch.

        Parameters:
        -----------
        - mini_batch: Dictionary containing the inputs and their names
        - post_transformations: Dictionary containing the post transformation for every output
        - device: Device on which to compute
        - phase: Either training, validation or test phase

        Returns:
        --------
        - Dictionary containing the predictions and their names
        - Dictionary containing the losses and their names
        """
        assert phase==Phase.VALIDATION or phase==Phase.TEST, "This inference function only supports val and test. Use perform_step for training"
        input = mini_batch["image"]
        pred = self.forward(input)
        losses = dict()
        outputs: Output = { "prediction": [post_transformations["prediction"](i) for i in decollate_batch(pred[0:1,0:1])]}
        return outputs, losses
    
    @overrides(BaseModelABC)
    def perform_training_step(self,
            mini_batch: dict[str, Any],
            scaler: torch.cuda.amp.grad_scaler.GradScaler,
            post_transformations: dict[str, Callable],
            device: torch.device = "cpu"
        ) -> Tuple[Output, dict[str, float]]:
        """
        Computes the output and losses of a single mini_batch.

        Parameters:
        -----------
        - mini_batch: Dictionary containing the inputs and their names
        - scaler: GradScaler
        - post_transformations: Dictionary containing the post transformation for every output
        - device: Device on which to compute
        
        Returns:
        --------
        - Dictionary containing the outputs and their names
        - Dictionary containing the loss values and their names
        """
        # Training
        real_A: torch.Tensor = mini_batch["real_A"].to(device, non_blocking=True)
        real_B: torch.Tensor = mini_batch["real_B"].to(device, non_blocking=True)
        
        ####################################
        with torch.cuda.amp.autocast():
            # forward
            real: torch.Tensor = torch.cat((real_A, real_B), dim=0) if self.nce_idt else real_A
            if self.flip_equivariance:
                self.flipped_for_equivariance = np.random.random() < 0.5
                if self.flipped_for_equivariance:
                    real = torch.flip(real, [3])

            fake: torch.Tensor = self.netG(real)
            fake_B = fake[:real_A.size(0)]
            if self.nce_idt:
                idt_B = fake[real_A.size(0):]

            # update D
            self.netD.requires_grad_(True)
            self.optimizer_D.zero_grad(set_to_none=True)

            fake = fake_B.detach()
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake: torch.Tensor = self.netD(fake)
            loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            # Real
            pred_real = self.netD(real_B)
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_real = loss_D_real.mean()
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5

        scaler.scale(loss_D).backward()
        scaler.step(self.optimizer_D)
        ####################################

        ####################################
        # update G and F
        self.optimizer_G.zero_grad(set_to_none=True)
        self.optimizer_F.zero_grad(set_to_none=True)
        self.netD.requires_grad_(False)
        with torch.cuda.amp.autocast():
            fake = fake_B
            # First, G(A) should fake the discriminator
            if self.lambda_GAN > 0.0:
                pred_fake: torch.Tensor = self.netD(fake)
                loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.lambda_GAN
            else:
                loss_G_GAN = 0.0

            if self.lambda_NCE > 0.0:
                loss_NCE = self._calculate_NCE_loss(real_A, fake_B)
            else:
                loss_NCE = 0.0

            if self.nce_idt and self.lambda_NCE > 0.0:
                loss_NCE_Y = self._calculate_NCE_loss(real_B, idt_B)
                loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5
            else:
                loss_NCE_both = loss_NCE

            loss_G = loss_G_GAN + loss_NCE_both

        scaler.scale(loss_G).backward()
        scaler.step(self.optimizer_G)
        scaler.step(self.optimizer_F)
        ####################################
        scaler.update()

        outputs: Output = {
            "prediction": [post_transformations["prediction"](i) for i in decollate_batch(fake_B[0:1, 0:1])],
            "label": [post_transformations["prediction"](i) for i in decollate_batch(real_B[0:1, 0:1])],
            "idt_B": idt_B[0:1,0:1].detach()
        }
        losses = {
            "G": loss_G.item(),
            "loss_NCE": loss_NCE.item(),
            "loss_NCE_Y": loss_NCE_Y.item(),
            "D_fake": loss_D_fake.item(),
            "D_real": loss_D_real.item()
        }
        return outputs, losses

    def _calculate_NCE_loss(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss: torch.Tensor = crit(f_q, f_k) * self.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    @overrides(BaseModelABC)
    def plot_sample(self, visualizer: Visualizer, mini_batch: dict[str, Any], outputs: Output, *, suffix: str = "") -> str:
        """
        Plots a sample for the given mini_batch each save_interval

        Parameters:
        -----------
        - visualizer: Visualizer instance
        - mini_batch: Current mini_batch
        - outputs: Generated outputs by forward pass
        - suffix: Text suffix for file name
        """
        if "fake_B" in outputs:
            return visualizer.plot_cut_sample(
                real_A=mini_batch["real_A"][0],
                fake_B=outputs["prediction"][0],
                real_B=mini_batch["real_B"][0],
                idt_B=outputs["idt_B"],
                path_A=mini_batch["real_A_path"][0],
                path_B=mini_batch["real_B_path"][0],
                suffix=suffix
            )
        else:
            return visualizer.plot_sample(
                mini_batch["real_A"][0],
                outputs["prediction"][0],
                outputs["label"][0],
                path=mini_batch["real_A_path"][0],
                suffix=suffix
            )
