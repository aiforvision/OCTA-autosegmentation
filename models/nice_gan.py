import torch
from torch import nn
from typing import Any, Callable, Type, Tuple
from models.base_model_abc import BaseModelABC, Output
from utils.enums import Phase
from utils.decorators import overrides
from utils.losses import get_loss_function_by_name
from monai.data import decollate_batch
from utils.visualizer import Visualizer

class NiceGAN(BaseModelABC):
    """
    Adapted from https://github.com/alpc91/NICE-GAN-pytorch
    """
    def __init__(self,
        phase: Phase,
        MODEL_DICT: dict[str, Type[nn.Module]],
        inference: str,
        gen2B_config: dict,
        gen2A_config: dict,
        disA_config: dict,
        disB_config: dict,
        adv_weight: float = 1,
        cycle_weight: float = 10,
        recon_weight: float = 1,
        **kwargs) -> None:
        super().__init__(
            optimizer_mapping={
                "G_optim": ["gen2A","gen2B"],
                "D_optim": ["disA", "disB"]
            }, **kwargs)

        """ Weight """
        self.adv_weight = adv_weight
        self.cycle_weight = cycle_weight
        self.recon_weight = recon_weight

        """ Define Generator, Discriminator """
        self.gen2A: nn.Module = None
        self.gen2B: nn.Module = None
        self.disA: nn.Module = None
        self.disB: nn.Module = None
        self.G_optim: torch.optim.Optimizer
        self.D_optim: torch.optim.Optimizer
        if phase == Phase.TRAIN or inference == "gen2A":
            self.gen2A = MODEL_DICT[gen2A_config.pop("name")](**gen2A_config)
        if phase == Phase.TRAIN or inference == "gen2B":
            self.gen2B = MODEL_DICT[gen2B_config.pop("name")](**gen2B_config)
        if phase == Phase.TRAIN:
            self.disA = MODEL_DICT[disA_config.pop("name")](**disA_config)
            self.disB = MODEL_DICT[disB_config.pop("name")](**disB_config)

    @overrides(BaseModelABC)
    def initialize_model_and_optimizer(self, init_mini_batch, init_weights: Callable, config: dict, args, scaler, phase: Phase=Phase.TRAIN):
        self.loss_name_ad = config[Phase.TRAIN]["loss_ad"]
        self.ad_loss = get_loss_function_by_name(self.loss_name_ad, config)

        self.loss_name_cycle = config[Phase.TRAIN]["loss_cycle"]
        self.cycle_loss = get_loss_function_by_name(self.loss_name_cycle, config)

        super().initialize_model_and_optimizer(init_mini_batch,init_weights,config,args,scaler,phase)

    
    @overrides(BaseModelABC)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs a minimal forward pass of the underlying model.

        Parameters:
        -----------
        input: Input image as tensor

        Returns:
        --------
        prediction: Predicted tensor
        """
        if self.gen2B is not None:
            return self.gen2B(input)
        else:
            return self.gen2A(input)
    
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
        if phase==Phase.VALIDATION or phase==Phase.TEST:
            input = mini_batch["image"]
            pred = self.forward(input)
            losses = dict()
            outputs: Output = { "prediction": [post_transformations["prediction"](i) for i in decollate_batch(pred[0:1,0:1])]}

            if self.gen2A is not None and phase == Phase.VALIDATION:
                labels: torch.Tensor = mini_batch["label"].to(device=device, non_blocking=True)
                outputs["label"] = [post_transformations["label"](i) for i in decollate_batch(labels[0:1,0:1])]
                losses[self.loss_name_cycle] = self.cycle_loss(pred, labels)
            return outputs, losses
        raise NotImplementedError("This inference function only supports val and test. Use perform_step for training")

    @overrides(BaseModelABC)
    def perform_training_step(self,
            mini_batch: dict[str, torch.Tensor],
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
        real_A: torch.Tensor = mini_batch["real_A"].to(device, non_blocking=True)
        real_B: torch.Tensor = mini_batch["real_B"].to(device, non_blocking=True)
        background: torch.Tensor = mini_batch["background"].to(device, non_blocking=True) if "background" in mini_batch else torch.rand_like(real_A, device=device, dtype=real_A.dtype)
        background = background*torch.zeros_like(real_A, device=device, dtype=real_A.dtype).uniform_(0,1)
        
        ####################################
        # Update D
        self.D_optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            real_LA_logit,real_GA_logit, real_A_cam_logit, _, real_A_z = self.disA(real_A)
            real_LB_logit,real_GB_logit, real_B_cam_logit, _, real_B_z = self.disB(real_B)

            fake_A2B: torch.Tensor = self.gen2B(real_A_z)
            fake_B2A: torch.Tensor = self.gen2A(real_B_z)

            fake_B2A: torch.Tensor = fake_B2A.detach()
            fake_A2B: torch.Tensor = fake_A2B.detach()

            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, _ = self.disA(fake_B2A)
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.disB(fake_A2B)

            D_ad_loss_GA = self.ad_loss(real_GA_logit, torch.ones_like(real_GA_logit, device=device)) + self.ad_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit, device=device))
            D_ad_loss_LA = self.ad_loss(real_LA_logit, torch.ones_like(real_LA_logit, device=device)) + self.ad_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit, device=device))
            D_ad_loss_GB = self.ad_loss(real_GB_logit, torch.ones_like(real_GB_logit, device=device)) + self.ad_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit, device=device))
            D_ad_loss_LB = self.ad_loss(real_LB_logit, torch.ones_like(real_LB_logit, device=device)) + self.ad_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit, device=device))            
            D_ad_cam_loss_A = self.ad_loss(real_A_cam_logit, torch.ones_like(real_A_cam_logit, device=device)) + self.ad_loss(fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit, device=device))
            D_ad_cam_loss_B = self.ad_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit, device=device)) + self.ad_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit, device=device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
        scaler.scale(Discriminator_loss).backward()
        scaler.step(self.D_optim)
        ####################################

        ####################################
        # Update G
        self.G_optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            _,  _,  _, _, real_A_z = self.disA(torch.maximum(real_A,background)) # TODO might be incorrect
            # _,  _,  _, _, real_A_z = self.disA(real_A)
            _,  _,  _, _, real_B_z = self.disB(real_B)

            fake_A2B: torch.Tensor = self.gen2B(real_A_z)
            fake_B2A: torch.Tensor = self.gen2A(real_B_z)

            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, fake_A_z = self.disA(torch.maximum(fake_B2A,background)) # TODO might be incorrect
            # fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, fake_A_z = self.disA(fake_B2A)
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, fake_B_z = self.disB(fake_A2B)
            
            fake_B2A2B: torch.Tensor = self.gen2B(fake_A_z)
            fake_A2B2A: torch.Tensor = self.gen2A(fake_B_z)


            G_ad_loss_GA = self.ad_loss(fake_GA_logit, torch.ones_like(fake_GA_logit, device=device))
            G_ad_loss_LA = self.ad_loss(fake_LA_logit, torch.ones_like(fake_LA_logit, device=device))
            G_ad_loss_GB = self.ad_loss(fake_GB_logit, torch.ones_like(fake_GB_logit, device=device))
            G_ad_loss_LB = self.ad_loss(fake_LB_logit, torch.ones_like(fake_LB_logit, device=device))

            G_ad_cam_loss_A = self.ad_loss(fake_A_cam_logit, torch.ones_like(fake_A_cam_logit, device=device))
            G_ad_cam_loss_B = self.ad_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit, device=device))

            G_cycle_loss_A = self.cycle_loss(fake_A2B2A, real_A)
            G_cycle_loss_B = self.cycle_loss(fake_B2A2B, real_B)

            fake_A2A: torch.Tensor = self.gen2A(real_A_z)
            fake_B2B: torch.Tensor = self.gen2B(real_B_z)

            G_recon_loss_A = self.cycle_loss(fake_A2A, real_A)
            G_recon_loss_B = self.cycle_loss(fake_B2B, real_B)

            G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA ) + self.cycle_weight * G_cycle_loss_A + self.recon_weight * G_recon_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB ) + self.cycle_weight * G_cycle_loss_B + self.recon_weight * G_recon_loss_B

            Generator_loss = G_loss_A + G_loss_B
        scaler.scale(Generator_loss).backward()
        scaler.step(self.G_optim)
        ####################################
        scaler.update()

        outputs: Output = {
            "prediction": [post_transformations["prediction"](i) for i in decollate_batch(fake_A2B2A[0:1, 0:1])],
            "label": [post_transformations["label"](i) for i in decollate_batch(real_A[0:1, 0:1])],
            "fake_B": fake_A2B[0:1,0:1].detach(),
            "idt_B": fake_B2B[0:1,0:1].detach(),
            "real_B_seg": fake_B2A[0:1,0:1].detach()
        }
        losses = {
            "G": Generator_loss.item(),
            "G_A": G_loss_A.item(),
            "G_B": G_loss_B.item(),
            "D_A": D_loss_A.item(),
            "D_B": D_loss_B.item(),
            "cycle_A": G_cycle_loss_A.item(),
            "cycle_B": G_cycle_loss_B.item(),
            "idt_A": G_recon_loss_A.item(),
            "idt_B": G_recon_loss_B.item()
        }
        return outputs, losses
    
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
            return visualizer.plot_gan_seg_sample(
                real_A=mini_batch["real_A"][0],
                fake_B=outputs["fake_B"][0],
                fake_B_seg=outputs["prediction"][0],
                real_B=mini_batch["real_B"][0],
                idt_B=outputs["idt_B"][0],
                real_B_seg=outputs["real_B_seg"][0],
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
