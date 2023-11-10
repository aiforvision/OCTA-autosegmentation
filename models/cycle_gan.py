from models.base_model_abc import BaseModelABC, Output
from utils.decorators import overrides
import torch
from typing import Any, Callable, Tuple
from utils.visualizer import Visualizer
from utils.enums import Phase
from utils.losses import get_loss_function_by_name
import torch.nn as nn
from monai.data import decollate_batch

class CycleGAN(BaseModelABC):
    """
    Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    def __init__(self,
                phase: Phase,
                MODEL_DICT: dict,
                inference:str,
                netG_A_config: dict,
                netG_B_config: dict,
                netD_A_config: dict,
                netD_B_config: dict,
                lambda_A: float,
                lambda_B: float,
                lambda_idt: float,
                pool_size: int,
                *args, **kwargs) -> None:
        super().__init__(optimizer_mapping={"optimizer_G": ["netG_A", "netG_B"], "optimizer_D": ["netD_A", "netD_B"]}, *args, **kwargs)

        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_idt = lambda_idt

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A: nn.Module = None
        self.netG_B: nn.Module = None
        self.netD_A: nn.Module = None
        self.netD_B: nn.Module = None
        self.optimizer_G: torch.optim.Optimizer
        self.optimizer_D: torch.optim.Optimizer
        if phase == Phase.TRAIN or inference == "netG_A":
            self.netG_A = MODEL_DICT[netG_A_config.pop("name")](**netG_A_config)
        if phase == Phase.TRAIN or inference == "netG_B":
            self.netG_B = MODEL_DICT[netG_B_config.pop("name")](**netG_B_config)

        if phase == Phase.TRAIN:
            # define discriminators
            self.netD_A = MODEL_DICT[netD_A_config.pop("name")](**netD_A_config)
            self.netD_B = MODEL_DICT[netD_B_config.pop("name")](**netD_B_config)

            self.fake_A_pool = ImagePool(pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(pool_size)  # create image buffer to store previously generated images

    @overrides(BaseModelABC)
    def initialize_model_and_optimizer(self, init_weights: Callable, config: dict, args, scaler, phase: Phase=Phase.TRAIN):
        self.loss_name_criterionGAN = config[Phase.TRAIN]["loss_criterionGAN"]
        self.criterionGAN = get_loss_function_by_name(self.loss_name_criterionGAN, config)

        self.loss_name_criterionCycle = config[Phase.TRAIN]["loss_criterionCycle"]
        self.criterionCycle = get_loss_function_by_name(self.loss_name_criterionCycle, config)

        self.loss_name_criterionIdt= config[Phase.TRAIN]["loss_criterionIdt"]
        self.criterionIdt = get_loss_function_by_name(self.loss_name_criterionIdt, config)

        super().initialize_model_and_optimizer(init_weights,config,args,scaler,phase)

    
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
        if self.netG_A is not None:
            return self.netG_A(input)
        else:
            return self.netG_B(input)
    
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

            if self.netG_B is not None and phase == Phase.VALIDATION:
                labels: torch.Tensor = mini_batch["label"].to(device=device, non_blocking=True)
                outputs["label"] = [post_transformations["label"](i) for i in decollate_batch(labels[0:1,0:1])]
                losses[self.loss_name_criterionCycle] = self.criterionCycle(pred, labels)
            return outputs, losses
        raise NotImplementedError("This inference function only supports val and test. Use perform_step for training")
    
    def _backward_D_basic(self, netD: nn.Module, real: torch.Tensor, fake: torch.Tensor):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D
        

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
        background: torch.Tensor = mini_batch["background"].to(device, non_blocking=True) if "background" in mini_batch else torch.rand_like(real_A, device=device, dtype=real_A.dtype)
        background = background*torch.zeros_like(real_A, device=device, dtype=real_A.dtype).uniform_(0,1)
        
        ####################################
        with torch.cuda.amp.autocast():
            fake_B: torch.Tensor = self.netG_A(torch.maximum(real_A,background))  # G_A(A)
            rec_A: torch.Tensor = self.netG_B(fake_B)   # G_B(G_A(A))
            fake_A: torch.Tensor = self.netG_B(real_B)  # G_B(B)
            rec_B: torch.Tensor = self.netG_A(torch.maximum(fake_A,background))   # G_A(G_B(B))

            # G_A and G_B
            self.netD_A.requires_grad_(False)
            self.netD_B.requires_grad_(False)
            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            # Identity loss
            if self.lambda_idt > 0:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                idt_A: torch.Tensor = self.netG_A(real_B)
                loss_idt_A = self.criterionIdt(idt_A, real_B) * self.lambda_B * self.lambda_idt
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                idt_B: torch.Tensor = self.netG_B(real_A)
                loss_idt_B = self.criterionIdt(idt_B, real_A) * self.lambda_A * self.lambda_idt
            else:
                loss_idt_A = 0
                loss_idt_B = 0

            # GAN loss D_A(G_A(A))
            loss_G_A = self.criterionGAN(self.netD_A(fake_B), True)
            # GAN loss D_B(G_B(B))
            loss_G_B = self.criterionGAN(self.netD_B(fake_A), True)
            # Forward cycle loss || G_B(G_A(A)) - A||
            loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.lambda_A
            # Backward cycle loss || G_A(G_B(B)) - B||
            loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_B
            # combined loss and calculate gradients
            loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        scaler.scale(loss_G).backward()
        scaler.step(self.optimizer_G)       # update G_A and G_B's weights
        ####################################

        ####################################
        # D_A and D_B
        self.netD_A.requires_grad_(True)
        self.netD_B.requires_grad_(True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero

        _fake_B = self.fake_B_pool.query(fake_B)
        with torch.cuda.amp.autocast():
            loss_D_A = self._backward_D_basic(self.netD_A, real_B, _fake_B)
        scaler.scale(loss_D_A).backward()
        _fake_A = self.fake_A_pool.query(fake_A)
        with torch.cuda.amp.autocast():
            loss_D_B = self._backward_D_basic(self.netD_B, real_A, _fake_A)
        scaler.scale(loss_D_B).backward()

        scaler.step(self.optimizer_D)  # update D_A and D_B's weights
        ####################################
        scaler.update()

        outputs: Output = {
            "prediction": [post_transformations["prediction"](i) for i in decollate_batch(rec_A[0:1, 0:1])],
            "label": [post_transformations["label"](i) for i in decollate_batch(real_A[0:1, 0:1])],
            "fake_B": fake_B[0:1,0:1].detach(),
            "idt_A": idt_A[0:1,0:1].detach(),
            "real_B": fake_A[0:1,0:1].detach()
        }
        losses = {
            "G": loss_G.item(),
            "G_A": loss_G_A.item(),
            "G_B": loss_G_B.item(),
            "D_A": loss_D_A.item(),
            "D_B": loss_D_B.item(),
            "cycle_A": loss_cycle_A.item(),
            "cycle_B": loss_cycle_B.item(),
            "idt_A": loss_idt_A.item(),
            "idt_B": loss_idt_B.item()
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
                mini_batch["real_A"][0],
                outputs["fake_B"][0],
                outputs["prediction"][0],
                mini_batch["real_B"][0],
                outputs["idt_A"][0],
                outputs["real_B"][0],
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
    



import random
class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
