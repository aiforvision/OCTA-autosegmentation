import os
import torch
from torch import nn
from typing import Any, Callable, Tuple, Literal
import itertools
from models.model_interface_abc import ModelInterface, Output
from utils.decorators import overrides
from abc import ABC, abstractmethod
from torch.cuda.amp.grad_scaler import GradScaler
from utils.metrics import MetricsManager

class BaseModelABC(nn.Module, ModelInterface, ABC):
    """
    Standard interfacce each model needs to implement.
    Takes care of basic functionality, e.g., initialization and performing a training step.
    """

    def __init__(self, optimizer_mapping=None,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer_mapping: dict[str, list[str]] = optimizer_mapping or { "optimizer": [] }

    @overrides(ModelInterface)
    def initialize_model_and_optimizer(self, init_weights: Callable, config: dict, args, scaler: GradScaler, phase="train"):
        """
        Initializes the model weights.
        If a pretrained model is used, the respective checkpoint will be loaded and all weights assigned (including the optimizer weights).
        The function returns the epoch of the loaded checkpoint, else None.
        """
        if not any([isinstance(getattr(self, net_name), nn.Module) for net_names in self.optimizer_mapping.values() for net_name in net_names]):
            pass
        model_path: str = os.path.join(config["Output"]["save_dir"], "checkpoints", f"{args.epoch}_model.pth")
        if phase == "train":
            # Initialize Optimizers
            for optim_name, net_names in self.optimizer_mapping.items():
                parameters = [self.parameters()] if len(net_names) == 0 else [getattr(self, net_name).parameters() for net_name in net_names]
                setattr(self,optim_name, torch.optim.Adam(itertools.chain(*parameters),
                    lr=config["Train"]["lr"],
                    betas=(0.5, 0.999),
                    weight_decay=config["Train"].get("weight_decay", 0)
                ))

            # Initialize LR scheduler TODO
            max_epochs = config["Train"]["epochs"]
            def schedule(step: int):
                if step < (max_epochs - config["Train"]["epochs_decay"]):
                    return 1
                else:
                    return (max_epochs-step) * (1/max(1,config["Train"]["epochs_decay"]))
            self.lr_schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
            for optimizer_name in self.optimizer_mapping.keys():
                self.lr_schedulers.append(torch.optim.lr_scheduler.LambdaLR(getattr(self, optim_name), schedule))

            # Initialize weights
            if hasattr(args, "start_epoch") and args.start_epoch>0:
                # Load from checkpoint
                for optimizer_name, net_names in self.optimizer_mapping.items():
                    if len(net_names)==0:
                        checkpoint = torch.load(model_path, map_location=torch.device(config["General"]["device"]))
                        self.load_state_dict(checkpoint["model"])
                    else:
                        for net_name in net_names:
                            net_path = model_path.replace('model.pth', f'{net_name}_model.pth')
                            checkpoint = torch.load(net_path, map_location=torch.device(config["General"]["device"]))
                            net: nn.Module = getattr(self, net_name)
                            net.load_state_dict(checkpoint["model"])
                    optimizer: torch.optim.Optimizer = getattr(self, optimizer_name)
                    if "optimizer" in checkpoint:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                    else:
                        optimizer_checkpoint = torch.load(model_path.replace('model.pth', f'{optimizer_name}.pth'), map_location=torch.device(config["General"]["device"]))
                        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
                    print(f"Loaded all network weights from epoch {checkpoint['epoch']}.")
            else:
                # Initialize weights according to init_weights function
                for net_name in [net_name for net_list in self.optimizer_mapping.values() for net_name in net_list]:
                    m: nn.Module = getattr(self, net_name)
                    activation = 'relu' if "resnet" in m._get_name().lower() else 'leaky_relu'
                    init_weights(m, init_type='kaiming', nonlinearity=activation)
                    print(f"Initialized {net_name} network weights")
        else:
            # Only load necessary network parts for inference
            checkpoint = torch.load(model_path.replace('model.pth', f'{config["General"]["inference"]}_model.pth'), map_location=torch.device(config["General"]["device"]))
            assert hasattr(self, config["General"]["inference"]), f'Inference mode {config["General"]["inference"]} not implemented.'
            net: nn.Module = getattr(self, config["General"]["inference"])
            net.load_state_dict(checkpoint['model'])
            print(f"Loaded network weights {config['General']['inference']} from epoch {checkpoint['epoch']}.")

    @abstractmethod
    @overrides(ModelInterface)
    def inference(self,
                mini_batch: dict[str, Any],
                post_transformations: dict[str, Callable],
                device: torch.device = "cpu",
                phase: Literal["train", "val", "test"] = "test"
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
        - Dictionary containing the loss values and their names
        """
        raise NotImplementedError()
    
    @abstractmethod
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
        raise NotImplementedError()
    
    @overrides(ModelInterface)
    def perform_training_step(self,
            mini_batch: dict[str, Any],
            scaler: torch.cuda.amp.grad_scaler.GradScaler,
            post_transformations: dict[str, Callable],
            device: torch.device = "cpu"
        ) -> Tuple[Output, dict[str, float]]:
        self.optimizer: torch.optim.Optimizer
        with torch.cuda.amp.autocast():
            outputs, losses = self.inference(mini_batch, post_transformations, device, phase="train")
            loss = sum(list(losses.values()))
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        losses = {k: v.item() for k,v in losses.items()}
        return outputs, losses
    
    @overrides(ModelInterface)
    def compute_metric(self, outputs: Output, metrics: MetricsManager) -> None:
        metrics(y_pred=outputs["prediction"], y=outputs["label"])
