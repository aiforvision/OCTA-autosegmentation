from abc import ABC, abstractmethod
import torch
from typing import Any, Tuple
from utils.metrics import MetricsManager
from utils.visualizer import Visualizer
from torch.cuda.amp.grad_scaler import GradScaler

class ModelInterface(ABC):
    """
    Standard interfacce each model needs to implement.
    Takes care of basic functionality, e.g., initialization and performing a training step.
    """

    @abstractmethod
    def initialize_model_and_optimizer(self, init_weights: function, config: dict, args, scaler: GradScaler, phase="train") -> None:
        raise NotImplementedError
    
    @abstractmethod
    def eval(self):
        raise NotImplementedError
    
    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def compute_metric(self, outputs: dict[str, Any], metrics: MetricsManager) -> None:
        """
        Computes the intermediate metric results for the outputs of a single mini_batch.

        Parameters:
        -----------
        - outputs: Dictionary containing the outputs and their names
        - metrics: MetricsManager
        """
        raise NotImplementedError()

    @abstractmethod
    def perform_step(self,
            mini_batch: dict[str, Any],
            scaler: torch.cuda.amp.grad_scaler.GradScaler,
            post_transformations: dict[str, function],
            device: torch.device = "cpu",
            optimize=False
        ) -> Tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """
        Computes the output and losses of a single mini_batch.

        Parameters:
        -----------
        - mini_batch: Dictionary containing the inputs and their names
        - scaler: GradScaler
        - post_transformations: Dictionary containing the post transformation for every output
        - device: Device on which to compute
        - optimize: If True, optimizes the weights. Default is False
        
        Returns:
        --------
        - Dictionary containing the outputs and their names
        - Dictionary containing the losses and their names
        """
        raise NotImplementedError()
    
    @abstractmethod
    def plot_sample(self, visualizer: Visualizer, mini_batch: dict[str, Any], outputs: dict[str, Any], epoch: int, *, prefix: str = "", save_interval: int=1):
        """
        Plots a sample for the given mini_batch each save_interval
        """
        raise NotImplementedError()
