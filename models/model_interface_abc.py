from abc import ABC, abstractmethod
import torch
from typing import Any, Tuple, Callable, TypedDict, Literal, NotRequired
from utils.metrics import MetricsManager
from utils.visualizer import Visualizer
from torch.cuda.amp.grad_scaler import GradScaler

class Output(TypedDict):
    prediction: list
    label: NotRequired[list]

class ModelInterface(ABC):
    """
    Standard interfacce each model needs to implement.
    Takes care of basic functionality, e.g., initialization and performing a training step.
    """

    @abstractmethod
    def initialize_model_and_optimizer(self, init_weights: Callable, config: dict, args, scaler: GradScaler, phase="train") -> None:
        raise NotImplementedError
    
    @abstractmethod
    def eval(self):
        raise NotImplementedError
    
    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def compute_metric(self, outputs: Output, metrics: MetricsManager) -> None:
        """
        Computes the intermediate metric results for the outputs of a single mini_batch.

        Parameters:
        -----------
        - outputs: Dictionary containing the outputs and their names
        - metrics: MetricsManager
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
    
    @abstractmethod
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
        - Dictionary containing the losses and their names
        """
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()
    
    @abstractmethod
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
        raise NotImplementedError()
