from models.model_interface_abc import ModelInterface
from models.base_model_abc import BaseModelABC
from typing import Any, Tuple, Literal
import torch
from utils.decorators import overrides
from utils.losses import get_loss_function_by_name
from utils.metrics import MetricsManager
from utils.visualizer import Visualizer
from torch.cuda.amp.grad_scaler import GradScaler

class LambdaModel(BaseModelABC):

    def __init__(self, model_name: str, phase: Literal["train", "val", "test"], MODEL_DICT: dict, inference: str, **kwargs) -> None:
        super().__init__(optimizer_mapping={"optimizer", "model"})
        self.model = MODEL_DICT[model_name](**kwargs)

    overrides(BaseModelABC)
    def initialize_model_and_optimizer(self, init_weights: function, config: dict, args, scaler: GradScaler, phase="train") -> None:
        self.loss_name = config["Train"]["loss"]
        self.loss_function = get_loss_function_by_name(self.loss_name, config)
        if config["Train"].get("AT") is not None:
            self.at = get_loss_function_by_name("AtLoss", config, scaler, self.loss_function)
        super().initialize_model_and_optimizer(init_weights, config, args, scaler, phase)

    overrides(ModelInterface)
    def forward(self, mini_batch: dict[str, Any], device: torch.device) -> Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        inputs, labels = (
            mini_batch["image"].to(device),
            mini_batch["label"].to(device),
        )
        if hasattr(self, "at"):
            inputs, labels = self.at(self.model, inputs, mini_batch["background"].to(device), labels)
        pred = self.model(inputs).squeeze(-1)
        loss = self.loss_function(y_pred=pred, y=labels)
        return { "pred": pred, "label": labels }, { f"{self.loss_name}": loss }
    
    overrides(ModelInterface)
    def compute_metric(self, outputs: dict[str, Any], metrics: MetricsManager) -> None:
        metrics(outputs["pred"], outputs["label"])
    
    overrides(ModelInterface)
    def plot_sample(visualizer: Visualizer,
            mini_batch: dict[str, Any],
            outputs: dict[str, Any],
            epoch: int,
            *,
            prefix: str = ""
        ) -> str:
        return visualizer.plot_sample(
            mini_batch["image"][0],
            outputs["pred"][0],
            outputs["label"][0],
            suffix=(epoch + 1) if not prefix else f"{prefix}_{(epoch + 1)}",
        )
