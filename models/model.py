from typing import Literal
import torch
from models.model_interface_abc import ModelInterface
from models.lambda_model import LambdaModel
from models.networks import MODEL_DICT

def define_model(config: dict[str, dict], phase: Literal["train", "val", "test"]):
    device = torch.device(config["General"].get("device") or "cpu")
    model_params: dict = config["General"]["model"]
    model_name = model_params.pop("name")
    model_params["phase"]=phase
    model_params["MODEL_DICT"]=MODEL_DICT
    model_params["inference"] = config["General"].get("inference")
    if issubclass(MODEL_DICT[model_name], ModelInterface):
        model = MODEL_DICT[model_name](**model_params).to(device, non_blocking=True)
    else:
        model = LambdaModel(model_name,**model_params).to(device, non_blocking=True)
    return model
