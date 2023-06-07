from typing import Union, Literal
import torch
from models.gan_seg_model import GanSegModel
from utils.metrics import Task
import os

from models.networks import MODEL_DICT, init_weights

def define_model(config: dict[str, dict], phase: Literal["train", "val", "test"]):
    device = torch.device(config["General"].get("device") or "cpu")
    model_params: dict = config["General"]["model"]
    model_name = model_params.pop("name")
    if config["General"]["task"] == Task.GAN_VESSEL_SEGMENTATION:
        model_params["phase"]=phase
        model_params["MODEL_DICT"]=MODEL_DICT
        model_params["inference"] = config["General"]["inference"]
    model = MODEL_DICT[model_name](**model_params)
    if isinstance(model, torch.nn.Module):
        model = model.to(device)
    return model

def initialize_model_and_optimizer(model: torch.nn.Module, config: dict, args, phase="train") -> Union[torch.optim.Optimizer, tuple[torch.optim.Optimizer]]:
    if not isinstance(model, torch.nn.Module):
        return None
    
    task = config["General"]["task"]
    model_path: str = os.path.join(config["Output"]["save_dir"], "checkpoints", f"{args.epoch}_model.pth")
    if task == Task.GAN_VESSEL_SEGMENTATION:
        model: GanSegModel = model
        if phase == "train":
            optimizer_G = torch.optim.Adam(model.generator.parameters(), lr=config["Train"]["lr"], betas=(0.5 , 0.999))
            optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=config["Train"]["lr"], betas=(0.5 , 0.999))
            optimizer_S = torch.optim.Adam(model.segmentor.parameters(), lr=config["Train"]["lr"])
            if hasattr(args, "start_epoch") and args.start_epoch>0:
                checkpoint_G = torch.load(model_path.replace('model.pth', 'G_model.pth'), map_location=torch.device(config["General"]["device"]))
                model.generator.load_state_dict(checkpoint_G['model'])
                optimizer_G.load_state_dict(checkpoint_G['optimizer'])

                checkpoint_D = torch.load(model_path.replace('model.pth', 'D_model.pth'), map_location=torch.device(config["General"]["device"]))
                model.discriminator.load_state_dict(checkpoint_D['model'])
                optimizer_D.load_state_dict(checkpoint_D['optimizer'])

                checkpoint_S = torch.load(model_path.replace('model.pth', 'S_model.pth'), map_location=torch.device(config["General"]["device"]))
                model.segmentor.load_state_dict(checkpoint_S['model'])
                optimizer_S.load_state_dict(checkpoint_S['optimizer'])
            elif task == Task.GAN_VESSEL_SEGMENTATION: 
                for m in [model.generator, model.discriminator, model.segmentor]:
                    activation = 'relu' if m._get_name().lower().startswith("resnet") else 'leaky_relu'
                    init_weights(m, init_type='kaiming', nonlinearity=activation)
            else:
                for m in [model.generator, model.discriminator]:
                    activation = 'relu' if (m._get_name().lower().startswith("resnet") or m._get_name().lower().startswith("patch")) else 'leaky_relu'
                    init_weights(m, init_type='kaiming', nonlinearity=activation)

            return (optimizer_G,optimizer_D,optimizer_S)
        else:
            checkpoint = torch.load(model_path.replace('model.pth', f'{config["General"]["inference"]}_model.pth'), map_location=torch.device(config["General"]["device"]))
            if config["General"]["inference"] == "S":
                model.segmentor.load_state_dict(checkpoint['model'])
            elif config["General"]["inference"] == "G":
                model.generator.load_state_dict(checkpoint['model'])
            else: raise NotImplementedError
        return None

    if hasattr(args, "start_epoch") and args.start_epoch>0:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.Adam(model.parameters(), config["Train"]["lr"], weight_decay=1e-6)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint from epoch", checkpoint['epoch']+1)
    elif phase != "train":
        checkpoint = torch.load(model_path, map_location=torch.device(config["General"]["device"]))
        model.load_state_dict(checkpoint['model'])
        optimizer=None
        print("Loaded checkpoint from epoch", checkpoint['epoch']+1)
    else:
        activation = 'relu' if model._get_name().lower().startswith("resnet") else 'leaky_relu'
        init_weights(model, init_type='kaiming', nonlinearity=activation)
        optimizer = torch.optim.Adam(model.parameters(), config["Train"]["lr"], weight_decay=1e-6)
    return optimizer
