from monai.data import DataLoader, Dataset
from data.data_transforms import *
import os
from numpy import array
import torch
import numpy as np

from monai.data.meta_obj import set_track_meta

from utils.metrics import Task
from data.unalignedZipDataset import UnalignedZipDataset
set_track_meta(False)

def get_custom_file_paths(folder: str, name: str) -> list[str]:
    image_file_paths = []
    for root, _, filenames in os.walk(folder):
        filenames: list[str] = sorted(filenames)
        for filename in filenames:
            if filename.lower().endswith(name.lower()):
                file_path = os.path.join(root, filename)
                image_file_paths.append(file_path)
    return image_file_paths

def _get_transformation(config, task: Task, phase: str, dtype=torch.float32) -> Compose:
    """
    Create and return the data transformations for 2D segmentation images the given phase.
    """
    if task == Task.VESSEL_SEGMENTATION or task == Task.GAN_VESSEL_SEGMENTATION:
        aug_config = config[phase.capitalize()]["data_augmentation"]
        return Compose(get_data_augmentations(aug_config, dtype))
    else:
        raise NotImplementedError("Task: "+ task)

def get_post_transformation(config: dict, phase: str) -> tuple[Compose]:
    """
    Create and return the data transformation that is applied to the model prediction before inference.
    """
    aug_config: dict = config[phase.capitalize()]["post_processing"]
    return Compose(get_data_augmentations(aug_config.get("prediction"))), Compose(get_data_augmentations(aug_config.get("label")))


def get_dataset(config: dict[str, dict], phase: str, batch_size=None) -> DataLoader:
    """
    Creates and return the dataloader for the given phase.
    """
    task = config["General"]["task"]
    transform = _get_transformation(config, task, phase, dtype=torch.float16 if bool(config["General"].get("amp")) else torch.float32)

    data_settings: dict = config[phase.capitalize()]["data"]
    data = dict()
    for key, val in data_settings.items():
        paths  = get_custom_file_paths(val["folder"], val["suffix"])
        paths.sort()
        if "split" in val:
            with open(val["split"], 'r') as f:
                lines = f.readlines()
                indices = [int(line.rstrip()) for line in lines]
                paths = array(paths)[indices].tolist()
        data[key] = paths
        data[key+"_path"] = paths

    if task == Task.VESSEL_SEGMENTATION:
        max_length = max([len(l) for l in data.values()])
        for k,v in data.items():
            data[k] = np.resize(np.array(v), max_length).tolist()
        train_files = [dict(zip(data, t)) for t in zip(*data.values())]
    elif task == Task.GAN_VESSEL_SEGMENTATION:
        if phase == "validation":
            max_length = max([len(l) for l in data.values()])
            for k,v in data.items():
                data[k] = np.resize(np.array(v), max_length).tolist()
            train_files = [dict(zip(data, t)) for t in zip(*data.values())]
        else:
            data_set = UnalignedZipDataset(data, transform, phase, config["General"]["inference"])
            loader = DataLoader(data_set, batch_size=batch_size or config[phase.capitalize()].get("batch_size") or 1, shuffle=phase!="test", num_workers=8, pin_memory=torch.cuda.is_available())
            return loader



    data_set = Dataset(train_files, transform=transform)
    loader = DataLoader(data_set, batch_size=batch_size or config[phase.capitalize()].get("batch_size") or 1, shuffle=phase!="test", num_workers=32, pin_memory=torch.cuda.is_available())
    return loader
