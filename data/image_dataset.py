from monai.data import DataLoader, Dataset
from data.data_transforms import *
from numpy import array
import torch
import numpy as np
from glob import glob
from natsort import natsorted
import os

from monai.data.meta_obj import set_track_meta
from multiprocessing import cpu_count
from math import ceil

from utils.metrics import Task
from data.unalignedZipDataset import UnalignedZipDataset
set_track_meta(False)

def _get_transformation(config, phase: str, dtype=torch.float32) -> Compose:
    """
    Create and return the data transformations for 2D segmentation images the given phase.
    """
    aug_config = config[phase.capitalize()]["data_augmentation"]
    return Compose(get_data_augmentations(aug_config, dtype))

def get_post_transformation(config: dict, phase: str) -> dict[str, Compose]:
    """
    Create and return the data transformation that is applied to the label and the model prediction before inference.
    """
    aug_config: dict = config[phase.capitalize()]["post_processing"]
    post_transformations = dict()
    for k,v in aug_config.items():
        try:
            post_transformations[k] =  Compose(get_data_augmentations(v))
        except Exception as e:
            print("Error: Your provided data augmentations for prediction are invalid.\n")
            raise e
    return post_transformations


def get_dataset(config: dict[str, dict], phase: str, batch_size=None) -> DataLoader:
    """
    Creates and return the dataloader for the given phase.
    """
    task = config["General"]["task"]
    transform = _get_transformation(config, phase, dtype=torch.float16 if phase=="train" and bool(config["General"].get("amp")) else torch.float32)

    data_settings: dict = config[phase.capitalize()]["data"]
    data = dict()
    for key, val in data_settings.items():
        paths = natsorted(glob(val["files"], recursive=True))
        assert len(paths)>0, f"Error: Your provided file path {val['files']} for {key} does not match any files!"
        if "split" in val:
            assert os.path.isfile(val["split"]), f"Error: Your provided split file path {val['split']} for {key} does not exist."
            with open(val["split"], 'r') as f:
                lines = f.readlines()
                indices = [int(line.rstrip()) for line in lines]
                assert max(indices)<len(paths), f"Error: Your provided split file for {key} does not seem to match your dataset! The index {max(indices)} was requested but the dataset only contains {len(paths)} files."
                paths = array(paths)[indices].tolist()
                assert len(paths)>0, f"Error: Your provided split file does not reference any file!"
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
    loader = DataLoader(data_set, batch_size=batch_size or config[phase.capitalize()].get("batch_size") or 1, shuffle=phase!="test", num_workers=ceil(cpu_count()/2), pin_memory=torch.cuda.is_available())
    return loader
