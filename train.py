import argparse
import os
import yaml

from utils.metrics import Task
from monai.utils import set_determinism
from random import randint
from utils.train_ves_seg import vessel_segmentation_train
from utils.train_gan_seg import gan_vessel_segmentation_train

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epoch', type=str, default='latest')
parser.add_argument('--split', type=str, default='')
args = parser.parse_args()

# Read config file
path: str = os.path.abspath(args.config_file)
with open(path, "r") as stream:
    config: dict[str,dict] = yaml.safe_load(stream)

if "seed" not in config["General"]:
    config["General"]["seed"] = randint(0,1e6)
set_determinism(seed=config["General"]["seed"])

if config["General"]["task"] == Task.GAN_VESSEL_SEGMENTATION:
    gan_vessel_segmentation_train(args, config)
elif config["General"]["task"] == Task.VESSEL_SEGMENTATION:
    vessel_segmentation_train(args, config)
else:
    raise NotImplementedError("Task {} does not exist!".format(config["General"]["task"]))