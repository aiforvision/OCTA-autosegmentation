import argparse
import json
import torch
import os
from tqdm import tqdm

from monai.data import decollate_batch
from monai.utils import set_determinism
import yaml
from models.model import define_model, initialize_model_and_optimizer

from data.image_dataset import get_dataset, get_post_transformation
from utils.metrics import Task

from utils.visualizer import plot_sample, plot_single_image

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--epoch', type=str, default="best")
args = parser.parse_args()
epoch_suffix = f"_{args.epoch}"

# Read config file
path: str = os.path.abspath(args.config_file)
with open(path, "r") as stream:
    if path.endswith(".json"):
        config: dict[str,dict] = json.load(stream)
    else:
        config: dict[str,dict] = yaml.safe_load(stream)
set_determinism(seed=config["General"]["seed"])

inference_suffix = "_"+config["General"]["inference"] if "inference" in config["General"] else ""
save_dir = config["Test"].get("save_dir") or config["Output"]["save_dir"]+"/test"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# set_determinism(seed=0)

task: Task = config["General"]["task"]

test_loader = get_dataset(config, "test")
post_pred, _ = get_post_transformation(config, "test")

device = torch.device(config["General"].get("device") or "cpu")

model = define_model(config, phase="test")
_ = initialize_model_and_optimizer(model, config, args, phase="test")
predictions = []

model.eval()
with torch.no_grad():
    num_sample=0
    for test_data in tqdm(test_loader, desc="Testing", total=min(len(test_loader), config["Test"].get("num_samples") or 9999999)):
        if config["Test"].get("num_samples") is not None and num_sample>=config["Test"]["num_samples"]:
            break
        num_sample+=1
        input_key = [k for k in test_data.keys() if not k.endswith("_path")][0]
        inputs = test_data[input_key].to(device).float()
        outputs = model(inputs)
        outputs = [post_pred(i).cpu() for i in decollate_batch(outputs)]

        if task == Task.VESSEL_SEGMENTATION or task == Task.GAN_VESSEL_SEGMENTATION:
            inference_mode = config["General"].get("inference") or "pred"
            image_name: str = test_data[input_key+"_path"][0].split("/")[-1]
            
            # plot_single_image(save_dir, inputs[0], image_name)
            plot_single_image(save_dir, outputs[0], inference_mode + "_" + image_name)
            if config["Output"].get("save_comparisons"):
                plot_sample(save_dir, inputs[0], outputs[0], None, test_data[input_key+"_path"][0], suffix=f"{inference_mode}_{image_name}", full_size=True)
