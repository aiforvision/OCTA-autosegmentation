import argparse
import json
import torch
import os
from tqdm import tqdm

from monai.utils import set_determinism
import yaml
from models.model import define_model
from models.networks import init_weights

from data.image_dataset import get_dataset, get_post_transformation
from utils.metrics import Task

from utils.visualizer import plot_sample, plot_single_image
from utils.enums import Phase

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--epoch', type=str, default="best")
args = parser.parse_args()
epoch_suffix = f"_{args.epoch}"

# Read config file
path: str = os.path.abspath(args.config_file)
assert os.path.isfile(path), f"Your provided config path {args.config_file} does not exist!"
with open(path, "r") as stream:
    if path.endswith(".json"):
        config: dict[str,dict] = json.load(stream)
    else:
        config: dict[str,dict] = yaml.safe_load(stream)
if config["General"].get("seed") is not None:
    set_determinism(seed=config["General"]["seed"])

inference_suffix = "_"+config["General"]["inference"] if "inference" in config["General"] else ""
save_dir = config[Phase.TEST].get("save_dir") or config["Output"]["save_dir"]+"/test"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# set_determinism(seed=0)

task: Task = config["General"]["task"]

test_loader = get_dataset(config, Phase.TEST)
post_transformations_test = get_post_transformation(config, Phase.TEST)

device = torch.device(config["General"].get("device") or "cpu")

scaler = torch.cuda.amp.GradScaler(enabled=False)
model = define_model(config, phase=Phase.TEST)
model.initialize_model_and_optimizer(init_weights, config, args, scaler, phase=Phase.VALIDATION)
predictions = []

model.eval()
with torch.no_grad():
    num_sample=0
    for test_mini_batch in tqdm(test_loader, desc="Testing", total=min(len(test_loader), config[Phase.TEST].get("num_samples") or 9999999)):
        if config[Phase.TEST].get("num_samples") is not None and num_sample>=config[Phase.TEST]["num_samples"]:
            break
        num_sample+=1
        input_key = [k for k in test_mini_batch.keys() if not k.endswith("_path")][0]
        test_mini_batch["image"] = test_mini_batch.pop(input_key)
        outputs, _ = model.inference(test_mini_batch, post_transformations_test, device=device, phase=Phase.TEST)
        inference_mode = config["General"].get("inference") or "pred"
        image_name: str = test_mini_batch[f"{input_key}_path"][0].split("/")[-1]
        
        # plot_single_image(save_dir, inputs[0], image_name)
        plot_single_image(save_dir, outputs["prediction"][0], inference_mode + "_" + image_name)
        if config["Output"].get("save_comparisons"):
            plot_sample(save_dir, test_mini_batch[input_key][0], outputs["prediction"][0], None, test_mini_batch[f"{input_key}_path"][0], suffix=f"{inference_mode}_{image_name}", full_size=True)
