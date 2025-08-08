import argparse
import json
import torch
import os

from monai.utils import set_determinism
import yaml
from models.model import define_model
from models.networks import init_weights
from data.image_dataset import get_dataset, get_post_transformation

from utils.visualizer import plot_sample, plot_single_image, DynamicDisplay
from utils.enums import Phase

from rich.live import Live
from rich.progress import Progress, TimeElapsedColumn
from rich.spinner import Spinner
from rich.console import  Group, Console
group = Group()

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--epoch', type=str, default="best")
parser.add_argument('--num_samples', type=int, default=9999999)
parser.add_argument('--num_workers', type=int, default=None, help="Number of cpu cores used for dataloading. By, use half of the available cores.")
args = parser.parse_args()
epoch_suffix = f"_{args.epoch}"
assert args.num_samples>0

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

device = torch.device(config["General"].get("device") or "cpu")
scaler = torch.amp.GradScaler(enabled=False)
# set_determinism(seed=0)

with Live(group, console=Console(force_terminal=True), refresh_per_second=10):
    with DynamicDisplay(group, Spinner("bouncingBall", text="Loading test data...")):
        test_loader = get_dataset(config, Phase.TEST, num_workers=args.num_workers)
        post_transformations_test = get_post_transformation(config, Phase.TEST)
        test_mini_batch = next(iter(test_loader))
        input_key = [k for k in test_mini_batch.keys() if not k.endswith("_path")][0]
        test_mini_batch["image"] = test_mini_batch.pop(input_key)

    with DynamicDisplay(group, Spinner("bouncingBall", text="Initializing model...")):
        model = define_model(config, phase=Phase.TEST)
        model.initialize_model_and_optimizer(test_mini_batch, init_weights, config, args, scaler, phase=Phase.TEST)
    predictions = []

    model.eval()
    progress = Progress(*Progress.get_default_columns(), TimeElapsedColumn())
    progress.add_task("Testing:", total=min(len(test_loader), args.num_samples))
    with DynamicDisplay(group, progress):
        with torch.no_grad():
            num_sample=0
            for test_mini_batch in test_loader:
                if num_sample>=args.num_samples:
                    break
                num_sample+=1
                test_mini_batch["image"] = test_mini_batch.pop(input_key)
                outputs, _ = model.inference(test_mini_batch, post_transformations_test, device=device, phase=Phase.TEST)
                inference_mode = config["General"].get("inference") or "pred"
                image_name: str = test_mini_batch[f"{input_key}_path"][0].split("/")[-1]
                
                # plot_single_image(save_dir, inputs[0], image_name)
                plot_single_image(save_dir, outputs["prediction"][0].cpu(), inference_mode + "_" + image_name)
                if config["Output"].get("save_comparisons"):
                    plot_sample(save_dir, test_mini_batch[input_key][0], outputs["prediction"][0].cpu(), None, test_mini_batch[f"{input_key}_path"][0], suffix=f"{inference_mode}_{image_name}", full_size=True)
                progress.advance(task_id=0)
