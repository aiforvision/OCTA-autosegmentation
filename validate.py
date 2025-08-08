import argparse
import json
import os

import torch
import yaml
from data.image_dataset import get_dataset, get_post_transformation
from models.model import define_model
from models.networks import init_weights
from rich.console import Group
from rich.live import Live
from rich.progress import Progress, TimeElapsedColumn
from rich.spinner import Spinner
from utils.config_overrides import apply_cli_overrides_from_unknown_args
from utils.enums import Phase
from utils.metrics import MetricsManager
from utils.visualizer import DynamicDisplay

group = Group()

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--epoch', type=str, default='best')
parser.add_argument('--num_workers', type=int, default=None, help="Number of cpu cores used for dataloading. By, use half of the available cores.")
args, _unknown_args = parser.parse_known_args()

# Read config file
path: str = os.path.abspath(args.config_file)
assert os.path.isfile(path), f"Your provided config path {args.config_file} does not exist!"
with open(path, "r") as stream:
    if path.endswith(".json"):
        config = json.load(stream)
    else:
        config = yaml.safe_load(stream)

# Apply CLI overrides before using config
apply_cli_overrides_from_unknown_args(config, _unknown_args)

config[Phase.VALIDATION]["batch_size"]=1

with Live(group, refresh_per_second=10):
    with DynamicDisplay(group, Spinner("bouncingBall", text="Loading validation data...")):
        val_loader = get_dataset(config, Phase.VALIDATION, num_workers=args.num_workers)
        post_transformations_val = get_post_transformation(config, phase=Phase.VALIDATION)
        init_mini_batch = next(iter(val_loader))

    device = torch.device(config["General"].get("device") or "cpu")

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    with DynamicDisplay(group, Spinner("bouncingBall", text="Initializing model...")):
        model = define_model(config, phase=Phase.VALIDATION)
        model.initialize_model_and_optimizer(init_mini_batch, init_weights, config, args, scaler, phase=Phase.VALIDATION)

    metrics = MetricsManager(Phase.VALIDATION)
    predictions = []

    model.eval()
    progress = Progress(*Progress.get_default_columns(), TimeElapsedColumn())
    progress.add_task("Validating:", total=len(val_loader))
    with DynamicDisplay(group, progress):
        with torch.no_grad():
            for val_mini_batch in val_loader:
                outputs, losses = model.inference(val_mini_batch, post_transformations_val, device=device, phase=Phase.VALIDATION)
                model.compute_metric(outputs, metrics)
                progress.advance(task_id=0)
                
metrics = {k: float(str(round(v, 3))) for k,v in metrics.aggregate_and_reset(Phase.VALIDATION).items()}
print(f'Metrics: {metrics}')
