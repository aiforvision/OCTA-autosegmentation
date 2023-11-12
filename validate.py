import argparse
import json
import torch
import os
from tqdm import tqdm
import yaml

from models.model import define_model
from models.networks import init_weights

from data.image_dataset import get_dataset, get_post_transformation
from utils.metrics import MetricsManager, Task
from utils.enums import Phase

# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--epoch', type=str, default='best')
args = parser.parse_args()

# Read config file
path: str = os.path.abspath(args.config_file)
assert os.path.isfile(path), f"Your provided config path {args.config_file} does not exist!"
with open(path, "r") as stream:
    if path.endswith(".json"):
        config = json.load(stream)
    else:
        config = yaml.safe_load(stream)

config[Phase.VALIDATION]["batch_size"]=1

task: Task = config["General"]["task"]

val_loader = get_dataset(config, Phase.VALIDATION)
post_transformations_val = get_post_transformation(config, phase=Phase.VALIDATION)

device = torch.device(config["General"].get("device") or "cpu")

scaler = torch.cuda.amp.GradScaler(enabled=False)

model = define_model(config, phase=Phase.VALIDATION)
model.initialize_model_and_optimizer(next(iter(val_loader)), init_weights, config, args, scaler, phase=Phase.VALIDATION)

metrics = MetricsManager(Phase.VALIDATION)
predictions = []

model.eval()
with torch.no_grad():
    for val_mini_batch in tqdm(val_loader, desc=Phase.VALIDATION.value):
        outputs, losses = model.inference(val_mini_batch, post_transformations_val, device=device, phase=Phase.VALIDATION)
        model.compute_metric(outputs, metrics)
            
    metrics = {k: float(str(round(v, 3))) for k,v in metrics.aggregate_and_reset(Phase.VALIDATION).items()}
    print(f'Metrics: {metrics}')
