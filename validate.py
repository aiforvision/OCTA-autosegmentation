import argparse
import json
import torch
import os
from tqdm import tqdm
import yaml
from monai.data import decollate_batch

from models.model import define_model, initialize_model_and_optimizer

from data.image_dataset import get_dataset, get_post_transformation
from utils.metrics import MetricsManager, Task

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

config["Validation"]["batch_size"]=1

task: Task = config["General"]["task"]

val_loader = get_dataset(config, 'validation')
post_pred, post_label = get_post_transformation(config, phase="validation")

device = torch.device(config["General"].get("device") or "cpu")

model = define_model(config, phase="val")
optimizer = initialize_model_and_optimizer(model, config, args, phase="val")

metrics = MetricsManager(task, "val")
predictions = []

model.eval()
with torch.no_grad():
    num_sample=0
    with torch.no_grad():
        step = 0
        for val_data in tqdm(val_loader, desc='Validation'):
            step += 1
            val_inputs, val_labels = (
                val_data["image"].to(device).float(),
                val_data["label"].to(device),
            )
            val_outputs: torch.Tensor = model(val_inputs)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            metrics(val_outputs, val_labels)
                
        metrics = {k: float(str(round(v, 3))) for k,v in metrics.aggregate_and_reset("val").items()}
        print(f'Metrics: {metrics}')
