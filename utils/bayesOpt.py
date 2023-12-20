import argparse
import json
import torch
import os
import yaml
from monai.data import decollate_batch
import numpy as np
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.model import define_model
from data.image_dataset import get_dataset, get_post_transformation
from utils.metrics import MetricsManager, Task

from ray import tune,init
from ray.air import session, FailureConfig, RunConfig
from ray.tune import CLIReporter
import ConfigSpace as CS
from ray.tune.search.bohb import TuneBOHB
from utils.enums import Phase


# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--debug_mode', action="store_true")
args = parser.parse_args()

if args.debug_mode:
    init(local_mode = True)

# Read config file
path: str = os.path.abspath(args.config_file)
with open(path, "r") as stream:
    if path.endswith(".json"):
        CONFIG = json.load(stream)
    else:
        CONFIG = yaml.safe_load(stream)

def training_function(config: dict):
    config[Phase.VALIDATION]["batch_size"]=1
    config[Phase.VALIDATION]["post_processing"]["prediction"][0]["threshold"] = config["threshold"]
    config[Phase.VALIDATION]["post_processing"]["prediction"][1]["min_size"] = config["min_size"]

    task: Task = config["General"]["task"]

    val_loader = get_dataset(config, Phase.VALIDATION)
    post_pred, post_label = get_post_transformation(config, phase=Phase.VALIDATION)

    device = torch.device(config["General"].get("device") or "cpu")
    model = define_model(config, phase=Phase.VALIDATION)
    metrics = MetricsManager()

    model.eval()
    # while True:
    with torch.no_grad():
        step = 0
        for val_data in val_loader:
            step += 1
            val_inputs, val_labels = (
                val_data["image"].to(device).float(),
                val_data["label"].to(device),
            )
            val_outputs: torch.Tensor = model(val_inputs)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            metrics(val_outputs, val_labels)

        session.report(metrics.aggregate_and_reset(Phase.VALIDATION)) 


METRIC = 'val_DSC'

config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformIntegerHyperparameter("min_size", lower=0, upper=64))
config_space.add_hyperparameter(CS.CategoricalHyperparameter("threshold", choices=list(np.arange(0.01,0.9,0.01))))

search_alg = TuneBOHB(
    space=config_space,
    metric=METRIC,
    mode="max",
    max_concurrent=20
)

num_samples = 100

reporter = CLIReporter(
    metric=METRIC,
    mode="max",
    sort_by_metric=True,
    max_report_frequency=10
)
reporter.add_metric_column(METRIC)

tuner = tune.Tuner(
        tune.with_resources(
            training_function,
            {"cpu": 0.5}
        ),
        # Config that is given to training function. Search parameters are added to the config by each trail individually
        param_space=CONFIG,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            num_samples=num_samples
        ),
        run_config= RunConfig(
            name="BayesOptSearch",
            local_dir=CONFIG["Output"]["save_dir"],
            progress_reporter=reporter,
            failure_config=FailureConfig(fail_fast=args.debug_mode)
        )
    )
results = tuner.fit()
print("Best Trail found was: ", results.get_best_result(metric=METRIC, mode='max').log_dir)