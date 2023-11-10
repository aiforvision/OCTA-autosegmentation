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
from utils.losses import get_loss_function_by_name
from models.model import define_model, initialize_model_and_optimizer

from ray import tune,init
from ray.air import session, FailureConfig, RunConfig
from ray.tune import CLIReporter
import ConfigSpace as CS
from ray.tune.search.bohb import TuneBOHB
import copy
from utils.enums import Phase


# Parse input arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--debug_mode', action="store_true")
parser.add_argument('--epoch', default='latest', type=str)
args = parser.parse_args()

if args.debug_mode:
    init(local_mode = True, num_gpus=1)

# Read config file
path: str = os.path.abspath(args.config_file)
with open(path, "r") as stream:
    if path.endswith(".json"):
        CONFIG = json.load(stream)
    else:
        CONFIG = yaml.safe_load(stream)

def training_function(config: dict):
    config[Phase.VALIDATION]["batch_size"]=1
    config[Phase.TRAIN]["data_augmentation"][5]["max_decrease_res"] = config["max_decrease_res"]
    config[Phase.TRAIN]["data_augmentation"][5]["lambda_speckle"] = config["lambda_speckle"]
    # config[Phase.TRAIN]["data_augmentation"][5]["lambda_gamma"] = config["lambda_gamma"]
    config[Phase.TRAIN]["data_augmentation"][5]["lambda_delta"] = config["lambda_delta"]


    max_epochs = config[Phase.TRAIN]["epochs"]
    VAL_AMP = bool(config["General"].get("amp"))
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler(enabled=VAL_AMP)
    device = torch.device(config["General"].get("device") or "cpu")
    task: Task = config["General"]["task"]

    train_loader = get_dataset(config, Phase.TRAIN)
    val_loader = get_dataset(config, Phase.VALIDATION)

    post_pred, post_label = get_post_transformation(config, Phase.TRAIN)
    post_pred_val, post_label_val = get_post_transformation(config, Phase.VALIDATION)

    model = define_model(copy.deepcopy(config), Phase.TRAIN)

    optimizer = initialize_model_and_optimizer(model, config, args)

    loss_name = config[Phase.TRAIN]["loss"]
    loss_function = get_loss_function_by_name(loss_name, config)
    def schedule(step: int):
        if step < max_epochs - config[Phase.TRAIN]["epochs_decay"]:
            return 1
        else:
            return (max_epochs-step) * (1/max(1,config[Phase.TRAIN]["epochs_decay"]))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    metrics = MetricsManager()

    for epoch in range(max_epochs):
        model.train()
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                outputs=outputs.squeeze(-1)

                loss = loss_function(outputs, labels)
                labels = [post_label(i) for i in decollate_batch(labels)]
                outputs = [post_pred(i) for i in decollate_batch(outputs)]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        lr_scheduler.step()
        
        if epoch in range(max_epochs-3,max_epochs):
            model.eval()
            with torch.no_grad():
                step = 0
                for val_data in val_loader:
                    step += 1
                    val_inputs, val_labels = (
                        val_data["image"].to(device).float(),
                        val_data["label"].to(device),
                    )
                    val_outputs: torch.Tensor = model(val_inputs)
                    # if config["Data"]["num_classes"]==1:
                    val_outputs=val_outputs.squeeze(-1)
                        # val_labels=val_labels.to(dtype=val_outputs.dtype)
                    val_labels = [post_label_val(i) for i in decollate_batch(val_labels)]
                    val_outputs = [post_pred_val(i) for i in decollate_batch(val_outputs)]
                    metrics(y_pred=val_outputs, y=val_labels)

    session.report(metrics.aggregate_and_reset(Phase.VALIDATION)) 


METRIC = 'val_DSC'

config_space = CS.ConfigurationSpace()
# config_space.add_hyperparameter(CS.UniformFloatHyperparameter("lambda_delta", lower=0.5, upper=1.))
# config_space.add_hyperparameter(CS.UniformFloatHyperparameter("lambda_speckle", lower=0.5, upper=1.))

config_space.add_hyperparameter(CS.CategoricalHyperparameter("lambda_speckle", choices=list(np.arange(0.3,0.71,0.1))))
# config_space.add_hyperparameter(CS.CategoricalHyperparameter("lambda_gamma", choices=list(np.arange(0,0.5,0.1))))
config_space.add_hyperparameter(CS.CategoricalHyperparameter("lambda_delta", choices=list(np.arange(0.5,1.1,0.1))))
config_space.add_hyperparameter(CS.CategoricalHyperparameter("max_decrease_res", choices=list(np.arange(0.3,1.1,0.1))))

# config_space.add_hyperparameter(CS.UniformFloatHyperparameter("min_radius", lower=0.002, upper=0.004))

search_alg = TuneBOHB(
    space=config_space,
    metric=METRIC,
    mode="max",
    max_concurrent=8,
    points_to_evaluate=[
        {
            "max_decrease_res": 1,
            "lambda_speckle": 0.5,
            "lambda_delta": 1
        }
    ]
)

num_samples = 32

reporter = CLIReporter(
    metric=METRIC,
    mode="max",
    sort_by_metric=True,
    max_report_frequency=8
)
reporter.add_metric_column(METRIC)
tuner = tune.Tuner(
        tune.with_resources(
            training_function,
            {"cpu": 2, "gpu": 1/8}
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