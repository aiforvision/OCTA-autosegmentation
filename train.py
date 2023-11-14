import argparse
import torch
import datetime
import os
import yaml
from monai.utils import set_determinism
from random import randint

from models.model import define_model
from models.networks import init_weights
import time
from tqdm import tqdm
from shutil import copyfile
from copy import deepcopy

from data.image_dataset import get_dataset, get_post_transformation
from utils.metrics import MetricsManager
from utils.visualizer import Visualizer
from models.base_model_abc import BaseModelABC
from utils.enums import Phase

def train(args: argparse.Namespace, config: dict[str,dict]):
    for phase in Phase:
        if phase not in config:
            continue
        for k in config[phase]["data"].keys():
            if not config[phase]["data"][k].get("split", ".txt").endswith(".txt"):
                assert bool(args.split), "You have to specify a split!"
                config[phase]["data"][k]["split"] = config[phase]["data"][k]["split"] + args.split + ".txt"

    max_epochs = config[Phase.TRAIN]["epochs"]
    val_interval = config[Phase.TRAIN].get("val_interval") or 1
    save_interval = config[Phase.TRAIN].get("save_interval") or 100
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["General"].get("amp")))
    device = torch.device(config["General"].get("device") or "cpu")
    visualizer = Visualizer(config, args.start_epoch>0, epoch=args.epoch)


    train_loader = get_dataset(config, Phase.TRAIN, use_all_workers=args.use_all_workers)
    post_transformations_train = get_post_transformation(config, Phase.TRAIN)

    if Phase.VALIDATION in config:
        val_loader = get_dataset(config, Phase.VALIDATION, use_all_workers=args.use_all_workers)
        post_transformations_val = get_post_transformation(config, Phase.VALIDATION)
    else:
        val_loader = None
        print("No validation config. Skipping validation steps.")
    
    init_mini_batch = next(iter(train_loader))
    input_key = [k for k in init_mini_batch.keys() if not k.endswith("_path")][0]
    init_mini_batch["image"] = init_mini_batch[input_key]

    model: BaseModelABC = define_model(deepcopy(config), phase = Phase.TRAIN)
    model.initialize_model_and_optimizer(init_mini_batch, init_weights, config, args, scaler, phase=Phase.TRAIN)

    visualizer.save_model_architecture(model, init_mini_batch["image"].to(device, non_blocking=True) if init_mini_batch else None)

    metrics = MetricsManager(phase=Phase.TRAIN)

    if args.start_epoch>0:
        best_metric, best_metric_epoch = visualizer.get_max_of_metric("metric", metrics.get_comp_metric(Phase.VALIDATION))
    else:
        best_metric = -1
        best_metric_epoch = -1

    total_start = time.time()
    epoch_tqdm = tqdm(range(args.start_epoch, max_epochs), desc="epoch")
    for epoch in epoch_tqdm:
        epoch_metrics: dict[str, dict[str, float]] = dict()
        epoch_metrics["loss"] = dict()
        model.train()
        epoch_loss = 0
        step = 0
        save_best = False
        # TRAINING LOOP
        mini_batch_tqdm = tqdm(train_loader, leave=False)
        for mini_batch in mini_batch_tqdm:
            step += 1

            outputs, losses = model.perform_training_step(mini_batch, scaler, post_transformations_train, device)
            with torch.cuda.amp.autocast():
                model.compute_metric(outputs, metrics)
            for loss_name, loss in losses.items():
                if f"train_{loss_name}" in epoch_metrics["loss"]:
                    epoch_metrics["loss"][f"train_{loss_name}"] += loss
                else:
                    epoch_metrics["loss"][f"train_{loss_name}"] = loss
            main_loss = list(losses.keys())[0]
            epoch_loss += losses[main_loss]
            mini_batch_tqdm.set_description(f"train_{main_loss}: {losses[main_loss]:.4f}")
        
        for lr_scheduler in model.lr_schedulers:
            lr_scheduler.step()

        epoch_metrics["loss"] = {k: v/step for k,v in epoch_metrics["loss"].items()}
        epoch_metrics["metric"] = metrics.aggregate_and_reset(prefix=Phase.TRAIN)
        epoch_loss /= step

        epoch_tqdm.set_description(f"avg train loss: {epoch_loss:.4f}")

        if args.save_latest or save_best or (epoch + 1) % save_interval == 0:
            train_sample_path = model.plot_sample(
                visualizer,
                mini_batch,
                outputs,
                suffix="train_latest"
            )

        # VALIDATION
        if val_loader is not None and (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                step = 0
                for val_mini_batch in tqdm(val_loader, desc=Phase.VALIDATION.value, leave=False):
                    step += 1
                    with torch.cuda.amp.autocast():
                        outputs, losses = model.inference(val_mini_batch, post_transformations_val, device=device, phase=Phase.VALIDATION)
                        model.compute_metric(outputs, metrics)

                    for loss_name, loss in losses.items():
                        if f"val_{loss_name}" in epoch_metrics["loss"]:
                            epoch_metrics["loss"][f"val_{loss_name}"] += loss.item()
                        else:
                            epoch_metrics["loss"][f"val_{loss_name}"] = loss.item()
                    main_loss = list(losses.keys())[0]
                    val_loss += losses[main_loss].item()
                    mini_batch_tqdm.set_description(f"train_{main_loss}: {losses[main_loss].item():.4f}")
                    if step >= 40:
                        break
                

                epoch_metrics["loss"] = {k: v/step if k.startswith(Phase.VALIDATION.value) else v for k,v in epoch_metrics["loss"].items()}
                epoch_metrics["metric"].update(metrics.aggregate_and_reset(prefix=Phase.VALIDATION))
                val_loss /= step
                epoch_tqdm.set_description(f"avg train loss: {val_loss:.4f}")
                metric_comp =  epoch_metrics["metric"][metrics.get_comp_metric(Phase.VALIDATION)]
                if metric_comp > best_metric:
                    best_metric = metric_comp
                    best_metric_epoch = epoch
                save_best = True

                if args.save_latest or save_best or (epoch + 1) % save_interval == 0:
                    val_sample_path = model.plot_sample(
                        visualizer,
                        val_mini_batch,
                        outputs,
                        suffix="val_latest"
                    )
        
        if (epoch + 1) % save_interval == 0:
            copyfile(train_sample_path, train_sample_path.replace("latest", str(epoch+1)))
            if (epoch + 1) % val_interval == 0:
                copyfile(val_sample_path, val_sample_path.replace("latest", str(epoch+1)))
        if save_best:
            copyfile(train_sample_path, train_sample_path.replace("latest", "best"))
            copyfile(val_sample_path, val_sample_path.replace("latest", "best"))

        
        # Checkpoint saving
        if args.save_latest or save_best or (epoch + 1) % save_interval == 0:
            for optimizer_name in model.optimizer_mapping.keys():
                checkpoint_path = visualizer.save_model(None, getattr(model,optimizer_name), epoch+1, f"latest_{optimizer_name}")
                if (epoch + 1) % save_interval == 0:
                    copyfile(checkpoint_path, checkpoint_path.replace("latest", str(epoch+1)))
                if save_best:
                    copyfile(checkpoint_path, checkpoint_path.replace("latest", "best"))

            for model_names in model.optimizer_mapping.values():
                for model_name in model_names:
                    visualizer.save_model(getattr(model,model_name), None, epoch+1, f"latest_{model_name}")
                    if (epoch + 1) % save_interval == 0:
                        copyfile(checkpoint_path, checkpoint_path.replace("latest", str(epoch+1)))
                    if save_best:
                        copyfile(checkpoint_path, checkpoint_path.replace("latest", "best"))

        visualizer.plot_losses_and_metrics(epoch_metrics, epoch)
        visualizer.log_model_params(model, epoch)

    total_time = time.time() - total_start

    print(f"Finished training after {str(datetime.timedelta(seconds=total_time))}.")
    if best_metric_epoch > -1:
        print(f'Best metric: {best_metric} at epoch: {best_metric_epoch}.')

if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoch', type=str, default='latest')
    parser.add_argument('--split', type=str, default='')
    parser.add_argument('--save_latest', type=bool, default=True, help="If true, save a checkpoint and visuals after each epoch under the tag 'latest'.")
    parser.add_argument('--use_all_workers', type=bool, default=False, help="If true, use all cpu cores for dataloading. If false, only use half.")
    args = parser.parse_args()

    # Read config file
    path: str = os.path.abspath(args.config_file)
    assert os.path.isfile(path), f"Your provided config path {args.config_file} does not exist!"
    with open(path, "r") as stream:
        config: dict[str,dict] = yaml.safe_load(stream)

    if "seed" not in config["General"]:
        config["General"]["seed"] = randint(0,1e6)
    set_determinism(seed=config["General"]["seed"])

    train(args, config)
