from argparse import Namespace
import torch
import datetime
from shutil import copyfile

# from torch.utils.data import Dataset
from monai.data import decollate_batch
from models.model import define_model, initialize_model_and_optimizer
import time
from tqdm import tqdm

from data.image_dataset import get_dataset, get_post_transformation
from utils.metrics import MetricsManager, Task
from utils.losses import get_loss_function_by_name
from utils.visualizer import Visualizer, plot_sample
import copy

def vessel_segmentation_train(args: Namespace, config: dict[str,dict]):
    if args.split != "" and "split" in config["Train"]["data"]["image"]:
        config["Train"]["data"]["image"]["split"] = config["Train"]["data"]["image"]["split"] + args.split + ".txt"
        config["Train"]["data"]["label"]["split"] = config["Train"]["data"]["label"]["split"] + args.split + ".txt"
        config["Validation"]["data"]["image"]["split"] = config["Validation"]["data"]["image"]["split"] + args.split + ".txt"
        config["Validation"]["data"]["label"]["split"] = config["Validation"]["data"]["label"]["split"] + args.split + ".txt"
        config["Test"]["data"]["image"]["split"] = config["Test"]["data"]["image"]["split"] + args.split + ".txt"

    max_epochs = config["Train"]["epochs"]
    val_interval = config["Train"].get("val_interval") or 1
    VAL_AMP = bool(config["General"].get("amp"))
    save_interval = config["Train"].get("save_interval") or 100
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler(enabled=VAL_AMP)
    device = torch.device(config["General"].get("device") or "cpu")
    task: Task = config["General"]["task"]


    train_loader = get_dataset(config, 'train')
    val_loader = get_dataset(config, 'validation')
    # test_loader = get_dataset(config, 'test')
    # test_loader_iter = iter(test_loader)
    post_pred, post_label = get_post_transformation(config, "train")
    post_pred_val, post_label_val = get_post_transformation(config, "validation")

    model = define_model(copy.deepcopy(config), "train")

    with torch.no_grad():
        inputs = next(iter(train_loader))["image"].to(device=device, dtype=torch.float32)
    visualizer = Visualizer(config, args.start_epoch>0)
    visualizer.save_model_architecture(model, inputs)

    optimizer = initialize_model_and_optimizer(model, config, args)

    loss_name = config["Train"]["loss"]
    loss_function = get_loss_function_by_name(loss_name, config)
    def schedule(step: int):
        if step < max_epochs - config["Train"]["epochs_decay"]:
            return 1
        else:
            return (max_epochs-step) * (1/max(1,config["Train"]["epochs_decay"]))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    metrics = MetricsManager(task)
    if config["Train"].get("AT"):
        at = get_loss_function_by_name("AtLoss", config, scaler, loss_function)


    # TRAINING BEGINS HERE
    if args.start_epoch>0:
        best_metric, best_metric_epoch = visualizer.get_max_of_metric("metric", metrics.get_comp_metric('val'))
    else:
        best_metric = -1
        best_metric_epoch = -1

    total_start = time.time()
    epoch_tqdm = tqdm(range(args.start_epoch,max_epochs), desc="epoch")
    for epoch in epoch_tqdm:
        epoch_metrics = dict()
        model.train()
        epoch_loss = 0
        step = 0
        mini_batch_tqdm = tqdm(train_loader, leave=False)
        for batch_data in mini_batch_tqdm:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            if config["Train"].get("AT"):
                inputs_at, reg = at(model, inputs, batch_data["background"].to(device), labels)
                inputs = torch.cat((inputs_at, inputs), dim=0)
                labels = torch.tile(labels, (2,1,1,1))
            else:
                reg=0
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                # if config["Data"]["num_classes"]==1:
                outputs=outputs.squeeze(-1)
                # labels=labels.to(dtype=outputs.dtype)

                loss = loss_function(outputs, labels)
                labels = [post_label(i) for i in decollate_batch(labels)]
                outputs = [post_pred(i) for i in decollate_batch(outputs)]
                metrics(y_pred=outputs, y=labels)
            scaler.scale(loss+reg).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            mini_batch_tqdm.set_description(f'train_{loss_name}: {loss.item():.4f}')
        lr_scheduler.step()

        epoch_loss /= step
        epoch_metrics["loss"] = {
            f"train_{loss_name}": epoch_loss
        }
        epoch_metrics["metric"] = metrics.aggregate_and_reset(prefix="train")
        epoch_tqdm.set_description(f'avg train loss: {epoch_loss:.4f}')
        if epoch%save_interval == 0:
            if task == Task.VESSEL_SEGMENTATION:
                image_path_train = visualizer.plot_sample(inputs[0], outputs[0], labels[0], suffix='latest_train', path=batch_data["image_path"][0])
            else:
                image_path_train = visualizer.plot_clf_sample(inputs, outputs, labels, batch_data["path"], suffix='latest_train')
        
        checkpoint_path = visualizer.save_model(model, optimizer, epoch, 'latest')

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                step = 0
                for val_data in tqdm(val_loader, desc='Validation', leave=False):
                    step += 1
                    val_inputs, val_labels = (
                        val_data["image"].to(device).float(),
                        val_data["label"].to(device),
                    )
                    val_outputs: torch.Tensor = model(val_inputs)
                    # if config["Data"]["num_classes"]==1:
                    val_outputs=val_outputs.squeeze(-1)
                        # val_labels=val_labels.to(dtype=val_outputs.dtype)
                    val_loss += loss_function(val_outputs, val_labels).item()
                    val_labels = [post_label_val(i) for i in decollate_batch(val_labels)]
                    val_outputs = [post_pred_val(i) for i in decollate_batch(val_outputs)]
                    metrics(y_pred=val_outputs, y=val_labels)

                epoch_metrics["loss"][f"val_{loss_name}"] = val_loss/step
                epoch_metrics["metric"].update(metrics.aggregate_and_reset(prefix="val"))

                metric_comp =  epoch_metrics["metric"][metrics.get_comp_metric('val')]

                if metric_comp > best_metric:
                    best_metric = metric_comp
                    best_metric_epoch = epoch
                    visualizer.save_model(model, optimizer, epoch, 'best')

                visualizer.plot_losses_and_metrics(epoch_metrics, epoch)
                if epoch%save_interval == 0:
                    if task == Task.VESSEL_SEGMENTATION:
                        image_path_val = visualizer.plot_sample(val_inputs[0], val_outputs[0], val_labels[0], suffix='latest_val')
                    else:
                        image_path_val = visualizer.plot_clf_sample(val_inputs, val_outputs, val_labels, val_data["path"], suffix= None if best_metric>metric_comp else 'best')


                if (epoch + 1) % save_interval == 0:
                    copyfile(checkpoint_path, checkpoint_path.replace("latest", str(epoch+1)))
                    copyfile(image_path_train, image_path_train.replace("latest_train", str(epoch+1)+"_train"))
                    copyfile(image_path_val, image_path_val.replace("latest_val", str(epoch+1)+"_val"))
        visualizer.log_model_params(model, epoch)

    total_time = time.time() - total_start

    print(f'Finished training after {str(datetime.timedelta(seconds=total_time))}. Best metric: {best_metric} at epoch: {best_metric_epoch}')