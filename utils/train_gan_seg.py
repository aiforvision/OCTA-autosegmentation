from argparse import Namespace
import torch
import datetime

# from torch.utils.data import Dataset
from monai.data import decollate_batch
from models.gan_seg_model import GanSegModel
from models.model import define_model, initialize_model_and_optimizer
import time
from tqdm import tqdm

from data.image_dataset import get_dataset, get_post_transformation
from utils.metrics import MetricsManager, Task
from utils.losses import get_loss_function_by_name
from utils.visualizer import Visualizer

def gan_vessel_segmentation_train(args: Namespace, config: dict[str,dict]):
    if "split" in config["Train"]["data"]["real_B"] and args.split!="":
        config["Train"]["data"]["real_B"]["split"] = config["Train"]["data"]["real_B"]["split"] + args.split + ".txt"

    max_epochs = config["Train"]["epochs"]
    save_interval = config["Train"].get("save_interval") or 1
    VAL_AMP = bool(config["General"].get("amp"))
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler(enabled=VAL_AMP)
    device = torch.device(config["General"].get("device") or "cpu")
    task: Task = config["General"]["task"]
    visualizer = Visualizer(config, args.start_epoch>0, epoch=args.epoch)


    train_loader = get_dataset(config, "train")
    post_pred, post_label = get_post_transformation(config, "train")

    model: GanSegModel = define_model(config, phase = "train")

    with torch.no_grad():
        batch = next(iter(train_loader))
        inputs = (batch["real_A"].to(device=device, dtype=torch.float32), batch["real_B"].to(device=device, dtype=torch.float32))
        model.forward(inputs,complete=True)
        if task == Task.GAN_VESSEL_SEGMENTATION:
            visualizer.save_model_architecture(model, inputs[1])
        else:
            visualizer.save_model_architecture(model, inputs)

    (optimizer_G, optimizer_D, optimizer_S ) = initialize_model_and_optimizer(model, config, args, phase="train")

    loss_name_dg = config["Train"]["loss_dg"]
    loss_name_s = config["Train"]["loss_s"]
    dg_loss = get_loss_function_by_name(loss_name_dg, config)
    s_loss = get_loss_function_by_name(loss_name_s, config)
    criterionIdt = torch.nn.L1Loss().to(device)
    def schedule(step: int):
        if step < (max_epochs - config["Train"]["epochs_decay"]):
            return 1
        else:
            return (max_epochs-step) * (1/max(1,config["Train"]["epochs_decay"]))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, schedule)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, schedule)
    lr_scheduler_S = torch.optim.lr_scheduler.LambdaLR(optimizer_S, schedule)
    metrics = MetricsManager(task)

    total_start = time.time()
    epoch_tqdm = tqdm(range(args.start_epoch,max_epochs), desc="epoch")
    for epoch in epoch_tqdm:
        epoch_metrics = dict()
        epoch_metrics["loss"] = {
            "D_fake": 0,
            "D_real": 0,
            "G": 0,
            "G_idt": 0,
            "S": 0,
            "S_idt": 0
        }
        model.train()
        epoch_loss = 0
        step = 0
        mini_batch_tqdm = tqdm(train_loader, leave=False)
        for batch_data in mini_batch_tqdm:
            step += 1
            real_A: torch.Tensor = batch_data["real_A"].to(device)
            real_B: torch.Tensor = batch_data["real_B"].to(device)
            if task == Task.GAN_VESSEL_SEGMENTATION:
                real_A_seg: torch.Tensor = batch_data["real_A_seg"].to(device)
            optimizer_D.zero_grad()
            tune_D = step%1==0
            with torch.cuda.amp.autocast():
                fake_B, idt_B, pred_fake_B, pred_real_B = model.forward_GD((real_A, real_B), tune_D=tune_D)
                loss_D_fake = dg_loss(pred_fake_B, False)
                loss_D_real = dg_loss(pred_real_B, True)
                loss_D = 0.5*(loss_D_fake + loss_D_real)
            if tune_D:
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)

            
            optimizer_G.zero_grad()
            optimizer_S.zero_grad()
            with torch.cuda.amp.autocast():
                if task==Task.GAN_VESSEL_SEGMENTATION:
                    pred_fake_B, fake_B_seg, real_B_seg, idt_B_seg  = model.forward_GS(real_B, fake_B, idt_B)
                    real_B_seg[real_B_seg<=0.5]=0
                    real_B_seg[real_B_seg>0.5]=1
                else:
                    pred_fake_B, fake_B_seg, real_B_seg, idt_B_seg, real_A_seg  = model.forward_GS(real_B, fake_B, idt_B, real_A)
                loss_G = dg_loss(pred_fake_B, True)
                if model.compute_identity:
                    loss_G_idt = criterionIdt(idt_B, real_B)
                else:
                    loss_G_idt = torch.tensor(0)

                loss_G += loss_G_idt

                loss_S = s_loss(fake_B_seg, real_A_seg)
                if model.compute_identity_seg:
                    loss_S_idt = s_loss(idt_B_seg, real_B_seg)
                    loss_SS = 0.5*(loss_S + loss_S_idt)
                else:
                    loss_S_idt = torch.tensor(0)
                    loss_SS = loss_S

                loss_GS = loss_G + loss_SS

            scaler.scale(loss_GS).backward()
            scaler.step(optimizer_G)
            scaler.step(optimizer_S)
            scaler.update()


            real_A = [post_label(i) for i in decollate_batch(real_A[0:1, 0:1])]
            fake_B_seg = [post_pred(i) for i in decollate_batch(fake_B_seg[0:1])]
            if real_B_seg is not None:
                real_B_seg = [post_pred(i) for i in decollate_batch(real_B_seg[0:1])]

            metrics(y_pred=fake_B_seg, y=real_A_seg)

            epoch_metrics["loss"]["D_fake"] += loss_D_fake.item()
            epoch_metrics["loss"]["D_real"] += loss_D_real.item()
            epoch_metrics["loss"]["G"] += loss_G.item()
            epoch_metrics["loss"]["G_idt"] += loss_G_idt.item()
            epoch_metrics["loss"]["S"] += loss_S.item()
            epoch_metrics["loss"]["S_idt"] += loss_S_idt.item()

            epoch_loss += loss_SS.item()
            mini_batch_tqdm.set_description(f"train_{loss_name_s}: {loss_SS.item():.4f}")
        lr_scheduler_G.step()
        lr_scheduler_D.step()
        lr_scheduler_S.step()

        epoch_metrics["loss"] = {k: v/step for k,v in epoch_metrics["loss"].items()}
        epoch_metrics["metric"] = metrics.aggregate_and_reset(prefix="train")

        epoch_tqdm.set_description(f"avg train loss: {epoch_loss:.4f}")

        visualizer.save_model(model.generator, optimizer_G, epoch+1, "latest_G")
        visualizer.save_model(model.discriminator, optimizer_D, epoch+1, "latest_D")
        visualizer.save_model(model.segmentor, optimizer_S, epoch+1, "latest_S")
        if (epoch + 1) % save_interval == 0:
            visualizer.save_model(model.generator, optimizer_G, epoch+1, f"{epoch+1}_G")
            visualizer.save_model(model.discriminator, optimizer_D, epoch+1, f"{epoch+1}_D")
            visualizer.save_model(model.segmentor, optimizer_S, epoch+1, f"{epoch+1}_S")

        visualizer.plot_losses_and_metrics(epoch_metrics, epoch)
        if task == Task.GAN_VESSEL_SEGMENTATION:
            visualizer.plot_gan_seg_sample(
                real_A[0],
                fake_B[0],
                fake_B_seg[0],
                real_B[0],
                idt_B[0],
                real_B_seg[0],
                (epoch + 1),
                path_A=batch_data["real_A_path"][0],
                path_B=batch_data["real_B_path"][0],
                save_epoch = (epoch + 1) % save_interval == 0
            )
        else:
            visualizer.plot_cut_sample(
                real_A[0],
                fake_B[0],
                real_B[0],
                idt_B[0],
                (epoch + 1),
                path_A=batch_data["real_A_path"][0],
                path_B=batch_data["real_B_path"][0],
                save_epoch = (epoch + 1) % save_interval == 0
            )

    total_time = time.time() - total_start

    print(f"Finished training after {str(datetime.timedelta(seconds=total_time))}.")