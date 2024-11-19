# train.py
import os
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from utils import LOGGER, mae
from torch import nn

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))

    return LambdaLR(optimizer, lr_lambda)

def train_one_epoch(train_dataloader, model, device, criterion, optimizer, scheduler, scaler, accumulation_steps, use_amp, wandb, fold, e):

    train_time = time.time()
    LOGGER.info("")
    LOGGER.info("+" * 30)
    LOGGER.info(f"+++++  Epoch {e} at fold {fold}")
    LOGGER.info("+" * 30)
    LOGGER.info("")
    progress_bar = tqdm(train_dataloader, dynamic_ncols=True)
    iters = len(train_dataloader)

    model.train()
    torch.set_grad_enabled(True)

    loss_list = []
    for step_train, data in enumerate(progress_bar):
        inputs = data["image"].to(device)
        target = data["target"].to(device)
        features = data["features"].to(device)

        bs = inputs.shape[0]

        with autocast(device_type="cuda"):
            output = model(inputs, features)
            loss = criterion(output, target).mean()

        if accumulation_steps > 1:
            loss_bw = loss / accumulation_steps
            scaler.scale(loss_bw).backward()
            if (step_train + 1) % accumulation_steps == 0 or step_train == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step(e + step_train / iters)
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        loss_list.append(loss.item())
        text_progress_bar = f"loss: {loss.item()} loss(avg): {np.array(loss_list).mean()}"
        progress_bar.set_description(text_progress_bar)

        wandb.log({
            f"steps": step_train + (e-1)*iters,
            f"Loss/train_fold{fold} (steps)": loss.item(),
            "Learning Rate": optimizer.param_groups[0]["lr"],
        })


    LOGGER.info(f"Train loss: {np.array(loss_list).mean()}")
    LOGGER.info(f"Train time: {(time.time() - train_time) / 60:.3f} min")

    wandb.log({
        f"epoch": e,
        f"Loss/train_fold{fold}": np.array(loss_list).mean(),
    })


def val_one_epoch(val_dataloader, model, device, criterion, scaler, use_amp, wandb, fold, e):

    val_time = time.time()
    progress_bar = tqdm(val_dataloader, dynamic_ncols=True)

    model.eval()
    torch.set_grad_enabled(False)

    loss_list = []
    pred_list = []
    target_list = []

    for step_val, data in enumerate(progress_bar):
        inputs = data["image"].to(device)
        target = data["target"].to(device)
        features = data["features"].to(device)

        with autocast(device_type="cuda"):
            output = model(inputs, features)
            loss = criterion(output, target).mean()

        loss_list.append(loss.item())
        pred_list.extend(output.detach().cpu().numpy().tolist())
        target_list.extend(target.cpu().numpy().tolist())

        text_progress_bar = f"loss: {loss.item()} loss(avg): {np.array(loss_list).mean()}"
        progress_bar.set_description(text_progress_bar)


    score = mae(np.array(target_list), np.array(pred_list))
    LOGGER.info(f"Val loss: {np.array(loss_list).mean()}")
    LOGGER.info(f"Val MAE score: {score}")
    LOGGER.info(f"Val time: {(time.time() - val_time) / 60:.3f} min")

    log_dict = {
        f"epoch": e,
        f"Loss/val_fold{fold}": np.array(loss_list).mean(),
        f"MAE/val_fold{fold}": score,
    }
    wandb.log(log_dict)

    return score

def inference(test_dataloader, model, device):
    progress_bar = tqdm(test_dataloader, dynamic_ncols=True)

    model.eval()
    torch.set_grad_enabled(False)

    pred_list = []
    for step_test, data in enumerate(progress_bar):
        inputs = data["image"].to(device)
        features = data["features"].to(device)

        with autocast(device_type="cuda"):
            output = model(inputs, features)

        pred_list.extend(output.detach().cpu().numpy().tolist())
    return np.array(pred_list)

