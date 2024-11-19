# run.py
import os
import warnings
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CustomDataset
from data import load_data
from utils import seed_torch, setup_logger, LOGGER
from model import Net
from train import train_one_epoch, val_one_epoch, inference, get_cosine_schedule_with_warmup
from traffic_light import TrafficLightMaskGenerator

def get_train_transforms(image_size):
    aug_compose = []
    if isinstance(image_size, int):
        aug_compose.append(A.Resize(image_size, image_size))
    elif isinstance(image_size, list):
        aug_compose.append(A.Resize(*image_size))
    aug_compose.append(
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        )
    )
    aug_compose.append(A.Normalize())
    aug_compose.append(ToTensorV2())
    return A.Compose(aug_compose)

def get_val_transforms(image_size):
    aug_compose = []
    if isinstance(image_size, int):
        aug_compose.append(A.Resize(image_size, image_size))
    elif isinstance(image_size, list):
        aug_compose.append(A.Resize(*image_size))
    aug_compose.append(A.Normalize())
    aug_compose.append(ToTensorV2())
    return A.Compose(aug_compose)

def main():
    # 学習関連のConfig
    project_name = "atmacup_18"
    exp_name = "baseline52"
    device_id = 0

    output_root = "../output"

    model_name = "resnet18"
    img_size = 192
    batch_size = 128
    n_workers = 10
    n_epochs = 6
    start_epoch = 1
    warmup_steps = 300
    accumulation_steps = 1
    early_stopping_steps = 5
    use_amp = True

    initial_lr = 1e-2
    final_lr = 1e-3
    fold_name = 'fold'
    is_depth = True
    in_chans = 9 + 8 + 3  # 画像x3 + 信号機 + Depth map

    # seed
    seed_torch()

    # Outputやloggingに関する設定
    output_path = os.path.join(output_root, exp_name)
    model_path = output_path + "/model"
    plot_path = output_path + "/plot"
    sample_img_path = output_path + "/sample_img"
    oof_path = output_path + "/oof"

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path + "/log", exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(sample_img_path, exist_ok=True)
    os.makedirs(oof_path, exist_ok=True)

    # logger
    log_path = os.path.join(output_path, "log/log.txt")
    setup_logger(out_file=log_path)

    # デバイスの設定
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # データの読み込み
    df_feature_train, df_feature_test, df_sample_submission, FEATIRE_COLMUNS = load_data()

    cv_best_score_list = []
    test_preds = np.zeros((5, df_sample_submission.shape[0], df_sample_submission.shape[1]))

    for fold in range(5):

        # wandb
        wandb.init(tags=[exp_name, f"fold{fold}", model_name],
                project=project_name, entity='rikuter67-wakayama-university',
                name=f"{exp_name}_{model_name}_fold{fold}", reinit=True)

        # Datasetの初期化
        df_train = df_feature_train[df_feature_train[fold_name] != fold].reset_index(drop=True)
        df_val = df_feature_train[df_feature_train[fold_name] == fold].reset_index(drop=True)
        val_index = df_feature_train[df_feature_train[fold_name] == fold].index

        # transform
        train_transform = get_train_transforms(img_size)
        val_transform = get_val_transforms(img_size)

        train_dataset = CustomDataset(
            df=df_train,
            feature_columns=FEATIRE_COLMUNS,
            transform=train_transform,
            mode="train",
            target_columns=df_sample_submission.columns,
            traffic_light_gen=TrafficLightMaskGenerator(image_size=img_size),
            is_depth=is_depth,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=n_workers,
            drop_last=True
        )
        val_dataset = CustomDataset(
            df=df_val,
            feature_columns=FEATIRE_COLMUNS,
            transform=val_transform,
            mode="val",
            target_columns=df_sample_submission.columns,
            traffic_light_gen=TrafficLightMaskGenerator(image_size=img_size),
            is_depth=is_depth,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=n_workers,
            drop_last=False
        )

        model = Net(
            feature_columns=FEATIRE_COLMUNS,
            target_columns=df_sample_submission.columns,
            name=model_name,
            in_chans=in_chans
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=float(initial_lr), eps=1e-7)
        num_training_steps = (
            len(train_dataloader) * n_epochs
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

        criterion = nn.L1Loss(reduction='none')
        scaler = GradScaler(enabled=use_amp)

        # wandb misc
        wandb.watch(model)
        LOGGER.info('# ===============================================================================')
        LOGGER.info(f'# Start fold: {fold}')
        LOGGER.info('# ===============================================================================')

        best = 1e10
        best_epoch = 0
        early_stopping_cnt = 0

        for e in range(start_epoch , start_epoch + n_epochs):
            if e > 0:
                train_one_epoch(train_dataloader, model, device, criterion, optimizer, scheduler, scaler, accumulation_steps, use_amp, wandb, fold, e)

            score = val_one_epoch(val_dataloader, model, device, criterion, scaler, use_amp, wandb, fold, e)

            LOGGER.info('Saving last model ...')
            model_save_path = os.path.join(model_path, f"fold{fold}_weight_checkpoint_last.pth")

            torch.save({
                "state_dict": model.state_dict(),
            }, model_save_path)

            if best > score:
                LOGGER.info(f'Best score update: {best:.5f} --> {score:.5f}')
                best = score
                best_epoch = e

                LOGGER.info('Saving best model ...')
                model_save_path = os.path.join(model_path, f"fold{fold}_weight_checkpoint_best.pth")

                torch.save({
                    "state_dict": model.state_dict(),
                }, model_save_path)

                early_stopping_cnt = 0
            else:
                # early stopping
                early_stopping_cnt += 1
                if early_stopping_cnt >= early_stopping_steps:
                    LOGGER.info(f"Early stopping at Epoch {e}")
                    break

            LOGGER.info('-' * 20)
            LOGGER.info(f'Best val score: {best}, at epoch {best_epoch} fold{fold}')
            LOGGER.info('-' * 20)
            wandb.log({
                "epoch": e,
                "Best MAE": best,
            })
        wandb.finish()
        cv_best_score_list.append(best)

        # Val/Test inference

        # Best modelのロード
        best_checkpoint = os.path.join(model_path, f"fold{fold}_weight_checkpoint_best.pth")
        model.load_state_dict(torch.load(best_checkpoint)["state_dict"])

        # テストデータのDataset/Loader用意
        test_dataset = CustomDataset(
            df=df_feature_test,
            feature_columns=FEATIRE_COLMUNS,
            transform=val_transform,
            mode="test",
            traffic_light_gen=TrafficLightMaskGenerator(image_size=img_size),
            is_depth=is_depth,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=n_workers,
            drop_last=False
        )

        # テストデータの予測
        pred_test = inference(test_dataloader, model, device)
        test_preds[fold] = pred_test

        # Valデータの予測を保存しておく
        pred_val = inference(val_dataloader, model, device)
        np.save(os.path.join(oof_path, f"fold{fold}_val_pred.npy"), pred_val)

    # 各FoldのTestの推論結果を保存しておく
    np.save(os.path.join(output_path, f"test_preds.npy"), test_preds)

if __name__ == '__main__':
    main()
