# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from config import TARGET_COLUMNS, traffic_light_path
from traffic_light import TrafficLightMaskGenerator

class CustomDataset(Dataset):
    def __init__(self,
                 df,
                 feature_columns,
                 transform=None,
                 mode="train",
                 target_columns=None,
                 traffic_light_gen=None,
                 is_depth=False,
                 ):
        self.df = df
        self.transform = transform
        self.feature_columns = feature_columns
        self.mode = mode
        self.target_columns = target_columns
        self.traffic_light_gen = traffic_light_gen
        self.is_depth = is_depth

    def __len__(self):
        return self.df.shape[0]

    def extract_features(self, row):
        features = row[self.feature_columns].values * 1
        return features.astype(float)

    def __getitem__(self, index):
        ret = {}

        # 訓練・推論に必要な情報を取得する
        row = self.df.iloc[index]

        # =====================================
        # 画像情報の取得
        # =====================================
        # 画像をチャンネル方向に結合する
        images = [
            np.array(Image.open(row.img_path_t_00)),
            np.array(Image.open(row.img_path_t_05)),
            np.array(Image.open(row.img_path_t_10)),
        ]
        images = np.concatenate(images, axis=2)

        if self.transform is not None:
            aug = self.transform(image=images)
            images = aug['image']  # 既に Tensor 型で (C, H, W) の形状
        else:
            # ToTensorV2 を使わない場合は自分で変換
            images = images.astype(np.float32).transpose(2, 0, 1) / 255
            images = torch.tensor(images, dtype=torch.float)

        # 信号機のラベルごとに1チャンネル割り当てて、画像にConcatする
        if self.traffic_light_gen is not None:
            traffic_image = self.traffic_light_gen.generate_masks(traffic_light_path.format(ID=row.ID))
            # traffic_image の形状は (H, W, C) なので (C, H, W) に変換
            traffic_image = traffic_image.transpose(2, 0, 1)
            traffic_image = torch.tensor(traffic_image, dtype=torch.float)
            images = torch.cat([images, traffic_image], dim=0)

        # Depth情報の追加
        if self.is_depth == True:
            _, image_height, image_width = images.shape
            depth_maps = [
                np.array(Image.open(row.depth_path_t_00).resize((image_width, image_height), Image.BILINEAR)),
                np.array(Image.open(row.depth_path_t_05).resize((image_width, image_height), Image.BILINEAR)),
                np.array(Image.open(row.depth_path_t_10).resize((image_width, image_height), Image.BILINEAR)),
            ]
            depth_maps = np.stack(depth_maps, axis=0) / 255
            depth_maps = torch.tensor(depth_maps, dtype=torch.float)
            images = torch.cat([images, depth_maps], dim=0)

        ret["image"] = images  # 既に Tensor 型

        # =====================================
        # Table特徴量の取得
        # =====================================
        features = self.extract_features(row)
        ret["features"] = torch.tensor(features, dtype=torch.float)

        # =====================================
        # Targetの取得
        # =====================================
        if self.mode != "test":
            assert self.target_columns is not None, "target_columnsが設定されていません。"
            target = row[self.target_columns].values.astype(np.float16)
            ret["target"] = torch.tensor(target, dtype=torch.float)

        return ret
