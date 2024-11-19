# traffic_light.py

import json
import numpy as np
from PIL import Image, ImageDraw
import os

class TrafficLightMaskGenerator:
    def __init__(self, image_size=None, image_root='/mnt/c/Users/rikuter/kaggle/atmaCup18/atmaCup18_dataset/images'):
        self.image_size = image_size
        self.image_root = image_root  # 画像ファイルのルートパス

    def generate_masks(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        # デバッグ用にdataの内容を表示
        print(f"Loaded data from {json_path}:")
        print(json.dumps(data, indent=2))

        # 画像のパスを構築
        image_id = os.path.basename(json_path).split('.')[0]
        image_path = os.path.join(self.image_root, f"{image_id}.png")  # 拡張子は適宜変更
        print(f"Constructed image path: {image_path}")  # デバッグ用

        # 画像の存在を確認
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None

        # 画像のサイズを取得
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # マスクの初期化
        masks = np.zeros((8, image_height, image_width), dtype=np.uint8)

        # クラス名とクラスIDのマッピング
        class_name_to_id = {
            'red': 0,
            'yellow': 1,
            'green': 2,
            'off': 3,
            'left': 4,
            'right': 5,
            'straight': 6,
            'empty': 7
        }

        # アノテーションがリストの場合に対応
        if isinstance(data, list):
            annotations = data
        else:
            annotations = [data]

        for annotation in annotations:
            class_name = annotation.get('class')
            bbox = annotation.get('bbox')
            if class_name is None or bbox is None:
                continue  # データが不足している場合はスキップ

            class_id = class_name_to_id.get(class_name)
            if class_id is None:
                print(f"Unknown class '{class_name}' in {json_path}")
                continue

            # マスクを作成
            mask = Image.new('L', (image_width, image_height), 0)
            draw = ImageDraw.Draw(mask)

            # bboxの座標をピクセル値に変換
            # bboxがピクセル単位の場合、そのまま使用
            x_min, y_min, x_max, y_max = bbox

            # 座標が正規化されている場合は、以下のように変換
            # x_min = int(bbox[0] * image_width)
            # y_min = int(bbox[1] * image_height)
            # x_max = int(bbox[2] * image_width)
            # y_max = int(bbox[3] * image_height)

            draw.rectangle([x_min, y_min, x_max, y_max], outline=1, fill=1)
            mask = np.array(mask)

            masks[class_id] = np.maximum(masks[class_id], mask)

        masks = masks.transpose(1, 2, 0)  # (H, W, C)

        if self.image_size is not None:
            masks = np.array(Image.fromarray(masks).resize((self.image_size, self.image_size)))

        return masks
