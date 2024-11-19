# config.py
import os

# データセットのパス
PATH_TO_DATASET = ""

def get_relative_path(path):
    return os.path.join(PATH_TO_DATASET, path)

# 画像へのパス
image_path_root_list = [
    get_relative_path("atmaCup18_dataset/images/{ID}/image_t.png"),
    get_relative_path("atmaCup18_dataset/images/{ID}/image_t-0.5.png"),
    get_relative_path("atmaCup18_dataset/images/{ID}/image_t-1.0.png")
]

# Depthへのパス
depth_path_root_list = [
    get_relative_path("atmaCup18_dataset/depth/{ID}/image_t.png"),
    get_relative_path("atmaCup18_dataset/depth/{ID}/image_t-0.5.png"),
    get_relative_path("atmaCup18_dataset/depth/{ID}/image_t-1.0.png")
]

# 特徴量のパス
train_feature_path = get_relative_path("atmaCup18_dataset/train_features.csv")
test_feature_path = get_relative_path("atmaCup18_dataset/test_features.csv")
traffic_light_path = get_relative_path("atmaCup18_dataset/traffic_lights/{ID}.json")

# サンプルサブミッション
sample_submission_path = get_relative_path("atmaCup18__sample_submit.csv")

# ターゲットカラムとFold数
TARGET_COLUMNS = ['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'x_2', 'y_2', 'z_2', 'x_3', 'y_3', 'z_3', 'x_4', 'y_4', 'z_4',
                  'x_5', 'y_5', 'z_5', 'x_6', 'y_6', 'z_6', 'x_7', 'y_7', 'z_7', 'x_8', 'y_8', 'z_8', 'x_9', 'y_9', 'z_9',
                  'x_10', 'y_10', 'z_10', 'x_11', 'y_11', 'z_11', 'x_12', 'y_12', 'z_12', 'x_13', 'y_13', 'z_13', 'x_14',
                  'y_14', 'z_14', 'x_15', 'y_15', 'z_15', 'x_16', 'y_16', 'z_16', 'x_17', 'y_17', 'z_17', 'x_18', 'y_18',
                  'z_18', 'x_19', 'y_19', 'z_19', 'x_20', 'y_20', 'z_20', 'x_21', 'y_21', 'z_21', 'x_22', 'y_22', 'z_22',
                  'x_23', 'y_23', 'z_23', 'x_24', 'y_24', 'z_24', 'x_25', 'y_25', 'z_25', 'x_26', 'y_26', 'z_26', 'x_27',
                  'y_27', 'z_27', 'x_28', 'y_28', 'z_28', 'x_29', 'y_29', 'z_29']

N_FOLD = 5
