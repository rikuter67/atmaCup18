# data.py
import pandas as pd
import numpy as np
from config import train_feature_path, test_feature_path, sample_submission_path
from config import image_path_root_list, depth_path_root_list, TARGET_COLUMNS
from utils import create_features

def load_data():
    # データの読み込み
    df_feature_train = pd.read_csv(train_feature_path)
    df_feature_test = pd.read_csv(test_feature_path)
    df_sample_submission = pd.read_csv(sample_submission_path)

    # シーンの名前を追加
    df_feature_train["scene"] = [i.split("_")[0] for i in df_feature_train.ID]
    df_feature_test["scene"] = [i.split("_")[0] for i in df_feature_test.ID]

    df_feature_train["is_train"] = True
    df_feature_test["is_train"] = False

    df_feature = pd.concat([
        df_feature_train,
        df_feature_test
    ], axis=0, ignore_index=True)

    # 特徴量エンジニアリング
    df_feature.vEgo = df_feature.vEgo / 30
    df_feature.aEgo = df_feature.aEgo
    df_feature.steeringAngleDeg = df_feature.steeringAngleDeg / 400
    df_feature.steeringTorque = df_feature.steeringTorque / 600

    gear_cols = []
    for gear_shift in df_feature.gearShifter.value_counts().index:
        col_name = f"label_{gear_shift}"
        gear_cols.append(col_name)
        df_feature[col_name] = 0
        df_feature.loc[df_feature.gearShifter == gear_shift, col_name] = 1

    all_features = create_features(df_feature)
    df_feature = df_feature.merge(all_features.fillna(0), left_on="scene", right_index=True)

    df_feature_train = df_feature[df_feature.is_train == True].reset_index(drop=True)

    # 存在するカラムのみをドロップ
    columns_to_drop = [col for col in TARGET_COLUMNS if col in df_feature.columns]
    df_feature_test = df_feature[df_feature.is_train == False].drop(columns_to_drop, axis=1).reset_index(drop=True)

    FEATIRE_COLMUNS = ['vEgo', 'aEgo', 'steeringAngleDeg', 'steeringTorque', 'brake', 'brakePressed', 'gas', 'gasPressed', 'leftBlinker', 'rightBlinker'] + gear_cols + list(all_features.columns)

    # Foldの作成
    N_FOLD = 5
    scene_list = df_feature_train.scene.unique()
    fold_map = {}
    for i, scene in enumerate(scene_list):
        fold_map[scene] = i % N_FOLD
    df_feature_train["fold"] = df_feature_train.scene.map(fold_map)

    # 画像とDepthのパスを追加
    df_feature_train["img_path_t_00"] = [image_path_root_list[0].format(ID=ID) for ID in df_feature_train.ID]
    df_feature_train["img_path_t_05"] = [image_path_root_list[1].format(ID=ID) for ID in df_feature_train.ID]
    df_feature_train["img_path_t_10"] = [image_path_root_list[2].format(ID=ID) for ID in df_feature_train.ID]

    df_feature_test["img_path_t_00"] = [image_path_root_list[0].format(ID=ID) for ID in df_feature_test.ID]
    df_feature_test["img_path_t_05"] = [image_path_root_list[1].format(ID=ID) for ID in df_feature_test.ID]
    df_feature_test["img_path_t_10"] = [image_path_root_list[2].format(ID=ID) for ID in df_feature_test.ID]

    df_feature_train["depth_path_t_00"] = [depth_path_root_list[0].format(ID=ID) for ID in df_feature_train.ID]
    df_feature_train["depth_path_t_05"] = [depth_path_root_list[1].format(ID=ID) for ID in df_feature_train.ID]
    df_feature_train["depth_path_t_10"] = [depth_path_root_list[2].format(ID=ID) for ID in df_feature_train.ID]

    df_feature_test["depth_path_t_00"] = [depth_path_root_list[0].format(ID=ID) for ID in df_feature_test.ID]
    df_feature_test["depth_path_t_05"] = [depth_path_root_list[1].format(ID=ID) for ID in df_feature_test.ID]
    df_feature_test["depth_path_t_10"] = [depth_path_root_list[2].format(ID=ID) for ID in df_feature_test.ID]

    return df_feature_train, df_feature_test, df_sample_submission, FEATIRE_COLMUNS
