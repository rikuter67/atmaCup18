# utils.py
import os
import sys
import logging
import random
import numpy as np
import torch
import pandas as pd

LOGGER = logging.getLogger()
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

def setup_logger(out_file=None, stderr=True, stderr_level=logging.INFO, file_level=logging.DEBUG):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER

def seed_torch(seed=516):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def mae(gt: np.array, pred: np.array) -> float:
    abs_diff = np.abs(gt - pred)
    mae = np.mean(abs_diff.reshape(-1, ))
    return mae.astype(float)

# 特徴量生成関数
def create_features(features):
    grouped = features.groupby('scene')
    
    # 数値データのみを選択
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    
    # 平均値
    mean_features = grouped[numeric_cols].mean().add_suffix('_mean')
    # 標準偏差
    std_features = grouped[numeric_cols].std().add_suffix('_std')
    # 最大値
    max_features = grouped[numeric_cols].max().add_suffix('_max')
    # 最小値
    min_features = grouped[numeric_cols].min().add_suffix('_min')
    # 中央値
    median_features = grouped[numeric_cols].median().add_suffix('_median')
    # 四分位範囲
    iqr_features = (grouped[numeric_cols].quantile(0.75) - grouped[numeric_cols].quantile(0.25)).add_suffix('_iqr')
    # 歪度
    skew_features = grouped[numeric_cols].skew().add_suffix('_skew')
    # 尖度
    kurtosis_features = grouped[numeric_cols].apply(pd.DataFrame.kurt).add_suffix('_kurtosis')
    # 合計
    sum_features = grouped[numeric_cols].sum().add_suffix('_sum')

    # 相関行列（オプション、計算量が多いため注意）
    # corr_features = grouped[numeric_cols].apply(lambda x: x.corr().unstack().drop_duplicates()).add_prefix('corr_')

    # 特徴量を結合
    all_features = pd.concat([mean_features, std_features, max_features, min_features, median_features,
                              iqr_features, skew_features, kurtosis_features, sum_features], axis=1)

    return all_features
