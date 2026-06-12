# -*- coding: utf-8 -*-
"""
数据加载模块
"""
import pandas as pd
from pathlib import Path


def load_raw_data(data_dir: Path) -> pd.DataFrame:
    """
    加载原始数据

    Parameters
    ----------
    data_dir : Path
        数据目录路径

    Returns
    -------
    pd.DataFrame
        原始数据
    """
    raw_path = data_dir / "raw" / "hz_house.csv"
    df = pd.read_csv(raw_path, encoding="utf-8")
    print(f"✓ 成功加载原始数据: {raw_path}")
    print(f"  数据形状: {df.shape[0]} 行, {df.shape[1]} 列")
    return df


def save_processed_data(df: pd.DataFrame, data_dir: Path, filename: str = "hz_house_clean.csv"):
    """
    保存处理后的数据

    Parameters
    ----------
    df : pd.DataFrame
        处理后的数据
    data_dir : Path
        数据目录路径
    filename : str
        保存文件名
    """
    save_path = data_dir / "processed" / filename
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✓ 数据已保存至: {save_path}")


def save_feature_data(df: pd.DataFrame, data_dir: Path, filename: str = "hz_house_feature.csv"):
    """
    保存特征工程后的数据

    Parameters
    ----------
    df : pd.DataFrame
        特征工程后的数据
    data_dir : Path
        数据目录路径
    filename : str
        保存文件名
    """
    save_path = data_dir / "feature" / filename
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✓ 特征数据已保存至: {save_path}")


def load_processed_data(data_dir: Path) -> pd.DataFrame:
    """加载清洗后的数据"""
    path = data_dir / "processed" / "hz_house_clean.csv"
    return pd.read_csv(path, encoding="utf-8-sig")


def load_feature_data(data_dir: Path) -> pd.DataFrame:
    """加载特征工程后的数据"""
    path = data_dir / "feature" / "hz_house_feature.csv"
    return pd.read_csv(path, encoding="utf-8-sig")
