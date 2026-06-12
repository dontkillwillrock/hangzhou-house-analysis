# -*- coding: utf-8 -*-
"""
特征工程模块
"""
import re
import pandas as pd
import numpy as np


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理缺失值

    Parameters
    ----------
    df : pd.DataFrame
        原始数据

    Returns
    -------
    pd.DataFrame
        处理缺失值后的数据
    """
    df = df.copy()

    # 朝向：类别型，用众数填补
    if "朝向" in df.columns:
        df["朝向"] = df["朝向"].fillna(df["朝向"].mode()[0])

    # 建筑年代：数值型，用中位数填补
    if "建筑年代" in df.columns:
        df["建筑年代"] = pd.to_numeric(df["建筑年代"], errors="coerce")
        df["建筑年代"] = df["建筑年代"].fillna(df["建筑年代"].median())

    # 建筑面积：数值型，用中位数填补
    if "建筑面积" in df.columns:
        df["建筑面积"] = pd.to_numeric(df["建筑面积"], errors="coerce")
        df["建筑面积"] = df["建筑面积"].fillna(df["建筑面积"].median())

    # 处理剩余缺失值
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    print(f"✓ 缺失值处理完成, 剩余缺失值: {df.isnull().sum().sum()}")
    return df


def extract_room_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从户型中提取室数和厅数

    Parameters
    ----------
    df : pd.DataFrame
        包含户型字段的数据

    Returns
    -------
    pd.DataFrame
        添加室、厅字段后的数据
    """
    df = df.copy()

    def extract_rooms(layout):
        """从户型文本中提取室数和厅数"""
        if pd.isna(layout):
            return pd.Series({"室": np.nan, "厅": np.nan})
        match = re.search(r"(\d+)室(\d+)厅", str(layout))
        if match:
            return pd.Series({"室": int(match.group(1)), "厅": int(match.group(2))})
        return pd.Series({"室": np.nan, "厅": np.nan})

    room_info = df["户型"].apply(extract_rooms)
    df["室"] = room_info["室"]
    df["厅"] = room_info["厅"]

    # 填充提取失败的记录
    df["室"] = df["室"].fillna(df["室"].mode()[0])
    df["厅"] = df["厅"].fillna(df["厅"].mode()[0])

    print("✓ 户型特征提取完成")
    return df


def extract_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    从距离学校字段提取数值距离

    Parameters
    ----------
    df : pd.DataFrame
        包含距离学校字段的数据

    Returns
    -------
    pd.DataFrame
        添加距离学校_米字段后的数据
    """
    df = df.copy()

    def extract_distance_value(dist_str):
        """从距离文本中提取数值（单位：米）"""
        if pd.isna(dist_str):
            return np.nan
        dist_str = str(dist_str)
        # 匹配"Xkm"或"X公里"
        match_km = re.search(r"([\d.]+)\s*(?:公里|km)", dist_str, re.IGNORECASE)
        if match_km:
            return float(match_km.group(1)) * 1000
        # 匹配"X米"或"Xm"
        match_m = re.search(r"([\d.]+)\s*(?:米|m)", dist_str, re.IGNORECASE)
        if match_m:
            return float(match_m.group(1))
        # 仅匹配数字
        match_num = re.search(r"([\d.]+)", dist_str)
        if match_num:
            return float(match_num.group(1))
        return np.nan

    df["距离学校_米"] = df["距离学校"].apply(extract_distance_value)
    df["距离学校_米"] = df["距离学校_米"].fillna(df["距离学校_米"].median())

    print("✓ 距离特征提取完成")
    return df


def create_building_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造房龄特征

    Parameters
    ----------
    df : pd.DataFrame
        包含建筑年代字段的数据

    Returns
    -------
    pd.DataFrame
        添加房龄字段后的数据
    """
    df = df.copy()
    df["房龄"] = 2026 - df["建筑年代"]
    # 处理不合理的房龄
    df.loc[df["房龄"] < 0, "房龄"] = 0
    df.loc[df["房龄"] > 100, "房龄"] = df["房龄"].median()
    print("✓ 房龄特征构造完成")
    return df


def encode_floor(df: pd.DataFrame) -> pd.DataFrame:
    """
    楼层编码

    Parameters
    ----------
    df : pd.DataFrame
        包含楼层字段的数据

    Returns
    -------
    pd.DataFrame
        添加楼层_数值字段后的数据
    """
    df = df.copy()
    floor_map = {"低层": 1, "中层": 2, "高层": 3}
    df["楼层_数值"] = df["楼层"].map(floor_map)
    df["楼层_数值"] = df["楼层_数值"].fillna(df["楼层_数值"].mode()[0])
    print("✓ 楼层编码完成")
    return df


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    二值特征编码（是/否 -> 0/1）

    Parameters
    ----------
    df : pd.DataFrame
        包含二值字段的数据

    Returns
    -------
    pd.DataFrame
        编码后的数据
    """
    df = df.copy()
    binary_cols = ["普通", "小班教学", "区重点", "体育类", "艺术类",
                   "语文类", "双语", "科技类", "名校附属", "市重点", "外语类"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"是": 1, "否": 0})
            df[col] = df[col].fillna(0).astype(int)
    print("✓ 二值特征编码完成")
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造衍生特征

    Parameters
    ----------
    df : pd.DataFrame
        基础特征数据

    Returns
    -------
    pd.DataFrame
        添加衍生特征后的数据
    """
    df = df.copy()

    # 1. 单位面积房间数（空间利用率）
    df["单位面积房间数"] = df["室"] / df["建筑面积"].clip(lower=1)

    # 2. 学区内小区个数转数值
    if "学区内小区个数" in df.columns:
        df["学区内小区个数_数值"] = pd.to_numeric(
            df["学区内小区个数"].str.replace("个小区", ""), errors="coerce"
        )
        df["学区内小区个数_数值"] = df["学区内小区个数_数值"].fillna(
            df["学区内小区个数_数值"].median()
        )

    # 3. 学校等级评分
    school_features = ["区重点", "市重点", "名校附属", "双语"]
    available_school = [f for f in school_features if f in df.columns]
    if available_school:
        df["学校等级评分"] = df[available_school].sum(axis=1)

    # 4. 学校特色数量
    specialty_features = ["体育类", "艺术类", "语文类", "科技类", "外语类"]
    available_specialty = [f for f in specialty_features if f in df.columns]
    if available_specialty:
        df["学校特色数量"] = df[available_specialty].sum(axis=1)

    # 5. 楼层比例
    df["楼层比例"] = df["楼层_数值"] / df["总层数"].clip(lower=1)

    # 6. 每室平均面积
    df["每室平均面积"] = df["建筑面积"] / df["室"].clip(lower=1)

    # 7. 房龄等级
    def age_category(age):
        if pd.isna(age):
            return np.nan
        if age <= 5:
            return 1  # 次新房
        elif age <= 10:
            return 2  # 较新
        elif age <= 20:
            return 3  # 一般
        else:
            return 4  # 老旧

    df["房龄等级"] = df["房龄"].apply(age_category)

    # 8. 总价单价比
    df["总价单价比"] = (df["总价（万元）"] * 10000) / \
        (df["单价（元/平米）"] * df["建筑面积"]).clip(lower=1)

    print("✓ 衍生特征构造完成")
    return df


def full_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    完整的特征工程流水线

    Parameters
    ----------
    df : pd.DataFrame
        原始数据

    Returns
    -------
    pd.DataFrame
        特征工程后的数据
    """
    print("=" * 50)
    print("开始特征工程流水线")
    print("=" * 50)

    # 步骤1：处理缺失值
    df = handle_missing_values(df)

    # 步骤2：删除标识字段
    df = df.drop(columns=["序号"], errors="ignore")
    print("✓ 标识字段删除完成")

    # 步骤3：提取户型特征
    df = extract_room_features(df)

    # 步骤4：提取距离特征
    df = extract_distance(df)

    # 步骤5：构造房龄
    df = create_building_age(df)

    # 步骤6：楼层编码
    df = encode_floor(df)

    # 步骤7：二值特征编码
    df = encode_binary_features(df)

    # 步骤8：构造衍生特征
    df = create_derived_features(df)

    print("=" * 50)
    print(f"特征工程完成, 数据形状: {df.shape}")
    print("=" * 50)

    return df
