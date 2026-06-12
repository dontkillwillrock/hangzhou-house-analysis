# -*- coding: utf-8 -*-
"""
模型工具模块
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time


def prepare_model_data(df: pd.DataFrame, target_col: str = "单价（元/平米）"):
    """
    准备建模数据（类别编码、划分训练集/测试集、标准化）

    Parameters
    ----------
    df : pd.DataFrame
        特征工程后的数据
    target_col : str
        目标变量列名

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler, feature_cols)
    """
    print("=" * 50)
    print("准备建模数据")
    print("=" * 50)

    # 类别编码（独热编码）
    df_encoded = pd.get_dummies(df, columns=["朝向", "所在区县"], drop_first=True)

    # 选择建模特征
    feature_cols = [
        "建筑面积", "室", "厅", "楼层_数值", "总层数", "房龄",
        "距离学校_米", "单位面积房间数", "学校等级评分", "学校特色数量",
        "楼层比例", "每室平均面积", "房龄等级", "总价单价比"
    ]

    # 添加二值特征
    binary_cols = ["普通", "小班教学", "区重点", "体育类", "艺术类",
                   "语文类", "双语", "科技类", "名校附属", "市重点", "外语类"]
    feature_cols.extend([c for c in binary_cols if c in df_encoded.columns])

    # 添加学区内小区个数
    if "学区内小区个数_数值" in df_encoded.columns:
        feature_cols.append("学区内小区个数_数值")

    # 添加独热编码后的列
    dummy_cols = [c for c in df_encoded.columns
                  if c.startswith("所在区县_") or c.startswith("朝向_")]
    feature_cols.extend(dummy_cols)

    # 过滤存在的列
    feature_cols = [c for c in feature_cols if c in df_encoded.columns]

    # 构造特征矩阵和目标变量
    X = df_encoded[feature_cols].copy()
    y = df_encoded[target_col].copy()

    # 处理缺失值
    X = X.fillna(X.median())

    print(f"特征矩阵形状: {X.shape}")
    print(f"特征数量: {len(feature_cols)}")

    # 划分训练集和测试集（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集: {X_train.shape[0]} 条")
    print(f"测试集: {X_test.shape[0]} 条")

    # 标准化（仅在训练集上fit）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"标准化后训练集均值: {np.mean(X_train_scaled, axis=0).mean():.6f}")
    print(f"标准化后训练集标准差: {np.std(X_train_scaled, axis=0).mean():.6f}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def train_baseline_models(X_train, y_train, X_test, y_test):
    """
    训练基线模型

    Parameters
    ----------
    X_train : np.ndarray
        训练集特征
    y_train : pd.Series
        训练集目标变量
    X_test : np.ndarray
        测试集特征
    y_test : pd.Series
        测试集目标变量

    Returns
    -------
    tuple
        (models_dict, results_df)
    """
    print("=" * 60)
    print("训练基线模型")
    print("=" * 60)

    results = []
    models = {}

    # 模型1：线性回归
    print("\n--- 模型1：线性回归 ---")
    t0 = time.time()
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    t1 = time.time()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae = mean_absolute_error(y_test, y_pred_lr)
    r2 = r2_score(y_test, y_pred_lr)
    results.append({
        "模型": "线性回归", "RMSE": round(rmse, 2),
        "MAE": round(mae, 2), "R²": round(r2, 4),
        "训练时间(s)": round(t1 - t0, 2)
    })
    models["线性回归"] = lr_model
    print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    # 模型2：随机森林
    print("\n--- 模型2：随机森林 ---")
    t0 = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    t1 = time.time()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae = mean_absolute_error(y_test, y_pred_rf)
    r2 = r2_score(y_test, y_pred_rf)
    results.append({
        "模型": "随机森林", "RMSE": round(rmse, 2),
        "MAE": round(mae, 2), "R²": round(r2, 4),
        "训练时间(s)": round(t1 - t0, 2)
    })
    models["随机森林"] = rf_model
    print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    # 模型3：梯度提升
    print("\n--- 模型3：梯度提升 ---")
    t0 = time.time()
    gb_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    t1 = time.time()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    mae = mean_absolute_error(y_test, y_pred_gb)
    r2 = r2_score(y_test, y_pred_gb)
    results.append({
        "模型": "梯度提升", "RMSE": round(rmse, 2),
        "MAE": round(mae, 2), "R²": round(r2, 4),
        "训练时间(s)": round(t1 - t0, 2)
    })
    models["梯度提升"] = gb_model
    print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("模型对比表")
    print(results_df.to_string(index=False))

    return models, results_df


def cross_validate_models(models: dict, X_train, y_train, cv: int = 5):
    """
    交叉验证模型

    Parameters
    ----------
    models : dict
        模型字典
    X_train : np.ndarray
        训练集特征
    y_train : pd.Series
        训练集目标变量
    cv : int
        交叉验证折数

    Returns
    -------
    pd.DataFrame
        交叉验证结果
    """
    print("=" * 60)
    print(f"{cv}折交叉验证")
    print("=" * 60)

    cv_results = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
        cv_results.append({
            "模型": name,
            "CV均值": round(scores.mean(), 4),
            "CV标准差": round(scores.std(), 4),
            "各折得分": [round(s, 4) for s in scores]
        })
        print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

    return pd.DataFrame(cv_results)
