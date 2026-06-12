# -*- coding: utf-8 -*-
"""
评估工具模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def calculate_metrics(y_true, y_pred, model_name="模型"):
    """
    计算回归评估指标

    Parameters
    ----------
    y_true : pd.Series
        真实值
    y_pred : np.ndarray
        预测值
    model_name : str
        模型名称

    Returns
    -------
    dict
        评估指标字典
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        "模型": model_name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R²": round(r2, 4),
        "MAPE(%)": round(mape, 2)
    }


def plot_model_comparison(results_df: pd.DataFrame, save_dir: Path):
    """
    绘制模型对比图

    Parameters
    ----------
    results_df : pd.DataFrame
        模型结果DataFrame
    save_dir : Path
        保存目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE与MAE对比
    results_df.plot(x="模型", y=["RMSE", "MAE"], kind="bar", ax=axes[0], rot=0)
    axes[0].set_title("RMSE与MAE对比", fontsize=13)
    axes[0].set_ylabel("误差值")
    axes[0].legend(loc="upper right")

    # R²对比
    results_df.plot(x="模型", y="R²", kind="bar", ax=axes[1], rot=0, color="green")
    axes[1].set_title("R²对比", fontsize=13)
    axes[1].set_ylabel("R²")
    axes[1].set_ylim(0, 1)

    plt.suptitle("回归模型横向对比", fontsize=16)
    plt.tight_layout()

    save_path = save_dir / "model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ 模型对比图已保存至: {save_path}")


def plot_prediction_vs_actual(y_true, y_pred, model_name: str, save_dir: Path):
    """
    绘制预测值vs真实值散点图

    Parameters
    ----------
    y_true : pd.Series
        真实值
    y_pred : np.ndarray
        预测值
    model_name : str
        模型名称
    save_dir : Path
        保存目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 散点图
    axes[0].scatter(y_true, y_pred, alpha=0.5, s=10, c="steelblue")
    max_val = max(y_true.max(), max(y_pred))
    min_val = min(y_true.min(), min(y_pred))
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="理想预测线")
    axes[0].set_xlabel("真实值（元/平米）", fontsize=12)
    axes[0].set_ylabel("预测值（元/平米）", fontsize=12)
    axes[0].set_title(f"{model_name} - 预测值vs真实值", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 残差分布
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=50, color="coral", edgecolor="black", alpha=0.7)
    axes[1].axvline(x=0, color="black", linestyle="--", linewidth=2)
    axes[1].set_xlabel("残差（元/平米）", fontsize=12)
    axes[1].set_ylabel("频数", fontsize=12)
    axes[1].set_title(f"{model_name} - 残差分布", fontsize=14)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    save_path = save_dir / f"{model_name}_prediction.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ {model_name}预测图已保存至: {save_path}")


def plot_feature_importance(model, feature_names: list, model_name: str, save_dir: Path, top_n: int = 20):
    """
    绘制特征重要性图

    Parameters
    ----------
    model : object
        训练好的模型
    feature_names : list
        特征名称列表
    model_name : str
        模型名称
    save_dir : Path
        保存目录
    top_n : int
        显示前N个特征
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        print(f"⚠ {model_name} 不支持特征重要性分析")
        return

    importance_df = pd.DataFrame({
        "特征": feature_names,
        "重要性": importances
    }).sort_values("重要性", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["特征"][::-1], importance_df["重要性"][::-1], color="steelblue")
    plt.xlabel("重要性", fontsize=12)
    plt.ylabel("特征", fontsize=12)
    plt.title(f"{model_name} - 特征重要性Top{top_n}", fontsize=14)
    plt.tight_layout()

    save_path = save_dir / f"{model_name}_feature_importance.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ {model_name}特征重要性图已保存至: {save_path}")

    return importance_df


def plot_residual_analysis(y_true, y_pred, model_name: str, save_dir: Path):
    """
    残差分析图

    Parameters
    ----------
    y_true : pd.Series
        真实值
    y_pred : np.ndarray
        预测值
    model_name : str
        模型名称
    save_dir : Path
        保存目录
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 残差分布
    axes[0, 0].hist(residuals, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(x=0, color="red", linestyle="--", linewidth=2)
    axes[0, 0].set_title("残差分布")
    axes[0, 0].set_xlabel("残差")
    axes[0, 0].set_ylabel("频数")

    # 残差vs预测值
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10, c="coral")
    axes[0, 1].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[0, 1].set_title("残差vs预测值")
    axes[0, 1].set_xlabel("预测值")
    axes[0, 1].set_ylabel("残差")

    # Q-Q图
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q图")

    # 残差自相关（按索引）
    axes[1, 1].plot(range(len(residuals)), residuals, color="steelblue", alpha=0.5)
    axes[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[1, 1].set_title("残差时序图")
    axes[1, 1].set_xlabel("样本索引")
    axes[1, 1].set_ylabel("残差")

    plt.suptitle(f"{model_name} - 残差分析", fontsize=14)
    plt.tight_layout()

    save_path = save_dir / f"{model_name}_residual_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ {model_name}残差分析图已保存至: {save_path}")
