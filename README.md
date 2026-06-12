# 实验6：杭州学区房数据分析与房价预测

## 项目概述

本项目对杭州市学区房市场数据进行分析，通过数据挖掘和机器学习方法，构建房价预测模型，帮助购房者评估学区房价格是否合理，辅助投资决策。

## 项目结构

```
experiment_06/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据（不可修改）
│   │   └── hz_house.csv           # 杭州房价原始数据
│   ├── processed/                 # 清洗后的数据
│   │   └── hz_house_clean.csv
│   └── feature/                   # 特征工程后的数据
│       └── hz_house_feature.csv
│
├── notebooks/                     # Jupyter Notebook 目录
│   ├── 01_data_exploration.ipynb  # 阶段一：探索性分析
│   ├── 02_feature_engineering.ipynb  # 阶段一/二：特征工程
│   ├── 03_model_training.ipynb    # 阶段二：模型训练
│   ├── 04_model_evaluation.ipynb  # 阶段三：评估优化
│   └── 05_final_report.ipynb      # 阶段三：最终报告
│
├── src/                           # 可复用的 Python 模块
│   ├── __init__.py               # 包初始化
│   ├── data_loader.py            # 数据加载函数
│   ├── feature_engineering.py    # 特征工程函数
│   ├── model_utils.py            # 模型工具函数
│   └── evaluation.py             # 评估工具函数
│
├── models/                        # 保存训练好的模型
│   ├── best_rf_model.pkl
│   ├── best_gb_model.pkl
│   └── scaler.pkl
│
├── figures/                       # 图表输出目录
│   ├── eda/                       # 探索性分析图表
│   ├── models/                    # 模型对比图表
│   └── evaluation/                # 评估结果图表
│
├── reports/                       # 报告文档
│   ├── stage1_eda_report.md       # 阶段一报告
│   ├── stage2_model_report.md     # 阶段二报告
│   └── final_report.md            # 最终实验报告
│
├── requirements.txt               # 依赖包列表
└── README.md                      # 项目说明
```

## 实验阶段

### 阶段一：数据探索性分析（90分钟）

**任务：**
1. 数据加载与结构检查
2. 数据质量评估（缺失值、异常值）
3. 基础统计描述
4. 数据分布可视化

**Notebook：** `01_data_exploration.ipynb`

### 阶段二：特征工程与模型训练（90分钟）

**任务：**
1. 数据清洗与缺失值处理
2. 特征提取与衍生特征构造
3. 类别编码（独热编码）
4. 训练集/测试集划分与标准化
5. 训练至少3种模型
6. 交叉验证与模型对比

**Notebook：** `02_feature_engineering.ipynb`, `03_model_training.ipynb`

### 阶段三：模型评估与优化（90分钟）

**任务：**
1. 超参调优（GridSearchCV）
2. 模型评估与可视化
3. 特征重要性分析
4. 业务洞察提炼
5. 实验报告撰写

**Notebook：** `04_model_evaluation.ipynb`, `05_final_report.ipynb`

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行Notebook

```bash
cd experiment_06/notebooks
jupyter notebook
```

按顺序运行：
1. `01_data_exploration.ipynb` - 数据探索
2. `02_feature_engineering.ipynb` - 特征工程
3. `03_model_training.ipynb` - 模型训练
4. `04_model_evaluation.ipynb` - 模型评估
5. `05_final_report.ipynb` - 最终报告

## 数据说明

| 字段 | 说明 |
|------|------|
| 序号 | 记录编号 |
| 项目名称 | 小区名称 |
| 房源描述 | 房源详细描述 |
| 建筑面积 | 房屋面积（平方米） |
| 户型 | 房屋户型（如3室2厅） |
| 楼层 | 楼层位置（低层/中层/高层） |
| 总层数 | 建筑总层数 |
| 朝向 | 房屋朝向 |
| 建筑年代 | 建造年份 |
| 单价（元/平米） | 每平方米价格 |
| 总价（万元） | 房屋总价 |
| 所在学校 | 对应学校名称 |
| 距离学校 | 与学校的距离 |
| 所在区县 | 所属区域 |

## 模型说明

### 基线模型
- **线性回归：** 简单线性模型，可解释性强

### 集成模型
- **随机森林：** Bagging集成方法，抗过拟合
- **梯度提升：** Boosting集成方法，预测精度高

## 评估指标

| 指标 | 说明 |
|------|------|
| RMSE | 均方根误差，越小越好 |
| MAE | 平均绝对误差，越小越好 |
| R² | 决定系数，越接近1越好 |
| MAPE | 平均绝对百分比误差 |

## 作者

数据分析实验6 - 杭州学区房数据分析

## 许可证

仅供学习使用
