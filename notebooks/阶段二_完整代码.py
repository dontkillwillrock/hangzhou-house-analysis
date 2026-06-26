# -*- coding: utf-8 -*-
"""
=================================================================
阶段二：深度特征工程与多模型训练（90分钟）
=================================================================
任务1：深度特征工程（10-30分钟）
任务2：系统性可视化分析（30-45分钟）
任务3：类别编码与数据准备（45-55分钟）
任务4：多模型训练（55-75分钟）
任务5：交叉验证与模型横向对比（75-90分钟）
=================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_DIR = Path("data")
FIG_DIR = Path("figures")
(FIG_DIR / "eda").mkdir(parents=True, exist_ok=True)
(FIG_DIR / "models").mkdir(parents=True, exist_ok=True)

# ============================================================
# 加载阶段一清洗后的数据
# ============================================================
print("=" * 60)
print("加载阶段一清洗后的数据")
print("=" * 60)

house = pd.read_csv(DATA_DIR / "processed" / "hz_house_clean.csv", encoding="utf-8-sig")
print(f"数据形状: {house.shape}")
print(f"缺失值总数: {house.isnull().sum().sum()}")


# ============================================================
# 任务1：深度特征工程（10-30分钟）
# ============================================================
print("\n" + "=" * 60)
print("任务1：深度特征工程")
print("=" * 60)

# 1. 单位面积房间数（空间利用率）
house['单位面积房间数'] = house['室'] / house['建筑面积'].clip(lower=1)
print("✓ 单位面积房间数")

# 2. 学区内小区个数转数值
if '学区内小区个数' in house.columns:
    house['学区内小区个数_数值'] = pd.to_numeric(
        house['学区内小区个数'].astype(str).str.replace('个小区', ''),
        errors='coerce'
    )
    house['学区内小区个数_数值'] = house['学区内小区个数_数值'].fillna(
        house['学区内小区个数_数值'].median()
    )
    print("✓ 学区内小区个数转数值")

# 3. 学校等级评分
school_features = ['区重点', '市重点', '名校附属', '双语']
available_school = [f for f in school_features if f in house.columns]
if available_school:
    house['学校等级评分'] = house[available_school].sum(axis=1)
    print("✓ 学校等级评分")

# 4. 学校特色数量
specialty_features = ['体育类', '艺术类', '语文类', '科技类', '外语类']
available_specialty = [f for f in specialty_features if f in house.columns]
if available_specialty:
    house['学校特色数量'] = house[available_specialty].sum(axis=1)
    print("✓ 学校特色数量")

# 5. 楼层比例
house['楼层比例'] = house['楼层_数值'] / house['总层数'].clip(lower=1)
print("✓ 楼层比例")

# 6. 每室平均面积
house['每室平均面积'] = house['建筑面积'] / house['室'].clip(lower=1)
print("✓ 每室平均面积")

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

house['房龄等级'] = house['房龄'].apply(age_category)
print("✓ 房龄等级")

# 8. 总价单价比
house['总价单价比'] = (house['总价（万元）'] * 10000) / \
    (house['单价（元/平米）'] * house['建筑面积']).clip(lower=1)
print("✓ 总价单价比")

# 验证新增特征
new_features = ['单位面积房间数', '学校等级评分', '学校特色数量',
                '楼层比例', '每室平均面积', '房龄等级', '总价单价比']
print("\n【衍生特征统计】")
print(house[new_features].describe())
print(f"\n当前特征总数（含原始字段）：{house.shape[1]} 列")

print("\n>>> 继续任务2...")


# ============================================================
# 任务2：系统性可视化分析（30-45分钟）
# ============================================================
print("\n" + "=" * 60)
print("任务2：系统性可视化分析")
print("=" * 60)

# ========== 图表1：房价分布直方图 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(house['单价（元/平米）'], bins=50, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('杭州房价单价分布', fontsize=14)
axes[0].set_xlabel('单价（元/平米）')
axes[0].set_ylabel('频数')
mean_price = house['单价（元/平米）'].mean()
median_price = house['单价（元/平米）'].median()
axes[0].axvline(mean_price, color='red', linestyle='--', label=f'均值: {mean_price:.0f}')
axes[0].axvline(median_price, color='green', linestyle='--', label=f'中位数: {median_price:.0f}')
axes[0].legend()

sns.histplot(np.log1p(house['单价（元/平米）']), bins=50, kde=True, ax=axes[1], color='coral')
axes[1].set_title('log(单价+1) 分布', fontsize=14)
axes[1].set_xlabel('log(单价+1)')
axes[1].set_ylabel('频数')
plt.tight_layout()
plt.savefig(FIG_DIR / "eda" / "房价分布.png", dpi=300)
plt.show()
print(f"单价均值：{mean_price:.0f}，中位数：{median_price:.0f}")
print(f"偏度：{house['单价（元/平米）'].skew():.3f}（>0为右偏）")

# ========== 图表2：特征相关性热力图 ==========
plt.figure(figsize=(16, 12))
numeric_cols = house.select_dtypes(include=[np.number]).columns
exclude_num = ['序号'] if '序号' in numeric_cols else []
corr_cols = [c for c in numeric_cols if c not in exclude_num]
corr_matrix = house[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('数值特征相关性热力图', fontsize=16)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda" / "相关性热力图.png", dpi=300)
plt.show()

corr_with_target = corr_matrix['单价（元/平米）'].drop('单价（元/平米）').sort_values(ascending=False)
print("与单价相关性最高的前10个特征：")
print(corr_with_target.head(10))

# ========== 图表3：不同区县房价箱线图 ==========
plt.figure(figsize=(14, 6))
district_order = house.groupby('所在区县')['单价（元/平米）'].median().sort_values(ascending=False).index
sns.boxplot(x='所在区县', y='单价（元/平米）', data=house, order=district_order, palette='Set2')
plt.title('不同区县房价分布（按中位数排序）', fontsize=14)
plt.xlabel('区县')
plt.ylabel('单价（元/平米）')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda" / "区县房价分布.png", dpi=300)
plt.show()
print("各区县房价中位数：")
print(house.groupby('所在区县')['单价（元/平米）'].median().sort_values(ascending=False))

# ========== 图表4：不同朝向的房价箱线图 ==========
plt.figure(figsize=(12, 5))
orient_order = house.groupby('朝向')['单价（元/平米）'].median().sort_values(ascending=False).index
sns.boxplot(x='朝向', y='单价（元/平米）', data=house, order=orient_order, palette='YlOrRd')
plt.title('不同朝向房价分布（按中位数排序）', fontsize=14)
plt.xlabel('朝向')
plt.ylabel('单价（元/平米）')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIG_DIR / "eda" / "朝向房价分布.png", dpi=300)
plt.show()

# ========== 图表5：建筑面积与总价的散点图 ==========
plt.figure(figsize=(10, 6))
plt.scatter(house['建筑面积'], house['总价（万元）'], alpha=0.3, s=10, c='steelblue')
plt.xlabel('建筑面积（平方米）', fontsize=12)
plt.ylabel('总价（万元）', fontsize=12)
plt.title('建筑面积与总价关系', fontsize=14)
z = np.polyfit(house['建筑面积'].dropna(), house.loc[house['建筑面积'].notna(), '总价（万元）'], 1)
p = np.poly1d(z)
x_line = np.linspace(house['建筑面积'].min(), house['建筑面积'].max(), 100)
plt.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'趋势线 (斜率={z[0]:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(FIG_DIR / "eda" / "面积总价关系.png", dpi=300)
plt.show()

# ========== 图表6：房龄与单价的关系 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(house['房龄'], house['单价（元/平米）'], alpha=0.3, s=10, c='coral')
axes[0].set_xlabel('房龄（年）')
axes[0].set_ylabel('单价（元/平米）')
axes[0].set_title('房龄与单价散点图')

age_labels = {1: '次新(≤5年)', 2: '较新(6-10年)', 3: '一般(11-20年)', 4: '老旧(>20年)'}
house['房龄标签'] = house['房龄等级'].map(age_labels)
sns.boxplot(x='房龄标签', y='单价（元/平米）', data=house,
            order=['次新(≤5年)', '较新(6-10年)', '一般(11-20年)', '老旧(>20年)'],
            palette='coolwarm', ax=axes[1])
axes[1].set_title('不同房龄段房价分布')
axes[1].set_xlabel('房龄段')
axes[1].set_ylabel('单价（元/平米）')
plt.tight_layout()
plt.savefig(FIG_DIR / "eda" / "房龄单价关系.png", dpi=300)
plt.show()

# ========== 图表7（创新）：学区等级评分与房价关系 ==========
plt.figure(figsize=(10, 5))
sns.boxplot(x='学校等级评分', y='单价（元/平米）', data=house, palette='Greens')
plt.title('学区等级评分与房价关系', fontsize=14)
plt.xlabel('学校等级评分（区重点+市重点+名校附属+双语）')
plt.ylabel('单价（元/平米）')
plt.savefig(FIG_DIR / "eda" / "学区评分房价关系.png", dpi=300)
plt.show()
print("不同学区评分的房价统计：")
print(house.groupby('学校等级评分')['单价（元/平米）'].agg(['count', 'mean', 'median']))

print("\n>>> 继续任务3...")


# ============================================================
# 任务3：类别编码与数据准备（45-55分钟）
# ============================================================
print("\n" + "=" * 60)
print("任务3：类别编码与数据准备")
print("=" * 60)

# ========== 类别编码 ==========
house_encoded = pd.get_dummies(house, columns=['朝向', '所在区县'], drop_first=True)
print(f"独热编码后数据形状：{house_encoded.shape}")

# ========== 选择建模特征 ==========
feature_cols = ['建筑面积', '室', '厅', '楼层_数值', '总层数', '房龄',
                '距离学校_米', '单位面积房间数', '学校等级评分', '学校特色数量',
                '楼层比例', '每室平均面积', '房龄等级', '总价单价比']

# 添加学区内小区个数
if '学区内小区个数_数值' in house_encoded.columns:
    feature_cols.append('学区内小区个数_数值')

# 二值特征
binary_cols = ['普通', '小班教学', '区重点', '体育类', '艺术类',
               '语文类', '双语', '科技类', '名校附属', '市重点', '外语类']
feature_cols.extend([c for c in binary_cols if c in house_encoded.columns])

# 独热编码后的区县和朝向列
dummy_cols = [c for c in house_encoded.columns
              if c.startswith('所在区县_') or c.startswith('朝向_')]
feature_cols.extend(dummy_cols)

# 过滤存在的列
feature_cols = [c for c in feature_cols if c in house_encoded.columns]

# 构造特征矩阵和目标变量
X = house_encoded[feature_cols].copy()
y = house_encoded['单价（元/平米）'].copy()

# 处理缺失值
X = X.fillna(X.median())

print(f"特征矩阵形状：{X.shape}")
print(f"特征数量：{len(feature_cols)}")

# ========== 划分训练集和测试集（8:2） ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n训练集：{X_train.shape[0]} 条")
print(f"测试集：{X_test.shape[0]} 条")

# ========== 标准化（仅在训练集上fit） ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n标准化后训练集均值：{np.mean(X_train_scaled, axis=0).mean():.6f}（接近0）")
print(f"标准化后训练集标准差：{np.std(X_train_scaled, axis=0).mean():.6f}（接近1）")

print("\n>>> 继续任务4...")


# ============================================================
# 任务4：多模型训练（55-75分钟）
# ============================================================
print("\n" + "=" * 60)
print("任务4：多模型训练")
print("=" * 60)

results = []
models = {}

# ===== 模型1：线性回归（基线模型） =====
print("\n--- 模型1：线性回归 ---")
t0 = time.time()
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
t1 = time.time()

rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae = mean_absolute_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)
results.append({'模型': '线性回归', 'RMSE': round(rmse, 2),
                'MAE': round(mae, 2), 'R²': round(r2, 4),
                '训练时间(s)': round(t1 - t0, 2)})
models['线性回归'] = lr_model
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")
print(f"  R²:   {r2:.4f}")
print(f"  训练时间: {t1-t0:.2f}s")

# ===== 模型2：随机森林回归 =====
print("\n--- 模型2：随机森林回归 ---")
t0 = time.time()
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=15, min_samples_split=5,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
t1 = time.time()

rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae = mean_absolute_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)
results.append({'模型': '随机森林', 'RMSE': round(rmse, 2),
                'MAE': round(mae, 2), 'R²': round(r2, 4),
                '训练时间(s)': round(t1 - t0, 2)})
models['随机森林'] = rf_model
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")
print(f"  R²:   {r2:.4f}")
print(f"  训练时间: {t1-t0:.2f}s")

# ===== 模型3：梯度提升回归 =====
print("\n--- 模型3：梯度提升回归 ---")
t0 = time.time()
gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42
)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
t1 = time.time()

rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
mae = mean_absolute_error(y_test, y_pred_gb)
r2 = r2_score(y_test, y_pred_gb)
results.append({'模型': '梯度提升', 'RMSE': round(rmse, 2),
                'MAE': round(mae, 2), 'R²': round(r2, 4),
                '训练时间(s)': round(t1 - t0, 2)})
models['梯度提升'] = gb_model
print(f"  RMSE: {rmse:.2f}")
print(f"  MAE:  {mae:.2f}")
print(f"  R²:   {r2:.4f}")
print(f"  训练时间: {t1-t0:.2f}s")

# ===== 汇总对比 =====
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("模型对比表：")
print(results_df.to_string(index=False))

# ========== 模型对比可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
results_df.plot(x='模型', y=['RMSE', 'MAE'], kind='bar', ax=axes[0], rot=0)
axes[0].set_title('RMSE与MAE对比', fontsize=13)
axes[0].set_ylabel('误差值')
results_df.plot(x='模型', y='R²', kind='bar', ax=axes[1], rot=0, color='green')
axes[1].set_title('R²对比', fontsize=13)
axes[1].set_ylabel('R²')
plt.suptitle('回归模型横向对比', fontsize=16)
plt.tight_layout()
plt.savefig(FIG_DIR / "models" / "模型对比.png", dpi=300)
plt.show()

print("\n>>> 继续任务5...")


# ============================================================
# 任务5：交叉验证与模型横向对比（75-90分钟）
# ============================================================
print("\n" + "=" * 60)
print("任务5：交叉验证与模型横向对比")
print("=" * 60)

print("\n【5折交叉验证】")

# 线性回归
lr_cv = cross_val_score(LinearRegression(), X_train_scaled, y_train,
                        cv=5, scoring='r2', n_jobs=-1)
print(f"线性回归 5折CV R²：{lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
print(f"  各折得分：{[f'{s:.4f}' for s in lr_cv]}")

# 随机森林
rf_cv_model = RandomForestRegressor(n_estimators=200, max_depth=15,
                                    random_state=42, n_jobs=-1)
rf_cv = cross_val_score(rf_cv_model, X_train_scaled, y_train,
                        cv=5, scoring='r2', n_jobs=-1)
print(f"随机森林 5折CV R²：{rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print(f"  各折得分：{[f'{s:.4f}' for s in rf_cv]}")

# 梯度提升
gb_cv_model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                        learning_rate=0.1, random_state=42)
gb_cv = cross_val_score(gb_cv_model, X_train_scaled, y_train,
                        cv=5, scoring='r2', n_jobs=-1)
print(f"梯度提升 5折CV R²：{gb_cv.mean():.4f} ± {gb_cv.std():.4f}")
print(f"  各折得分：{[f'{s:.4f}' for s in gb_cv]}")

# 汇总CV结果
cv_results = pd.DataFrame({
    '模型': ['线性回归', '随机森林', '梯度提升'],
    'CV均值': [lr_cv.mean(), rf_cv.mean(), gb_cv.mean()],
    'CV标准差': [lr_cv.std(), rf_cv.std(), gb_cv.std()],
    '测试集R²': [r2_score(y_test, y_pred_lr),
                 r2_score(y_test, y_pred_rf),
                 r2_score(y_test, y_pred_gb)]
})
cv_results['CV-测试集差距'] = cv_results['CV均值'] - cv_results['测试集R²']
print("\n交叉验证汇总：")
print(cv_results.to_string(index=False))

# ========== 特征重要性分析 ==========
print("\n【特征重要性分析（随机森林）】")
importance_df = pd.DataFrame({
    '特征': feature_cols,
    '重要性': rf_model.feature_importances_
}).sort_values('重要性', ascending=False)
print(importance_df.head(15))

plt.figure(figsize=(10, 6))
top_features = importance_df.head(15)
plt.barh(top_features['特征'][::-1], top_features['重要性'][::-1], color='steelblue')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.title('随机森林特征重要性Top15')
plt.tight_layout()
plt.savefig(FIG_DIR / "models" / "特征重要性.png", dpi=300)
plt.show()


# ============================================================
# 保存模型
# ============================================================
import joblib
joblib.dump(rf_model, DATA_DIR / ".." / "models" / "random_forest_model.pkl")
joblib.dump(gb_model, DATA_DIR / ".." / "models" / "gradient_boosting_model.pkl")
joblib.dump(scaler, DATA_DIR / ".." / "models" / "scaler.pkl")
print("\n✓ 模型已保存至 models/ 目录")


# ============================================================
# 阶段检查清单
# ============================================================
print("\n" + "=" * 60)
print("阶段二检查清单")
print("=" * 60)
checklist = """
序号  检查项目                                    通过
1     至少构造了3个有业务含义的衍生特征           [ ]
2     每个衍生特征的分布已检查                    [ ]
3     完成了至少6张可视化图表                     [ ]
4     每张图表都有分析结论记录                    [ ]
5     类别编码策略合理                            [ ]
6     训练集/测试集已正确划分（8:2）             [ ]
7     Scaler仅在训练集上fit                       [ ]
8     至少训练了3种模型                           [ ]
9     每种模型都输出了测试集预测结果              [ ]
10    完成了5折交叉验证                           [ ]
11    模型对比表包含至少3个指标                   [ ]
12    对模型差异做了初步分析                      [ ]
"""
print(checklist)
print("✓ 阶段二完成！")
