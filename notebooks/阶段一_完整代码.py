# -*- coding: utf-8 -*-
"""
=================================================================
阶段一：数据获取、探索分析与特征工程基础（90分钟）
=================================================================
任务1：理解业务问题与字段角色（0-10分钟）
任务2：读取数据并检查结构（10-25分钟）
任务3：数据质量评估（25-40分钟）
任务4：基础统计描述（40-55分钟）
任务5：初步数据清洗与基础特征提取（55-75分钟）
=================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = Path("data")

# ============================================================
# 任务1：理解业务问题与字段角色（0-10分钟）
# ============================================================
print("=" * 60)
print("任务1：理解业务问题与字段角色")
print("=" * 60)

# 业务背景
print("""
【业务问题】预测杭州学区房的单价，为购房者提供价格参考

【目标变量】单价（元/平米）（数值型）

【任务类型】回归任务——预测连续数值

【关键特征】建筑面积、户型、楼层、建筑年代、朝向、距离学校距离、学校属性等

【业务价值】帮助购房者评估学区房价格是否合理，辅助投资决策
""")

print("\n>>> 继续任务2...")


# ============================================================
# 任务2：读取数据并检查结构（10-25分钟）
# ============================================================
print("=" * 60)
print("任务2：读取数据并检查结构")
print("=" * 60)

# 读取数据
house = pd.read_csv(DATA_DIR / "raw" / "hz_house.csv", encoding="utf-8")

# 1. 查看数据规模
print("\n【数据规模】")
print(f"行数：{house.shape[0]}")
print(f"列数：{house.shape[1]}")
print()

# 2. 查看前5行数据
print("【前5行数据预览】")
print(house.head())
print()

# 3. 查看数据基本信息
print("【数据基本信息】")
house.info()
print()

# 4. 查看数据类型分布
print("【数据类型分布】")
print(house.dtypes.value_counts())
print()

# 5. 查看数值特征的基本统计
print("【数值特征基本统计】")
print(house.describe())
print()

# 6. 查看所有列名
print("【所有列名】")
for i, col in enumerate(house.columns):
    print(f"  {i+1:2d}. {col} ({house[col].dtype})")

print("\n>>> 继续任务3...")


# ============================================================
# 任务3：数据质量评估（25-40分钟）
# ============================================================
print("\n" + "=" * 60)
print("任务3：数据质量评估")
print("=" * 60)

# --- 1. 缺失值统计 ---
print("\n【缺失值统计】")
missing_count = house.isnull().sum()
missing_pct = (house.isnull().sum() / len(house) * 100).round(2)
missing_df = pd.DataFrame({
    '缺失数量': missing_count,
    '缺失比例(%)': missing_pct
})
missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=False)
print(missing_df)
print(f"\n共有 {len(missing_df)} 个字段存在缺失值")
print()

# --- 2. 缺失值可视化 ---
missing = house.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
plt.figure(figsize=(10, 4))
missing.plot(kind='bar', color='coral')
plt.title('各字段缺失值统计', fontsize=14)
plt.xlabel('字段名')
plt.ylabel('缺失数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figures/eda/缺失值统计.png', dpi=300)
plt.show()

# --- 3. 数值特征异常值检测（箱线图） ---
print("\n【数值特征异常值检测】")
numeric_cols = house.select_dtypes(include=[np.number]).columns.tolist()
key_numeric = [col for col in numeric_cols if col not in ['序号']]
n_cols = min(len(key_numeric), 8)
n_rows = (len(key_numeric) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
# 统一展平 axes，兼容 1x1 / 1xN / NxM 三种情况
if n_rows == 1 and n_cols == 1:
    axes = np.array([axes])
axes_flat = axes.flatten()
for i, col in enumerate(key_numeric[:n_rows * n_cols]):
    sns.boxplot(y=house[col], ax=axes_flat[i], color='lightblue')
    axes_flat[i].set_title(f'{col}')
plt.suptitle('关键数值特征箱线图（异常值检测）', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/eda/异常值检测箱线图.png', dpi=300)
plt.show()

# --- 4. 数据类型检查 ---
print("\n【数据类型检查】")
for col in house.columns:
    dtype = house[col].dtype
    non_null = house[col].notna().sum()
    sample_val = house[col].dropna().iloc[0] if non_null > 0 else "全部缺失"
    print(f"  {col:15s} | 类型: {str(dtype):10s} | 非空: {non_null:5d} | 示例值: {sample_val}")

print("\n>>> 继续任务4...")


# ============================================================
# 任务4：基础统计描述（40-55分钟）
# ============================================================
print("\n" + "=" * 60)
print("任务4：基础统计描述")
print("=" * 60)

# --- 1. 数值特征详细统计 ---
print("\n【数值特征详细统计】")
numeric_stats = house[numeric_cols].describe().T
numeric_stats['中位数'] = house[numeric_cols].median()
numeric_stats['偏度'] = house[numeric_cols].skew()
numeric_stats['峰度'] = house[numeric_cols].kurt()
print(numeric_stats)
print()

# --- 2. 类别特征分布 ---
print("【类别特征分布】")
category_cols = house.select_dtypes(include=['object']).columns.tolist()
for col in category_cols:
    print(f"\n--- {col} ---")
    vc = house[col].value_counts()
    print(f"唯一值数量：{house[col].nunique()}")
    print("前5个类别分布：")
    print(vc.head())
    print(f"前5类占比：{(vc.head().sum() / len(house) * 100):.1f}%")
print()

# --- 3. 目标变量分布可视化 ---
print("【目标变量分布可视化】")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 3a. 单价分布直方图
axes[0, 0].hist(house['单价（元/平米）'].dropna(), bins=50, color='steelblue', edgecolor='white', alpha=0.8)
axes[0, 0].set_title('单价分布')
axes[0, 0].set_xlabel('单价（元/平米）')
axes[0, 0].set_ylabel('频数')
axes[0, 0].axvline(house['单价（元/平米）'].median(), color='red', linestyle='--', label=f'中位数: {house["单价（元/平米）"].median():.0f}')
axes[0, 0].legend()

# 3b. 总价分布
axes[0, 1].hist(house['总价（万元）'].dropna(), bins=50, color='coral', edgecolor='white', alpha=0.8)
axes[0, 1].set_title('总价分布')
axes[0, 1].set_xlabel('总价（万元）')
axes[0, 1].set_ylabel('频数')

# 3c. 建筑面积分布
axes[0, 2].hist(house['建筑面积'].dropna(), bins=50, color='green', edgecolor='white', alpha=0.8)
axes[0, 2].set_title('建筑面积分布')
axes[0, 2].set_xlabel('面积（平方米）')
axes[0, 2].set_ylabel('频数')

# 3d. 楼层分布
if '楼层' in house.columns:
    floor_counts = house['楼层'].value_counts()
    floor_counts.plot(kind='bar', ax=axes[1, 0], color='teal')
    axes[1, 0].set_title('楼层分布')
    axes[1, 0].set_xlabel('楼层')
    axes[1, 0].set_ylabel('数量')
    for i, v in enumerate(floor_counts):
        axes[1, 0].text(i, v + 5, str(v), ha='center', fontsize=9)

# 3e. 朝向分布
if '朝向' in house.columns:
    orient_counts = house['朝向'].value_counts().head(8)
    orient_counts.plot(kind='bar', ax=axes[1, 1], color='orange')
    axes[1, 1].set_title('朝向分布（前8）')
    axes[1, 1].set_xlabel('朝向')
    axes[1, 1].set_ylabel('数量')
    axes[1, 1].tick_params(axis='x', rotation=45)

# 3f. 户型分布
if '户型' in house.columns:
    layout_counts = house['户型'].value_counts().head(8)
    layout_counts.plot(kind='bar', ax=axes[1, 2], color='purple')
    axes[1, 2].set_title('户型分布（前8）')
    axes[1, 2].set_xlabel('户型')
    axes[1, 2].set_ylabel('数量')
    axes[1, 2].tick_params(axis='x', rotation=45)

plt.suptitle('杭州学区房数据分布总览', fontsize=16)
plt.tight_layout()
plt.savefig('figures/eda/数据分布总览.png', dpi=300)
plt.show()

# --- 4. 关键数值特征相关性 ---
print("\n【总价与其他数值特征的相关系数】")
corr_with_price = house[numeric_cols].corr()['总价（万元）'].sort_values(ascending=False)
print(corr_with_price)

print("\n>>> 继续任务5...")


# ============================================================
# 任务5：初步数据清洗与基础特征提取（55-75分钟）
# ============================================================
print("\n" + "=" * 60)
print("任务5：初步数据清洗与基础特征提取")
print("=" * 60)

# --- 1. 处理缺失值 ---
print("\n【处理缺失值】")
# 朝向：类别型，用众数填补
house['朝向'] = house['朝向'].fillna(house['朝向'].mode()[0])
# 建筑年代：数值型，用中位数填补
house['建筑年代'] = pd.to_numeric(house['建筑年代'], errors='coerce')
house['建筑年代'] = house['建筑年代'].fillna(house['建筑年代'].median())
# 建筑面积：数值型，用中位数填补
house['建筑面积'] = pd.to_numeric(house['建筑面积'], errors='coerce')
house['建筑面积'] = house['建筑面积'].fillna(house['建筑面积'].median())

# 检查是否还有其他缺失值
remaining_missing = house.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]
if len(remaining_missing) > 0:
    print("仍有缺失值的字段：")
    print(remaining_missing)
    for col in remaining_missing.index:
        if house[col].dtype in ['float64', 'int64']:
            house[col] = house[col].fillna(house[col].median())
        else:
            house[col] = house[col].fillna(house[col].mode()[0])
print("缺失值处理完成")
print()

# --- 2. 删除标识字段 ---
print("【删除标识字段】")
house = house.drop(columns=['序号'], errors='ignore')
print("标识字段删除完成")
print()

# --- 3. 基础特征提取：从户型提取室数和厅数 ---
print("【提取户型特征】")
def extract_rooms(layout):
    """从户型文本中提取室数和厅数"""
    if pd.isna(layout):
        return pd.Series({'室': np.nan, '厅': np.nan})
    match = re.search(r'(\d+)室(\d+)厅', str(layout))
    if match:
        return pd.Series({'室': int(match.group(1)), '厅': int(match.group(2))})
    return pd.Series({'室': np.nan, '厅': np.nan})

room_info = house['户型'].apply(extract_rooms)
house['室'] = room_info['室']
house['厅'] = room_info['厅']
# 填充提取失败的记录
house['室'] = house['室'].fillna(house['室'].mode()[0])
house['厅'] = house['厅'].fillna(house['厅'].mode()[0])
print("户型特征提取完成")
print()

# --- 4. 从距离学校提取数值距离 ---
print("【提取距离特征】")
def extract_distance(dist_str):
    """从距离学校的文本描述中提取数值距离（单位：米）"""
    if pd.isna(dist_str):
        return np.nan
    dist_str = str(dist_str)
    # 匹配"Xkm"或"X公里"
    match_km = re.search(r'([\d.]+)\s*(?:公里|km)', dist_str, re.IGNORECASE)
    if match_km:
        return float(match_km.group(1)) * 1000
    # 匹配"X米"或"Xm"
    match_m = re.search(r'([\d.]+)\s*(?:米|m)', dist_str, re.IGNORECASE)
    if match_m:
        return float(match_m.group(1))
    # 仅匹配数字
    match_num = re.search(r'([\d.]+)', dist_str)
    if match_num:
        return float(match_num.group(1))
    return np.nan

house['距离学校_米'] = house['距离学校'].apply(extract_distance)
# 用中位数填补提取失败的记录
house['距离学校_米'] = house['距离学校_米'].fillna(house['距离学校_米'].median())
print("距离特征提取完成")
print()

# --- 5. 构造房龄 ---
print("【构造房龄特征】")
house['房龄'] = 2026 - house['建筑年代']
# 处理不合理的房龄
house.loc[house['房龄'] < 0, '房龄'] = 0
house.loc[house['房龄'] > 100, '房龄'] = house['房龄'].median()
print("房龄特征构造完成")
print()

# --- 6. 楼层编码 ---
print("【楼层编码】")
floor_map = {'低层': 1, '中层': 2, '高层': 3}
house['楼层_数值'] = house['楼层'].map(floor_map)
# 处理映射失败的值
house['楼层_数值'] = house['楼层_数值'].fillna(house['楼层_数值'].mode()[0])
print("楼层编码完成")
print()

# --- 7. 是/否二值特征转0/1 ---
print("【二值特征转换】")
binary_cols = ['普通', '小班教学', '区重点', '体育类', '艺术类',
               '语文类', '双语', '科技类', '名校附属', '市重点', '外语类']
for col in binary_cols:
    if col in house.columns:
        house[col] = house[col].map({'是': 1, '否': 0})
        house[col] = house[col].fillna(0).astype(int)
print("二值特征转换完成")
print()

# --- 8. 验证清洗结果 ---
print("=" * 50)
print("清洗结果验证")
print("=" * 50)
print(f"清洗后数据形状：{house.shape}")
print(f"清洗后缺失值总数：{house.isnull().sum().sum()}")
print()
print("新增特征统计：")
new_features = ['室', '厅', '距离学校_米', '房龄', '楼层_数值']
available_new = [f for f in new_features if f in house.columns]
print(house[available_new].describe())
print()
print("二值特征分布：")
available_binary = [col for col in binary_cols if col in house.columns]
for col in available_binary:
    ones = house[col].sum()
    print(f"  {col:10s}：是={int(ones)} ({ones/len(house)*100:.1f}%)，否={len(house)-int(ones)} ({(len(house)-ones)/len(house)*100:.1f}%)")


# ============================================================
# 保存清洗后的数据
# ============================================================
house.to_csv(DATA_DIR / "processed" / "hz_house_clean.csv", index=False, encoding='utf-8-sig')
print("\n✓ 清洗后数据已保存至: data/processed/hz_house_clean.csv")


# ============================================================
# 阶段检查清单
# ============================================================
print("\n" + "=" * 60)
print("阶段一检查清单")
print("=" * 60)
checklist = """
序号  检查项目                                    通过
1     能说出目标变量和任务类型                    [ ]
2     知道数据规模                                [ ]
3     了解数据类型分布                            [ ]
4     完成缺失值统计                              [ ]
5     绘制了箱线图                                [ ]
6     完成数值特征描述                            [ ]
7     完成类别特征分布                            [ ]
8     缺失值全部处理 (isnull().sum().sum()==0)    [ ]
9     删除了标识字段                              [ ]
10    提取了室和厅                                [ ]
11    提取了距离数值                              [ ]
12    构造了房龄                                  [ ]
13    楼层已编码                                  [ ]
14    二值特征已转换                              [ ]
15    数据可正常运行                              [ ]
"""
print(checklist)
print("✓ 阶段一完成！")
