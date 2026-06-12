# -*- coding: utf-8 -*-
"""
运行所有阶段的代码
"""
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，不显示图表

import sys
import os

# 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

print("=" * 60)
print("开始运行实验6：杭州学区房数据分析")
print("=" * 60)

# 运行阶段一
print("\n" + "=" * 60)
print("运行阶段一：数据获取、探索分析与特征工程基础")
print("=" * 60)
exec(open("notebooks/阶段一_完整代码.py", encoding="utf-8").read())

# 运行阶段二
print("\n" + "=" * 60)
print("运行阶段二：深度特征工程与多模型训练")
print("=" * 60)
exec(open("notebooks/阶段二_完整代码.py", encoding="utf-8").read())

# 运行阶段三
print("\n" + "=" * 60)
print("运行阶段三：迭代优化与模型评估")
print("=" * 60)
exec(open("notebooks/阶段三_完整代码.py", encoding="utf-8").read())

print("\n" + "=" * 60)
print("所有阶段运行完成！")
print("=" * 60)
