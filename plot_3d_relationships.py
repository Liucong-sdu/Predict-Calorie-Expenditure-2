import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# 读取CSV文件
df = pd.read_csv('train.csv')

# 目标变量
target = 'Calories'

# 只选择数值型特征变量（排除id、目标变量和非数值型列）
numeric_features = []
for col in df.columns:
    if col != 'id' and col != target and pd.api.types.is_numeric_dtype(df[col]):
        numeric_features.append(col)

print(f"将使用以下数值型特征: {numeric_features}")

# 创建输出目录
output_dir = 'plots_3d'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 对每两个特征变量和目标变量绘制3D图
for i in range(len(numeric_features)):
    for j in range(i+1, len(numeric_features)):
        feature1 = numeric_features[i]
        feature2 = numeric_features[j]
        
        # 创建3D图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制散点图
        scatter = ax.scatter(df[feature1], df[feature2], df[target], 
                   c=df[target], cmap='viridis', s=50, alpha=0.6)
        
        # 添加标签和标题
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_zlabel(target)
        ax.set_title(f'3D关系图: {feature1} vs {feature2} vs {target}')
        
        # 添加颜色条
        fig.colorbar(scatter, ax=ax, label=target)
        
        # 保存图像
        plt.savefig(f'{output_dir}/{feature1}_vs_{feature2}_vs_{target}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {feature1} vs {feature2} vs {target}")

print(f"所有3D图像已保存到 {output_dir} 目录")