import plotly.express as px
import pandas as pd

# 创建一个简单的数据集
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 11, 12, 13, 14],
    'z': [5, 4, 3, 2, 1]
})

# 创建一个简单的3D散点图
fig = px.scatter_3d(df, x='x', y='y', z='z')

# 保存为HTML文件
fig.write_html('test_plot.html')
print("测试图表已保存为 test_plot.html")