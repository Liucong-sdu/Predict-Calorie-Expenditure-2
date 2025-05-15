import pandas as pd
import plotly.express as px
import os
import traceback
import numpy as np
from datetime import datetime
import gc

# 创建带时间戳的输出目录，避免覆盖之前的结果
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'interactive_plots_{timestamp}'  # 使用时间戳创建唯一目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建日志文件
import logging
logging.basicConfig(
    filename=f'{output_dir}/error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 安全的文件名处理函数
def safe_filename(name):
    # 替换所有不安全的字符
    unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    safe_name = str(name)
    for char in unsafe_chars:
        safe_name = safe_name.replace(char, '_')
    return safe_name

try:
    # 读取CSV文件
    print("正在读取数据...")
    df = pd.read_csv('train_processed.csv')
    print(f"成功读取数据，共 {len(df)} 行")
    
    # 数据采样 - 使用10000个随机样本点
    sample_size = 10000  # 可以调整这个数字
    if len(df) > sample_size:
        print(f"数据量较大，正在进行随机采样 ({sample_size} 个样本)...")
        df = df.sample(n=sample_size, random_state=42)
        print(f"采样后数据量: {len(df)} 行")
    
    # 检查数据是否有问题
    print("检查数据...")
    print(f"数据类型:\n{df.dtypes}")
    print(f"是否有缺失值: {df.isnull().sum().sum()}")
    
    # 目标变量
    target = 'Calories'
    
    # 只选择数值型特征变量（排除id、目标变量和非数值型列）
    numeric_features = []
    for col in df.columns:
        if col != 'id' and col != target and pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
    
    print(f"将使用以下数值型特征: {numeric_features}")
    
    # 处理分类变量 - 检查是否有性别列
    has_sex_column = False
    if 'Sex' in df.columns:
        has_sex_column = True
        # 将Sex转换为数值特征
        df['Sex_numeric'] = df['Sex'].map({'male': 0, 'female': 1})
        # 不将Sex_numeric添加到特征列表，因为我们将按性别分开绘图
        print("已将'Sex'转换为数值特征'Sex_numeric'")
    
    # 计算特征与目标变量的相关性
    correlations = {}
    for feature in numeric_features:
        corr = df[feature].corr(df[target])
        correlations[feature] = abs(corr)  # 使用绝对值，关注强度而非方向
    
    # 按相关性排序特征
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    print("\n特征与目标变量的相关性（按绝对值排序）:")
    for feature, corr in sorted_features:
        print(f"{feature}: {corr:.4f}")
    
    # 使用所有特征而不是只用前5个
    all_features = [feature for feature, _ in sorted_features]
    print(f"\n将使用所有特征: {all_features}")
    
    # 对每两个特征变量和目标变量绘制交互式3D图
    plot_count = 0
    
    # 如果有性别列，则按性别分开绘图
    if has_sex_column:
        sex_groups = [('male', '男性'), ('female', '女性')]
    else:
        sex_groups = [(None, '所有数据')]
    
    # 计算总图表数量并显示进度
    total_plots = 0
    for sex_value, sex_label in sex_groups:
        if sex_value is not None:
            sex_df = df[df['Sex'] == sex_value]
        else:
            sex_df = df
        
        feature_pairs = []
        for i in range(len(all_features)):
            for j in range(i+1, len(all_features)):
                feature_pairs.append((all_features[i], all_features[j]))
        
        total_plots += len(feature_pairs)
    
    print(f"将生成总共 {total_plots} 个图表...")
    current_plot = 0
    
    for sex_value, sex_label in sex_groups:
        # 如果有性别值，则筛选数据
        if sex_value is not None:
            sex_df = df[df['Sex'] == sex_value]
            print(f"为{sex_label}生成图表，数据量: {len(sex_df)} 行")
        else:
            sex_df = df
        
        for i in range(len(all_features)):
            for j in range(i+1, len(all_features)):
                feature1 = all_features[i]
                feature2 = all_features[j]
                
                current_plot += 1
                print(f"正在生成 ({sex_label}): {feature1} vs {feature2} vs {target} [{current_plot}/{total_plots}]")
                
                try:
                    # 创建交互式3D散点图 - 直接设置点的大小
                    fig = px.scatter_3d(
                        sex_df, 
                        x=feature1, 
                        y=feature2, 
                        z=target,
                        color=target,
                        opacity=0.7,
                        color_continuous_scale='viridis',
                        title=f'{sex_label} - 交互式3D关系图: {feature1} vs {feature2} vs {target}',
                        # 直接在这里设置点的大小
                        size_max=2  # 最大点大小
                    )
                    
                    # 添加趋势面
                    try:
                        # 创建网格数据
                        x_range = np.linspace(sex_df[feature1].min(), sex_df[feature1].max(), 20)
                        y_range = np.linspace(sex_df[feature2].min(), sex_df[feature2].max(), 20)
                        x_grid, y_grid = np.meshgrid(x_range, y_range)
                        
                        # 拟合平面 - 添加异常值处理
                        from sklearn.linear_model import LinearRegression, RANSACRegressor
                        X = sex_df[[feature1, feature2]]
                        y = sex_df[target]
                        
                        # 尝试使用RANSAC算法处理异常值
                        try:
                            min_samples = min(50, len(X) // 10)  # 动态调整样本数
                            if min_samples < 3:  # RANSAC需要至少3个样本
                                min_samples = 3
                                
                            ransac = RANSACRegressor(
                                LinearRegression(), 
                                max_trials=100,
                                min_samples=min_samples,
                                loss='absolute_error',
                                random_state=42
                            )
                            ransac.fit(X, y)
                            model = ransac
                        except Exception as e:
                            print(f"RANSAC拟合失败，回退到普通线性回归: {str(e)}")
                            model = LinearRegression().fit(X, y)
                        
                        # 预测z值
                        z_grid = model.predict(np.column_stack([x_grid.flatten(), y_grid.flatten()]))
                        z_grid = z_grid.reshape(x_grid.shape)
                        
                        # 添加趋势面
                        fig.add_surface(x=x_grid, y=y_grid, z=z_grid, opacity=0.5, 
                                       colorscale='Blues', showscale=False,
                                       name='趋势面')
                        
                        # 添加相关系数信息
                        corr1 = sex_df[feature1].corr(sex_df[target])
                        corr2 = sex_df[feature2].corr(sex_df[target])
                        r_squared = model.score(X, y)
                        
                        fig.update_layout(
                            annotations=[
                                dict(
                                    x=0.02,
                                    y=0.98,
                                    xref='paper',
                                    yref='paper',
                                    text=f'{sex_label}<br>相关系数:<br>{feature1} vs {target}: {corr1:.4f}<br>{feature2} vs {target}: {corr2:.4f}<br>R²: {r_squared:.4f}',
                                    showarrow=False,
                                    bgcolor='rgba(255,255,255,0.8)',
                                    bordercolor='black',
                                    borderwidth=1,
                                    font=dict(size=12)
                                )
                            ]
                        )
                    except Exception as e:
                        print(f"添加趋势面时出错: {str(e)}")
                        logging.error(f"添加趋势面时出错 ({sex_label} - {feature1} vs {feature2}): {str(e)}")
                        logging.error(traceback.format_exc())
                    
                    # 删除这一行，因为它会导致错误
                    # fig.update_traces(marker=dict(size=2))
                    
                    # 如果需要单独更新散点图的点大小，可以使用索引
                    # 只对散点图设置marker属性，不影响趋势面
                    if len(fig.data) > 0 and hasattr(fig.data[0], 'marker'):
                        fig.data[0].marker.size = 2
                    
                    fig.update_layout(
                        scene = dict(
                            xaxis_title=feature1,
                            yaxis_title=feature2,
                            zaxis_title=target,
                        ),
                        width=900,
                        height=700,
                        margin=dict(l=0, r=0, b=0, t=30)
                    )
                    
                    # 保存为HTML文件（可交互）
                    safe_feature1_name = safe_filename(feature1)
                    safe_feature2_name = safe_filename(feature2)
                    safe_target_name = safe_filename(target)
                    safe_sex_name = safe_filename(sex_label)
                    
                    output_file = f'{output_dir}/{safe_sex_name}_{safe_feature1_name}_vs_{safe_feature2_name}_vs_{safe_target_name}.html'
                    fig.write_html(
                        output_file,
                        include_plotlyjs=True,  # 包含plotly.js，不依赖CDN
                        full_html=True
                    )
                    
                    print(f"已保存: {output_file}")
                    plot_count += 1
                    
                    # 每10个图表清理一次内存
                    if plot_count % 10 == 0:
                        gc.collect()  # 强制垃圾回收
                        print("已清理内存...")
                
                except Exception as e:
                    error_msg = f"生成图表时出错 ({sex_label} - {feature1} vs {feature2}): {str(e)}"
                    print(error_msg)
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
    
    # 创建索引页面
    index_file = f'{output_dir}/index.html'
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n<head>\n')
        f.write('<meta charset="utf-8">\n')
        f.write('<title>卡路里消耗预测 - 3D交互式可视化</title>\n')
        f.write('<style>\n')
        f.write('body { font-family: Arial, sans-serif; margin: 20px; }\n')
        f.write('h1 { color: #2c3e50; }\n')
        f.write('h2 { color: #3498db; margin-top: 30px; }\n')
        f.write('ul { list-style-type: none; padding: 0; }\n')
        f.write('li { margin: 10px 0; padding: 10px; background-color: #f5f5f5; border-radius: 5px; }\n')
        f.write('a { color: #3498db; text-decoration: none; }\n')
        f.write('a:hover { text-decoration: underline; }\n')
        f.write('.info { color: #7f8c8d; font-size: 0.9em; margin-top: 5px; }\n')
        f.write('table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n')
        f.write('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n')
        f.write('th { background-color: #f2f2f2; }\n')
        f.write('tr:nth-child(even) { background-color: #f9f9f9; }\n')
        f.write('.tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }\n')
        f.write('.tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }\n')
        f.write('.tab button:hover { background-color: #ddd; }\n')
        f.write('.tab button.active { background-color: #ccc; }\n')
        f.write('.tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }\n')
        f.write('</style>\n')
        f.write('<script>\n')
        f.write('function openTab(evt, tabName) {\n')
        f.write('  var i, tabcontent, tablinks;\n')
        f.write('  tabcontent = document.getElementsByClassName("tabcontent");\n')
        f.write('  for (i = 0; i < tabcontent.length; i++) {\n')
        f.write('    tabcontent[i].style.display = "none";\n')
        f.write('  }\n')
        f.write('  tablinks = document.getElementsByClassName("tablinks");\n')
        f.write('  for (i = 0; i < tablinks.length; i++) {\n')
        f.write('    tablinks[i].className = tablinks[i].className.replace(" active", "");\n')
        f.write('  }\n')
        f.write('  document.getElementById(tabName).style.display = "block";\n')
        f.write('  evt.currentTarget.className += " active";\n')
        f.write('}\n')
        f.write('</script>\n')
        f.write('</head>\n<body>\n')
        f.write('<h1>卡路里消耗预测 - 3D交互式可视化</h1>\n')
        f.write(f'<p>数据集大小: {len(df)} 个样本 (从原始数据集中随机采样)</p>\n')
        f.write(f'<p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
        
        # 添加相关性表格
        f.write('<h2>特征与目标变量的相关性</h2>\n')
        f.write('<table>\n')
        f.write('<tr><th>特征</th><th>与卡路里的相关性</th></tr>\n')
        for feature, corr in sorted_features:
            f.write(f'<tr><td>{feature}</td><td>{corr:.4f}</td></tr>\n')
        f.write('</table>\n')
        
        # 创建选项卡
        f.write('<div class="tab">\n')
        for _, sex_label in sex_groups:
            tab_id = sex_label.replace(' ', '_')
            f.write(f'<button class="tablinks" onclick="openTab(event, \'{tab_id}\')">{sex_label}</button>\n')
        f.write('</div>\n')
        
        # 为每个性别创建内容区域
        for sex_idx, (sex_value, sex_label) in enumerate(sex_groups):
            tab_id = sex_label.replace(' ', '_')
            # 第一个选项卡默认显示
            display_style = 'block' if sex_idx == 0 else 'none'
            f.write(f'<div id="{tab_id}" class="tabcontent" style="display: {display_style};">\n')
            f.write(f'<h2>{sex_label}数据的3D交互式图表</h2>\n')
            f.write('<ul>\n')
            
            for i in range(len(all_features)):
                for j in range(i+1, len(all_features)):
                    feature1 = all_features[i]
                    feature2 = all_features[j]
                    
                    # 使用与保存文件时相同的安全文件名
                    safe_feature1_name = safe_filename(feature1)
                    safe_feature2_name = safe_filename(feature2)
                    safe_target_name = safe_filename(target)
                    safe_sex_name = safe_filename(sex_label)
                    
                    filename = f'{safe_sex_name}_{safe_feature1_name}_vs_{safe_feature2_name}_vs_{safe_target_name}.html'
                    
                    # 获取相关系数
                    corr1 = correlations.get(feature1, 0)
                    corr2 = correlations.get(feature2, 0)
                    
                    f.write(f'<li><a href="{filename}" target="_blank">{feature1} vs {feature2} vs {target}</a>\n')
                    f.write(f'<div class="info">相关性: {feature1} ({corr1:.4f}), {feature2} ({corr2:.4f})</div></li>\n')
            
            f.write('</ul>\n')
            f.write('</div>\n')
        
        f.write('<p>注意: 点击链接将在新标签页中打开交互式3D图表。您可以旋转、缩放和平移图表以查看不同角度。</p>\n')
        f.write('<script>document.getElementsByClassName("tablinks")[0].className += " active";</script>\n')
        f.write('</body>\n</html>\n')
    
    print(f"已生成 {plot_count} 个交互式3D图像，保存到 {output_dir} 目录")
    print(f"请打开索引页面查看结果: {index_file}")

except Exception as e:
    print(f"发生错误: {str(e)}")
    traceback.print_exc()
    logging.error(f"主程序错误: {str(e)}")
    logging.error(traceback.format_exc())