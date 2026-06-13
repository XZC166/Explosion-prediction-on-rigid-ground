import pandas as pd

# 读取Excel文件
df = pd.read_excel('./cases_info.xlsx')

# 保存为CSV文件
df.to_csv('./case_info.csv', index=False, encoding='utf-8-sig')

print("转换完成！文件已保存为 'data/case_info.csv'")