import pandas as pd
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)
sys.path.append(project_root)
from parameter import parameter

target = 'MHE'

drive_path = f"../csv/Electricity_{target}.csv"
label_txt_fileName = 'lable'
slice_len = parameter.split_size
# 从CSV文件中读取数据
df = pd.read_csv(drive_path)

# 初始化一个空列表，用于存储每1000行的P列的平均值
label = []

# 每1000行分组计算平均值
for i in range(0, len(df) , slice_len):
    group = df.iloc[i:i+slice_len]  # 获取每250行数据
    average_p = group["P"].mean()  # 计算P列的平均值
    label.append(average_p)  # 将平均值添加到label列表中

# 打印或使用label列表中的平均值
print("Length = ", len(label))

# Open the file in write mode
with open(f'../myData/Lable_Text/{target}_{slice_len}.txt', 'w') as f:
    # Write the entire list to the file as a string
    f.write(str(label))

print("處理Label完成")
