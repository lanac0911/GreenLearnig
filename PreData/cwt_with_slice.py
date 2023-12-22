#!/usr/bin/env python
# coding: utf-8
import sys
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt


current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_path)
sys.path.append(project_root)
from parameter import parameter


print("處理轉換 size=", parameter.split_size)

target = 'MHE'
drive_path = "../myData/Cwt_Images/"
sliced_len = parameter.split_size

# 读取CSV文件
data = pd.read_csv("../csv/" + f"Electricity_{target}.csv")

# 将数据切割成1000行一次的片段
sliced_data = [data[i:i+sliced_len] for i in range(0, len(data), sliced_len)]

# --------------------------
# 获取当前工作目录
# current_directory = os.getcwd()
# 拼接要检查的目录路径
# cwt_result_directory = os.path.join(current_directory, 'cwt_result')
# 检查目录是否存在，如果不存在则创建它
# if not os.path.exists(cwt_result_directory):
    # os.makedirs(cwt_result_directory)

# 创建一个目录来存储图像
image_dir = drive_path + f'cwt_{target}_result'  # 设置图像目录
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    print("創建資料夾",  drive_path + f'cwt_{target}_result')
#---------------------


# 选择母小波函数（这里选择 Morlet 小波）
wavelet = 'morl'
level = 5
feature = 'P'

print("進入回圈，開始生成圖片...")
# 循环处理每个数据片段
for idx, chunk in enumerate(sliced_data):
    # 获取所需的列数据，假设您需要处理的是前三列
    current_signal = chunk[feature].values
    # I = chunk['I'].values
    # f = chunk['f'].values

    # 设置尺度参数
    scales = np.arange(1, 128)  # 这里选择了一个简单的尺度范围

    # 执行CWT转换
    coeffs, freqs = pywt.cwt(current_signal, scales, wavelet, 1)

# ------------------

    # 创建图像
    plt.figure(figsize=(6, 6))
    plt.imshow(abs(coeffs), aspect='auto', cmap='jet', origin='lower', extent=[0, len(data), 1, 31])
    plt.axis('off')

    # 保存图像
    # image_filename = f'cwt_result/image_{idx}.png'
    image_filename = os.path.join(image_dir, f'{target}_{idx}.png')
    # plt.savefig(f'/content/drive/MyDrive/GL/CWT/BME_CWT_result.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)
    # 关闭图像
    plt.close()
    
    if(idx % 50 == 0): print("已處理", idx, "圖片")


print("图像处理完成")
plt.show()

txt_filename = os.path.join(image_dir, f'{target}info.txt')
# 打开一个文本文件（如果文件不存在，将创建一个新文件）
with open(txt_filename, "w") as file:
    # 在文件中写入文本内容
    file.write(f"文件：Electricity_{target}.csv\n")
    file.write(f"擷取特徵：{feature}\n")
    file.write(f"筆數{len(sliced_data)}\n")

# 文件会在退出"with"代码块时自动关闭

