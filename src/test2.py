# Alex
# yifanwang0916@outlook.com
# 2019.09.25


# 使用 python 3.7
import numpy as np 
import cv2
import time

from framework.layer import *
from framework.utli import *
from framework.pixelhop import *
from framework.data import *
from framework.LAG import LAG_Unit

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from skimage.measure import block_reduce

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os

if True:
    SAVE={}

    # 文件夹路径
    img_dir = "../CWT result/cwt_MHE_result"

    # 初始化用于存储图像数据和标签的列表
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    total_images = []

    # 遍历图像文件夹中的图像
    for filename in os.listdir(img_dir):
        if filename.endswith(".png"):  # 仅处理.jpg格式的图像文件
            image_path = os.path.join(img_dir, filename)
            image = Image.open(image_path)  # 打开图像文件
            if "MHE_" in filename:
                total_images.append(image)
    # ---------------------------------------------------
    drive_path = "../csv/Electricity_BME.csv"
    slice_len = 500
    # 从CSV文件中读取数据
    df = pd.read_csv(drive_path)

    # 初始化一个空列表，用于存储每1000行的P列的平均值
    label = []

    # 每1000行分组计算平均值
    for i in range(0, len(df), slice_len):
        group = df.iloc[i:i+slice_len]  # 获取每1000行数据
        average_p = group["P"].mean()  # 计算P列的平均值
        label.append(average_p)  # 将平均值添加到label列 表中

    # 打印或使用label列表中的平均值
    print("len.label",len(label))


    # 将图像转换为NumPy数组
    total_images = [np.array(image) for image in total_images]
    # 归一化图像数据（将像素值缩放到0到1之间）
    total_images = np.array(total_images) / 255.0

    # 将数据划分为训练集和测试集
    train_images, test_images, train_labels, test_labels = train_test_split(
        total_images, labels, test_size=0.2, random_state=42
    )

    # 打印示例图像和标签
    print("Train Images:", len(train_images))
    print("Train Labels:", len(train_labels))
    print("Test Images:", len(test_images))
    print("Test Labels:", len(test_labels))

#-------------------------
    print("after------------------------------------\n")

    # 顯示資料集的形狀
    print("Initial shape or dimensions of x_train", str(train_images.shape))
    print("Initial shape or dimensions of x_test", str(test_images.shape))
    print("Initial shape or dimensions of y_train", str(train_labels.shape))
    print("Initial shape or dimensions of y_test", str(test_labels.shape))
    print('\n')

    # 顯示每組資料的sample數
    print ("Number of samples in our training data: " + str(len(train_images)))
    print ("Number of labels in our training data: " + str(len(train_labels)))
    print ("Number of samples in our test data: " + str(len(test_images)))
    print ("Number of labels in our test data: " + str(len(test_labels)))

    print("------------------------------------\n")
#-------------------------

