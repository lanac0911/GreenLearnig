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


def imshow(title, image = None, size = 6):
    if image.any():
      w, h = image.shape[0], image.shape[1]
      aspect_ratio = w/h
      plt.figure(figsize=(size * aspect_ratio,size))
      plt.imshow(image)
      plt.title(title)
      plt.show()
    else:
      print("Image not found")

def myModel(x, getK=1):
    x1 = PixelHop_Unit(x, dilate=1, pad='reflect', num_AC_kernels=9, weight_name='pixelhop1.pkl', getK=getK)

    x2 = PixelHop_Unit(x1, dilate=2, pad='reflect', num_AC_kernels=25, weight_name='pixelhop2.pkl', getK=getK)
    x2 = AvgPooling(x2)

    x3 = PixelHop_Unit(x2, dilate=2, pad='reflect', num_AC_kernels=35, weight_name='pixelhop3.pkl', getK=getK)
    x3 = AvgPooling(x3)

    x4 = PixelHop_Uanit(x3, dilate=2, pad='reflect', num_AC_kernels=55, weight_name='pixelhop4.pkl', getK=getK)




    x2 = myResize(x2, x.shape[1], x.shape[2])
    x3 = myResize(x3, x.shape[1], x.shape[2])
    x4 = myResize(x4, x.shape[1], x.shape[2])
    return np.concatenate((x1,x2,x3,x4), axis=3)
if False:
    print("in test img")
    x = cv2.imread('../data/test.jpg')
    x = x.reshape(1, x.shape[0], x.shape[1], -1)
    feature = myModel(x, getK=1)
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

            # 根据文件名规则判断该图像属于训练集还是测试集
            if "MHE_" in filename:
                label = int(filename.split("_")[1].split(".")[0])  # 解析文件名中的数字作为标签
                total_images.append(image)
                # if label <= (1681):  # 将前80%的图像用于训练
                #     train_images.append(image)
                # else:
                #     test_images.append(image)
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


    # # 可以进一步处理图像和标签，例如将其转换为NumPy数组
    # train_labels = label[:1682]
    # test_labels = label[-421:]
    # # 将标签转换为NumPy数组
    # train_labels = np.array(train_labels)
    # test_labels = np.array(test_labels)

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


    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)  # Create a 2x3 grid of subplots, and select the i-th subplot
    #     plt.imshow(train_images[i], cmap='gray')  # Display the image in grayscale
    #     plt.title(f"Label: {train_labels[i]}")  # Set the title with the label

    # plt.tight_layout()  # Adjust subplot layout for better spacing
    # plt.show()


    test_images=train_images[:N_test]
    test_labels=train_labels[:N_test]
    
    train_feature = PixelHop_Unit(train_images, dilate=1, pad='reflect', num_AC_kernels=5, weight_name='pixelhop1_mnist.pkl', getK=1)
    train_feature = block_reduce(train_feature, (1, 4, 4, 1), np.mean).reshape(N_train,-1)
    train_feature_reduce = LAG_Unit(train_feature,train_labels=train_labels, class_list=class_list,
                             SAVE=SAVE,num_clusters=50,alpha=5,Train=True)
    
    test_feature = PixelHop_Unit(test_images, dilate=1, pad='reflect', num_AC_kernels=5, weight_name='pixelhop1_mnist.pkl', getK=0)
    test_feature = block_reduce(test_feature, (1, 4, 4, 1), np.mean).reshape(N_test,-1)
    test_feature_reduce = LAG_Unit(test_feature,train_labels=None, class_list=class_list,
                         SAVE=SAVE,num_clusters=50,alpha=5,Train=False)
    
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn import preprocessing
    scaler=preprocessing.StandardScaler()
    feature = scaler.fit_transform(train_feature_reduce)
    feature_test = scaler.transform(test_feature_reduce)     
   
    clf=SVC().fit(feature, train_labels) 
##        clf=RandomForestClassifier(n_estimators=500,max_depth=5).fit(train_f, train_labels) 
    print('***** Train ACC:', accuracy_score(train_labels,clf.predict(feature)))
    print('***** Test ACC:', accuracy_score(test_labels,clf.predict(feature_test)))