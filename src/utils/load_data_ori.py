import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_DIR = '../../myData_Cwt_Images/'

def map_label_to_class(label):
    class_ranges = [(0, 10), (10, 30), (30, 50), (50,70), (70,90), (90, 110), (110, 130), (130, 150), (150, 180), (180, 200), ]
    for i, (start, end) in enumerate(class_ranges):
        if start <= label < end :
            return i
    return len(class_ranges)

def generate_class_list(train_labels, test_labels):
    class_list = np.unique(np.concatenate(train_labels, test_labels))
    print("mped class_list=",class_list)
    return class_list

def generate_maped_labels(labels):
    maped_labels = [map_label_to_class(label) for label in labels]
    return maped_labels

def load_lable ():
    # 讀取 LABLE TXT 檔
    label_file = "lable.txt"
    with open(label_file, "r") as f:
        content = f.read()
    labels = [float(x) for x in content.split(',')]
    return np.array(labels)


# 讀取並處理圖片
def load_and_preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.resize((32, 32))  # 舉例：將圖片大小調整為 224x224
    img_array = np.array(img) / 255.0  # 正規化到 [0, 1]
    return img_array



def load_data():
    ori_labels = load_lable()  
    bins = np.arange(0, 301, 30)
    mapped_labels = np.digitize(ori_labels, bins)
    class_list = np.arange(1, 11)
    print("Mapped Labels:", mapped_labels)

    IMG_DIR = "../CWT result/CWT_MHE_result"
    # 獲取圖片文件列表
    img_files = [os.path.join(IMG_DIR, file) for file in os.listdir(IMG_DIR) if file.endswith(('png'))]


    # 讀取所有圖片並堆疊成數組
    data = [load_and_preprocess_image(file) for file in img_files]
    data = np.stack(data)

    # 假設你有一個對應的標籤列表 labels，它的長度應該和圖片數目相同

    # 使用 train_test_split 函數拆分訓練集和測試集
    train_images, test_images, train_labels, test_label = train_test_split(data, mapped_labels, test_size=0.2, random_state=42)
    
    train_images = np.dot(train_images[:, :, :, :3], [0.299, 0.587, 0.114])
    train_images = np.expand_dims(train_images, axis=-1)   
    test_images = np.dot(test_images[:, :, :, :3], [0.299, 0.587, 0.114])
    test_images = np.expand_dims(test_images, axis=-1)

    return train_images, train_labels, test_images, test_label, class_list
