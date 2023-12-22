import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re


# IMG_DIR = '../../myData/Cwt_Images/cwt_MHE_result'

# def map_label_to_class(label):
#     class_ranges = [(0, 10), (10, 30), (30, 50), (50,70), (70,90), (90, 110), (110, 130), (130, 150), (150, 180), (180, 200), ]
#     for i, (start, end) in enumerate(class_ranges):
#         if start <= label < end :
#             return i
#     return len(class_ranges)

# def generate_class_list(train_labels, test_labels):
#     class_list = np.unique(np.concatenate(train_labels, test_labels))
#     print("mped class_list=",class_list)
#     return class_list

# def generate_maped_labels(labels):
#     maped_labels = [map_label_to_class(label) for label in labels]
#     return maped_labels

def load_lable ():
    # 讀取 LABLE TXT 檔
    label_file = "MHE_250.txt"
    with open(label_file, "r") as f:
        content = f.read()


           # 移除中括號
    content = content.replace("[", "").replace("]", "")

    # 使用正則表達式過濾掉非數字和逗號之外的字符
    content = re.sub(r"[^0-9,\.]", "", content)

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
    print("ori_labels len=", max(ori_labels), len(ori_labels), ori_labels)

    MAX = 3500
    STRIDE = 100
    # bins = np.arange(0, MAX+1, STRIDE)
    hist, bin_edges = np.histogram(ori_labels, bins='auto')
    # mapped_labels = np.digitize(ori_labels, bins)
    mapped_labels = np.digitize(ori_labels, bin_edges)
    print("ori_labels len=", len(ori_labels), mapped_labels)

    # bins_0_to_100 = np.arange(0, 101, 2)
    # bins_101_to_150 = np.arange(101, 151, 30)
    # bins_151_and_beyond = np.arange(151, 300, 50)

    # 將原始標籤映射到不同的組
    # mapped_labels = np.zeros_like(ori_labels)

    # # 將 0 到 100 的範圍內每2分一組
    # mask_0_to_100 = (ori_labels <= 100)
    # mapped_labels[mask_0_to_100] = np.digitize(ori_labels[mask_0_to_100], bins_0_to_100)

    # # 將 101 到 150 的範圍內每30分一組
    # mask_101_to_150 = (ori_labels > 100) & (ori_labels <= 150)
    # mapped_labels[mask_101_to_150] = np.digitize(ori_labels[mask_101_to_150], bins_101_to_150)

    # # 將 151 後的範圍內每50分一組
    # mask_151_and_beyond = (ori_labels > 150)
    # mapped_labels[mask_151_and_beyond] = np.digitize(ori_labels[mask_151_and_beyond], bins_151_and_beyond)

    #     # 計算不同範圍的組數
    # num_groups_0_to_100 = len(bins_0_to_100) - 1
    # num_groups_101_to_150 = len(bins_101_to_150) - 1
    # num_groups_151_and_beyond = len(bins_151_and_beyond) - 1
    # # 總共的組數
    # total_num_groups = num_groups_0_to_100 + num_groups_101_to_150 + num_groups_151_and_beyond
    # print("Total number of groups:", total_num_groups)
    class_list = np.arange(1, len(bin_edges)+1)
    print("Mapped Labels:", mapped_labels)
    print("Mapped Labels Len:", len(mapped_labels))

    IMG_DIR = "../myData/Cwt_Images/cwt_MHE_result"
    # IMG_DIR = "../CWT result/CWT_MHE_result"
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
