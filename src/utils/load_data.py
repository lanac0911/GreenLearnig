import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

def map_label_to_class(label):
    class_ranges = [(0, 10), (10, 30), (30, 50), (50,70), (70,90), (90, 110), (110, 130), (130, 150), (150, 180), (180, 200), ]
    for i, (start, end) in enumerate(class_ranges):
        if start <= label < end :
            return i
    return len(class_ranges)

def generate_class_list(train_labels, test_labels):
    # class_list = [map_label_to_class(label) for label in train_labels]
    class_list = np.unique(np.concatenate(train_labels, test_labels))

    # file_path = 'C:\Users\awinl\OneDrive\桌面\Green Learing\Pixelhop-master\src\label_int.txt'
    # with open(file_path, 'r') as file:
    #     data = [float(value) for value in file.readline().split(',')]
    # max_value = np.max(data) # 找出最大值

    # max_num = int(max_value) + 2
    # max_num = 276 + 2
    # class_list = np.arange(max_num)
    print("mped class_list=",class_list)
    return class_list

def generate_maped_labels(labels):
    maped_labels = [map_label_to_class(label) for label in labels]
    return maped_labels

def load_lable ():
    # # 讀取圖片資料夾
    # img_files = os.listdir(IMG_DIR)

    # 讀取 LABLE TXT 檔
    label_file = "lable.txt"
    with open(label_file, "r") as f:
        content = f.read()
    labels = [float(x) for x in content.split(',')]
    return np.array(labels)
    # print("labels=",len(labels))


# 讀取並處理圖片
def load_and_preprocess_image(file_path):
    img = Image.open(file_path)
    # 在這裡，你可以進行圖片的預處理，例如調整大小、正規化等
    img = img.resize((32, 32))  # 舉例：將圖片大小調整為 224x224
    img_array = np.array(img) / 255.0  # 正規化到 [0, 1]
    return img_array



def load_data():
    ori_labels = load_lable()  
    bins = np.arange(0, 301, 30)
    mapped_labels = np.digitize(ori_labels, bins)
    class_list = np.arange(1, 11)
    # class_list = [1,2,3,4,5,6,7,8,9, 10]
    # # 定義範圍和對應的標籤
    # ranges = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # # class_list = ['0-20', '20-40', '40-60', '60-80', '80-100', '100-120', '120-140', '140-160', '160-180', '180-200']
    # class_list = [0,1,2,3,4,5,6,7,8,9]
    # # 將數字映射到範圍
    # result = np.digitize(ori_labels, ranges, right=True)
    # # right=True 表示閉區間右側，即 (a, b]
    # # 根據映射的結果得到對應的標籤
    # mapped_labels = np.array(class_list)[result - 1]

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

    # print("train_images=",train_images)
    # print("--------------------------")
    # print("train_labels=",train_labels)
    # 打印示例图像和标签
    # print("Train Images:", len(train_images))
    # print("Train Labels:", len(train_labels))
    # print("Test Images:", len(test_images))
    # print("Test Labels:", len(test_label))

    # #-------------------------
    # print("after------------------------------------\n")

    # # 顯示資料集的形狀
    # print("Initial shape or dimensions of x_train", str(train_images.shape))
    # print("Initial shape or dimensions of x_test", str(test_images.shape))
    # print("Initial type x_train", type(train_images))
    # print("Initial type x_test", type(test_images))
    # print("Initial type train_labels", type(train_labels))
    # print("Initial type test_label", type(test_label))

    # print("Initial shape or dimensions of train_labels", str(train_labels.shape))
    # print("Initial shape or dimensions of test_label", str(test_label.shape))
    # print('\n')

    # # 顯示每組資料的sample數
    # print ("Number of samples in our training data: " + str(len(train_images)))
    # print ("Number of labels in our training data: " + str(len(train_labels)))
    # print ("Number of samples in our test data: " + str(len(test_images)))
    # print ("Number of labels in our test data: " + str(len(test_label)))

    # print("------------------------------------\n")
    return train_images, train_labels, test_images, test_label, class_list
    # # 創建 TensorFlow Dataset
    # dataset = tf.data.Dataset.from_tensor_slices(data)
    # num_images = tf.data.experimental.cardinality(dataset).numpy()
    # print("Number of images:", num_images)
    # print("=",dataset)