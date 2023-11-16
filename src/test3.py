import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_DIR = "../CWT result/CWT_MHE_result"

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


# 獲取圖片文件列表
img_files = [os.path.join(IMG_DIR, file) for file in os.listdir(IMG_DIR) if file.endswith(('png'))]
# 讀取並處理圖片
def load_and_preprocess_image(file_path):
    img = Image.open(file_path)
    # 在這裡，你可以進行圖片的預處理，例如調整大小、正規化等
    img = img.resize((32, 32))  # 舉例：將圖片大小調整為 224x224
    img_array = np.array(img) / 255.0  # 正規化到 [0, 1]
    return img_array

# 讀取所有圖片並堆疊成數組
data = [load_and_preprocess_image(file) for file in img_files]
data = np.stack(data)

# 假設你有一個對應的標籤列表 labels，它的長度應該和圖片數目相同
labels = load_lable()  # 這裡是你的標籤列表

# 使用 train_test_split 函數拆分訓練集和測試集
train_images, test_images, train_labels, test_label = train_test_split(data, labels, test_size=0.2, random_state=42)
train_labels


# print("train_images=",train_images)
# print("--------------------------")
# print("train_labels=",train_labels)
# 打印示例图像和标签
print("Train Images:", len(train_images))
print("Train Labels:", len(train_labels))
print("Test Images:", len(test_images))
print("Test Labels:", len(test_label))

#-------------------------
print("after------------------------------------\n")

# 顯示資料集的形狀
print("Initial shape or dimensions of x_train", str(train_images.shape))
print("Initial shape or dimensions of x_test", str(test_images.shape))
print("Initial type x_train", type(train_images))
print("Initial type x_test", type(test_images))
print("Initial type train_labels", type(train_labels))
print("Initial type test_label", type(test_label))

print("Initial shape or dimensions of train_labels", str(train_labels.shape))
print("Initial shape or dimensions of test_label", str(test_label.shape))
print('\n')

# 顯示每組資料的sample數
print ("Number of samples in our training data: " + str(len(train_images)))
print ("Number of labels in our training data: " + str(len(train_labels)))
print ("Number of samples in our test data: " + str(len(test_images)))
print ("Number of labels in our test data: " + str(len(test_label)))

print("------------------------------------\n")

# # 創建 TensorFlow Dataset
# dataset = tf.data.Dataset.from_tensor_slices(data)
# num_images = tf.data.experimental.cardinality(dataset).numpy()
# print("Number of images:", num_images)
# print("=",dataset)