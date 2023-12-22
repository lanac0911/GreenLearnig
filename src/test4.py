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
from utils.load_data import load_data
train_images, train_labels, test_images, test_labels, class_list = load_data()
# class_list = generate_class_list(train_labels, test_labels)
print(f"class_list.len = {len(class_list)}, = , {class_list}")
#-------------------------
print("after------------------------------------\n")

# 顯示資料集的形狀
print("Initial shape or dimensions of x_train", str(train_images.shape))
print("Initial shape or dimensions of x_test", str(test_images.shape))

print("Initial shape or dimensions of train_labels", str(train_labels.shape))
print("Initial shape or dimensions of test_labels", str(test_labels.shape))
print('\n')

# 顯示每組資料的sample數
print ("Number of samples in our training data: " + str(len(train_images)))
print ("Number of labels in our training data: " + str(len(train_labels)))
print ("Number of samples in our test data: " + str(len(test_images)))
print ("Number of labels in our test data: " + str(len(test_labels)))

print("------------------------------------\n")
#-------------------------
SAVE = {}
N_train = len(train_images)
N_test = len(test_images)

train_feature=PixelHop_Unit(train_images, dilate=1, pad='reflect', num_AC_kernels=5, weight_name='pixelhop1_mnist.pkl', getK=1)
train_feature = block_reduce(train_feature, (1, 4, 4, 1), np.mean).reshape(N_train,-1)
train_feature_reduce=LAG_Unit(train_feature,train_labels=train_labels, class_list=class_list,
                            SAVE=SAVE,num_clusters=50,alpha=5,Train=True)

test_feature=PixelHop_Unit(test_images, dilate=1, pad='reflect', num_AC_kernels=5, weight_name='pixelhop1_mnist.pkl', getK=0)
test_feature=block_reduce(test_feature, (1, 4, 4, 1), np.mean).reshape(N_test,-1)
test_feature_reduce=LAG_Unit(test_feature,train_labels=None, class_list=class_list,
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