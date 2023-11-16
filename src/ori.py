# Alex
# yifanwang0916@outlook.com
# 2019.09.25

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

def myModel(x, getK=1):
    x1 = PixelHop_Unit(x, dilate=1, pad='reflect', num_AC_kernels=9, weight_name='pixelhop1.pkl', getK=getK)

    x2 = PixelHop_Unit(x1, dilate=2, pad='reflect', num_AC_kernels=25, weight_name='pixelhop2.pkl', getK=getK)
    x2 = AvgPooling(x2)

    x3 = PixelHop_Unit(x2, dilate=2, pad='reflect', num_AC_kernels=35, weight_name='pixelhop3.pkl', getK=getK)
    x3 = AvgPooling(x3)

    x4 = PixelHop_Unit(x3, dilate=2, pad='reflect', num_AC_kernels=55, weight_name='pixelhop4.pkl', getK=getK)

    x2 = myResize(x2, x.shape[1], x.shape[2])
    x3 = myResize(x3, x.shape[1], x.shape[2])
    x4 = myResize(x4, x.shape[1], x.shape[2])
    return np.concatenate((x1,x2,x3,x4), axis=3)
if False:
    x = cv2.imread('../data/test.jpg')
    x = x.reshape(1, x.shape[0], x.shape[1], -1)
    feature = myModel(x, getK=1)
if True:
    train_images, train_labels, test_images, test_labels, class_list = import_data_mnist("0-9")  
    N_train=1000
    N_test=500
    SAVE={}
    train_images=train_images[:N_train]
    train_labels=train_labels[:N_train]
    test_images=train_images[:N_test]
    test_labels=train_labels[:N_test]
    
    # 打印示例图像和标签
    print("Train Images:", len(train_images))
    print("Train Labels:", len(train_labels))
    print("Test Images:", len(test_images))
    print("Test Labels:", len(test_labels))

#-------------------------
    print("after------------------------------------\n")
    print("Initial type train_labels", type(train_labels))
    print("Initial type test_labels", type(test_labels))

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