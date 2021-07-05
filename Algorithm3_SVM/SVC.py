import numpy as np
import cv2
import os
from xlrd import open_workbook
from sklearn.svm import SVC
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

sign=0
y_Rg=np.empty(shape=0)
y_Rome=np.empty(shape=0)
path = "D:/ESP/images/"
data_dir = []

is_color = 1   #是否输入彩色图，为0则为灰度图

# #############################################################################
# 读取数据集

for data_dir in os.listdir(path):   #遍历Base11——Base34
    print("reading dir: %s"%data_dir)
    data_dir = path+data_dir+'/'    #进入Base**
    for data in os.listdir(data_dir):
        if data[-3:] == "xls":      #找到Excel文件
            workbook = open_workbook(data_dir+data)    #打开Excel表
            break
    sheet = workbook.sheet_by_index(0)      #第一个工作表
    img_name = sheet.col_values(0)[1:]      #提取第一列数据（图片名）
    Retinopathy_grade = sheet.col_values(2)[1:]     #提取第三列数据（视网膜病变等级）
    Risk_of_macular_edema = sheet.col_values(3)[1:] #提取第四列数据（黄斑水肿风险等级）

    for i in range(len(img_name)):          #（图片的数量）遍历图片文件提取特征值
        if sign == 0:                       #data_dir+img_name[i]图片名
            if is_color:
                # x_1 = cv2.imread(data_dir + img_name[i],0)
                x_2 = cv2.imread(data_dir + img_name[i])
                # x_1 = cv2.resize(x_1,(224,224))
                x_2 = cv2.resize(x_2, (224, 224))
                x_2 = cv2.split(x_2)
                # cv2.imshow("1",x_1)
                x= cv2.resize(cv2.imread(data_dir+img_name[i]),(224,224))
            else:
                x = cv2.resize(cv2.imread(data_dir + img_name[i], 0), (224, 224))
            x = np.expand_dims(x, 0)
            sign = 1
        else:
            if is_color:
                img = np.expand_dims(cv2.resize(cv2.imread(data_dir + img_name[i]), (224, 224)), 0)
            else:
                img = np.expand_dims(cv2.resize(cv2.imread(data_dir+img_name[i],0),(224,224)),0)
            x = np.concatenate((x,img),axis=0)  #150528=3*(224^2)（RGB三个通道）一张图片特征拼接为x的一行
        y_Rg = np.append(y_Rg,Retinopathy_grade[i])     #1200张视网膜病变等级拼接为y_Rg一行
        y_Rome = np.append(y_Rome,Risk_of_macular_edema[i]) #1200张黄斑水肿拼接为y_Rome一行

if is_color:
    n_samples, h, w, c = x.shape #n_samples=1200  h=224  w=224  c=3（彩色图像）
    x = x.reshape((n_samples, h * w * c)) #转换数组为1200行  224*224*3列
else:
    n_samples, h, w = x.shape   #n_samples=1200  h=224  w=224
    x = x.reshape((n_samples,h*w))        #转换数组为1200行  224*224列
dataset_Rg=[x,y_Rg]        #视网膜病变等级数据集（图像特征+等级）
dataset_Rome=[x,y_Rome]    #黄斑水肿风险数据集  （图像特征+等级）
n_classes =4    #等级（0、1、2、3）
n_features = x.shape[1] #数组x的列数（提取出的特征量）

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# 数据集划分
# 这里没有使用交叉验证，在最终呈现过程中应使用交叉验证

# split into a training and testing set
# X_train, X_test, y_train, y_test = train_test_split(
#     x, y_Rg, test_size=0.2, random_state=42)      #取出数据集的1/4用于测试
from sklearn.model_selection import KFold
kf = KFold(n_splits=10,shuffle=True)
for train_index, test_index in kf.split(x):
    X_train, y_train = x[train_index], y_Rg[train_index]
    X_test, y_test = x[test_index], y_Rg[test_index]
#############################################################################
# Compute a PCA (eigenfaces) on the dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
# PCA降维

    n_components = 25

    print("Extracting the top %d eigen from %d imgs"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    if is_color:
        eigen = pca.components_.reshape((n_components, h, w, c))
    else:
        eigen = pca.components_.reshape((n_components, h, w))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    pca.fit(X_train,y_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))


    # #############################################################################


    # 训练SVM模型

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

    svc = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)

    # clf = GridSearchCV(
    #     SVC(kernel='rbf', class_weight='balanced'), param_grid
    # )
    #clf = clf.fit(X_train_pca, y_train)
    svc.fit(X_train_pca,y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(svc.best_estimator_)


    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = svc.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# 画图

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        if is_color:
            B,G,R = cv2.split(images[i].reshape((h, w, c)))
            img = np.dstack((R,G,B))
            plt.imshow(img,cmap=plt.cm.gray)
        else:
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = ['predicted: %s\ntrue:      %s' % (y_pred[i], y_test[i])
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigen_titles = ["eigenface %d" % i for i in range(eigen.shape[0])]
if not is_color:
    plot_gallery(eigen, eigen_titles, h, w)
plt.show()







