import numpy as np  #导入矩阵操作函数库
import cv2
import os
from xlrd import open_workbook
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from time import time





import warnings

warnings.filterwarnings('ignore')






#忽略一些版本不兼容等警告
warnings.filterwarnings("ignore")

sign=0
y = []
x = []

is_color = 1   #是否输入彩色图，为0则为灰度图
acc_res = []
# recall_res=[]
# hunxiao_res=[]

# #############################################################################
# 读取数据集

with open('messidor_features.artff.txt', 'r') as f:
    lines = f.readlines()[25:]
    for line in lines:
        l = [float(i) for i in line.split(',')]
        x.append(l[:-1])
        y.append(l[-1])


x = np.array(x)
y = np.array(y)
n_classes = 2
n_samples = x.shape[0]
n_features = x.shape[1]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# 数据集划分
# 交叉验证


from sklearn.model_selection import KFold
kf = KFold(n_splits=10,shuffle=True)
i=0
for i, (train_index, test_index) in enumerate(kf.split(x)):
    print('train_index{}'.format(i))
    X_train, y_train = x[train_index], y[train_index]
    X_test, y_test = x[test_index], y[test_index]
    print("训练样本上采样...")


    #贝叶斯主代码
    # from sklearn.naive_bayes import BernoulliNB
    # clf = GaussianNB()

    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    print(clf)

    # 测试evaluate y_test == y_pred rate
    y_pred = clf.predict(X_test)
    y_pred2 = clf.predict(X_train)
    acc = np.sum(y_test == y_pred) / X_test.shape[0]
    acc2 = np.sum(y_train == y_pred2) / X_train.shape[0]
    print('X_test:', X_test.shape[0])
    print('test acc: %.3f' % acc)
    print('X_train:', X_train.shape[0])
    print('train acc: %.3f' % acc2)
    acc_res.append(acc)

    print(clf.score(X_test, y_test))                         #预测准确率
    print(metrics.classification_report(y_test,y_pred)) #包含准确率，召回率等信息表
    print(metrics.confusion_matrix(y_test,y_pred))      #混淆矩阵
print(sum(acc_res)/len(acc_res))


# #精确度：precision，正确预测为正的，占全部预测为正的比例，TP / (TP+FP)
# 召回率：recall，正确预测为正的，占全部实际为正的比例，TP / (TP+FN)
# F1-score：精确率和召回率的调和平均数，2 * precision*recall / (precision+recall)







