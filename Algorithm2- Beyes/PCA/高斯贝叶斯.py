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
y_Rg=np.empty(shape=0)
y_Rome=np.empty(shape=0)
path = "images/"
data_dir = []

is_color = 1   #是否输入彩色图，为0则为灰度图
acc_res = []
# recall_res=[]
# hunxiao_res=[]

# #############################################################################
# 读取数据集

for data_dir in os.listdir(path):
    print("reading dir: %s"%data_dir)
    data_dir = path+data_dir+'/'
    for data in os.listdir(data_dir):
        if data[-3:] == "xls":
            workbook = open_workbook(data_dir+data)
            break
    sheet = workbook.sheet_by_index(0)
    img_name = sheet.col_values(0)[1:]
    Retinopathy_grade = sheet.col_values(2)[1:]
    Risk_of_macular_edema = sheet.col_values(3)[1:]

    for i in range(len(img_name)):
        if sign == 0:
            if is_color:
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
            x = np.concatenate((x,img),axis=0)
        y_Rg = np.append(y_Rg,Retinopathy_grade[i])
        y_Rome = np.append(y_Rome,Risk_of_macular_edema[i])

if is_color:
    n_samples, h, w, c = x.shape
    x = x.reshape((n_samples, h * w * c))
else:
    n_samples, h, w = x.shape
    x = x.reshape((n_samples,h*w))
dataset_Rg=[x,y_Rg]
dataset_Rome=[x,y_Rome]
n_classes =4
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
for train_index, test_index in kf.split(x):
    print('train_index {}'.format(i))
    i+=1
    X_train, y_train = x[train_index], y_Rg[train_index]
    X_test, y_test = x[test_index], y_Rg[test_index]
    # #############################################################################
    # Compute a PCA (eigenfaces) on the dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    # PCA降维

    n_components = 20

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
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    #贝叶斯主代码
    # from sklearn.naive_bayes import GaussianNB
    # clf = GaussianNB()

    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train_pca, y_train)
    print(clf)

    # 测试evaluate y_test == y_pred rate
    y_pred = clf.predict(X_test_pca)
    y_pred2 = clf.predict(X_train_pca)
    acc = np.sum(y_test == y_pred) / X_test_pca.shape[0]
    acc2 = np.sum(y_train == y_pred2) / X_train_pca.shape[0]
    print('X_test:', X_test_pca.shape[0])
    print('test acc: %.3f' % acc)
    print('X_train:', X_train_pca.shape[0])
    print('train acc: %.3f' % acc2)
    acc_res.append(acc)

    print(clf.score(X_test_pca, y_test))                         #预测准确率
    print(metrics.classification_report(y_test,y_pred)) #包含准确率，召回率等信息表
    print(metrics.confusion_matrix(y_test,y_pred))      #混淆矩阵
print(sum(acc_res)/len(acc_res))


# #精确度：precision，正确预测为正的，占全部预测为正的比例，TP / (TP+FP)
# 召回率：recall，正确预测为正的，占全部实际为正的比例，TP / (TP+FN)
# F1-score：精确率和召回率的调和平均数，2 * precision*recall / (precision+recall)







