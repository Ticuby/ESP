import numpy as np
import cv2
import os
from xlrd import open_workbook

from time import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier


sign = 0
y_Rg = np.empty(shape=0)
y_Rome = np.empty(shape=0)
path = "images/"
data_dir = []

is_color = 1   #是否输入彩色图，为0则为灰度图

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
                x= cv2.resize(cv2.imread(data_dir+img_name[i]), (224, 224))
            else:
                x = cv2.resize(cv2.imread(data_dir + img_name[i], 0), (224, 224))
            x = np.expand_dims(x, 0)
            sign = 1
        else:
            if is_color:
                img = np.expand_dims(cv2.resize(cv2.imread(data_dir + img_name[i]), (224, 224)), 0)
            else:
                img = np.expand_dims(cv2.resize(cv2.imread(data_dir+img_name[i], 0), (224, 224)), 0)
            x = np.concatenate((x, img), axis=0)
        y_Rg = np.append(y_Rg, Retinopathy_grade[i])
        y_Rome = np.append(y_Rome, Risk_of_macular_edema[i])

if is_color:
    n_samples, h, w, c = x.shape
    x = x.reshape((n_samples, h * w * c))
else:
    n_samples, h, w = x.shape
    x = x.reshape((n_samples, h*w))
dataset_Rg = [x, y_Rg]
dataset_Rome = [x, y_Rome]
n_classes =4
n_features = x.shape[1]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# 数据集划分
# 这里没有使用交叉验证，在最终呈现过程中应使用交叉验证

# split into a training and testing set
# X_train, X_test, y_train, y_test = train_test_split(
#     x, y_Rg, test_size=0.20, random_state=42)
from sklearn.model_selection import KFold
kf = KFold(n_splits=10,shuffle=True)
for train_index, test_index in kf.split(x):
    X_train, y_train = x[train_index], y_Rg[train_index]
    X_test, y_test = x[test_index], y_Rg[test_index]
# skf = KFold(n_splits=5)
# for train_index, test_index in skf.split(x, y_Rg):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = x[train_index], x[test_index]
#     y_train, y_test = y_Rg[train_index], y_Rg[test_index]


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

    # 数据标准化
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train_pca = sc.fit_transform(X_train_pca)
    X_test_pca = sc.transform(X_test_pca)

    # #############################################################################
    # KNN模型

    # 交叉验证k:
    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
       knn = KNeighborsClassifier(n_neighbors=k)
       knn.fit(X_train_pca, y_train)
       scores = cross_val_score(knn, X_train_pca, y_train, cv=10, scoring='accuracy')
       k_scores.append(scores.mean())

    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

    # knn&bagging:
    clf = KNeighborsClassifier(n_neighbors=15, weights='distance',algorithm='kd_tree')
    clf.fit(X_train_pca, y_train)
    print(clf.score(X_test_pca, y_test))
    scores = cross_val_score(clf, X_train_pca, y_train, cv=5, scoring='accuracy')
    print('accuracy of knn:', scores.mean())

    bagging = BaggingClassifier(clf, n_estimators=20, max_samples=0.5, max_features=0.5)
    scores = cross_val_score(bagging, X_train_pca, y_train, cv=5, scoring='accuracy')
    print('accuracy of bagging on knn：', scores.mean())

    # #############################################################################
    #Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
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
            B, G, R = cv2.split(images[i].reshape((h, w, c)))
            img = np.dstack((R,G,B))
            plt.imshow(img, cmap=plt.cm.gray)
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









