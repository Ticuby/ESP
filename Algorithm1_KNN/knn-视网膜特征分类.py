import numpy as np
from sklearn.metrics import accuracy_score
from time import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier

sign=0
y = []
x = []
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
# 这里没有使用交叉验证，在最终呈现过程中应使用交叉验证

# splitinto a training and testing set
# X_train, X_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=56)
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(x)):
    print('train_index{}'.format(i))
    X_train, y_train = x[train_index], y[train_index]
    X_test, y_test = x[test_index], y[test_index]
    print("训练样本上采样...")

     # X_train,y_train = up_sample((X_train,y_train))

    # 训练KNN模型

    print("Fitting the classifier to the training set")
    t0 = time()

    clf = KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='kd_tree')
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print(clf.predict_proba(X_test))
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
    print('单个分类器:', scores.mean())

    bagging = BaggingClassifier(clf, n_estimators=20, max_samples=0.5, max_features=0.5)
    scores = cross_val_score(bagging, X_train, y_train, cv=5, scoring='roc_auc')
    print('Bagging：', scores.mean())
    x_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, x_pred)
    print("训练准确率:{}".format(train_acc))
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")

    print(clf.score(X_test, y_test))
    print(clf.predict_proba(X_test))

# #############################################################################
# Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))