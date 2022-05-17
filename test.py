#import time
import pandas as pd
from sklearn import datasets, model_selection, svm, metrics
iris = datasets.load_iris()

print("\r\nprint(type(iris))\r\n")
print(type(iris))

print("\r\nprint(iris.keys())\r\n")
print(iris.keys())
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

print("\r\nprint(iris_data.head())\r\n")
print(iris_data.head())

iris_label = pd.Series(data=iris.target)

print("\r\nprint(iris_label.head())\r\n")
print(iris_label.head())

print("\r\nprint(len(iris_data))\r\n")
print(len(iris_data))

data_train, data_test, label_train, label_test = model_selection.train_test_split(iris_data, iris_label)

print("\r\nprint(data_train.head())\r\n")
print(data_train.head())

print("\r\nprint(label_train.head())\r\n")
print(label_train.head())

print("\r\nprint(len(data_train), len(data_test))\r\n")
print(len(data_train), len(data_test))

clf = svm.SVC()
clf.fit(data_train, label_train)
pre = clf.predict(data_test)

print("\r\nprint(type(pre))\r\n")
print(type(pre))

print("\r\nprint(pre)\r\n")
print(pre)
ac_score = metrics.accuracy_score(label_test, pre)

print("\r\nprint(ac_score)\r\n")
print(ac_score)
scores = model_selection.cross_val_score(clf, iris_data, iris_label, cv=3)


print("\r\nprint(scores)\r\n")
print(scores)

print("\r\nprint(scores.mean())\r\n")
print(scores.mean())

input()