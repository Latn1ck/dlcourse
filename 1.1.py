import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from dataset import load_svhn
from knn import KNN
from metrics import binary_classification_metrics, multiclass_accuracy


train_X, train_y, test_X, test_y = load_svhn("data", max_train=1000, max_test=100)
binary_train_mask = (train_y == 0) | (train_y == 9)
binary_train_X = train_X[binary_train_mask]
binary_train_y = train_y[binary_train_mask] == 0

binary_test_mask = (test_y == 0) | (test_y == 9)
binary_test_X = test_X[binary_test_mask]
binary_test_y = test_y[binary_test_mask] == 0
binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1)
binary_test_X = binary_test_X.reshape(binary_test_X.shape[0], -1)
num_folds = 5
train_folds_X = np.array_split(binary_train_X,num_folds)
train_folds_y = np.array_split(binary_train_y,num_folds)
k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
k_to_f1 = {}
print('BINARY')
for k in k_choices:
    cl=KNN(k=k)
    f1s=np.zeros(num_folds,np.float32)
    for i in range(num_folds):
        cv_X=np.concatenate([part for j, part in enumerate(train_folds_X) if j != i])
        cv_y=np.concatenate([part for j, part in enumerate(train_folds_y) if j != i])
        cl.fit(cv_X,cv_y)
        pred=cl.predict(train_folds_X[i])
        precision, recall, f1, accuracy = binary_classification_metrics(pred, train_folds_y[i])
        f1s[i]=f1
    k_to_f1[k]=np.average(f1s)
k_to_f1=dict(sorted(k_to_f1.items(), key=lambda k:k[1],reverse=True))
for k in k_to_f1:
    print('k = %d, f1 = %f' % (k, k_to_f1[k]))
best_k = list(k_to_f1.keys())[0]
print(f'best k={best_k}')
best_knn_classifier = KNN(k=best_k)
best_knn_classifier.fit(binary_train_X, binary_train_y)
prediction = best_knn_classifier.predict(binary_test_X)

precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("Best KNN with k = %s" % best_k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1)) 

print('MULTI')
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

num_folds = 5
train_folds_X = np.array_split(train_X,num_folds)
train_folds_y = np.array_split(train_y,num_folds)

# TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y

k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
k_to_accuracy = {}

for k in k_choices:
    cl=KNN(k=k)
    f1s=np.zeros(num_folds,np.float32)
    for i in range(num_folds):
        cv_X=np.concatenate([part for j, part in enumerate(train_folds_X) if j != i])
        cv_y=np.concatenate([part for j, part in enumerate(train_folds_y) if j != i])
        cl.fit(cv_X,cv_y)
        pred=cl.predict(train_folds_X[i])
        accuracy = multiclass_accuracy(pred, train_folds_y[i])
        f1s[i]=accuracy
    k_to_f1[k]=np.average(f1s)
k_to_f1=dict(sorted(k_to_f1.items(), key=lambda k:k[1],reverse=True))
for k in k_to_f1:
    print('k = %d, f1 = %f' % (k, k_to_f1[k]))
best_k = list(k_to_f1.keys())[0]
print(f'best k={best_k}')
best_knn_classifier = KNN(k=best_k)
best_knn_classifier.fit(train_X, train_y)
prediction = best_knn_classifier.predict(test_X)
print("Accuracy: %4.2f" % multiclass_accuracy(prediction, test_y)) 