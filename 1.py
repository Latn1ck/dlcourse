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
train_folds_X = np.array_split(binary_train_X, num_folds)
train_folds_y = np.array_split(binary_train_y, num_folds)

# TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y

k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
k_to_f1 = {}  # dict mapping k values to mean F1 scores (int -> float)
for k in k_choices:
    # TODO: perform cross-validation
    # Go through every fold and use it for testing and all other folds for training
    # Perform training and produce F1 score metric on the validation dataset
    # Average F1 from all the folds and write it into k_to_f1
    cl=KNN(k=k)
    f1s=np.zeros(num_folds,np.float32)
    for j in range(num_folds):
        l=list(range(num_folds)).pop(j)
        cl.fit(train_folds_X[l], train_folds_y[l])
        prediction=cl.predict(train_folds_X[j])
        precision, recall, f1, accuracy = binary_classification_metrics(prediction, train_folds_y[j])
        f1s[j]=f1
    k_to_f1[k]=np.average(f1s)


for k in sorted(k_to_f1):
    print('k = %d, f1 = %f' % (k, k_to_f1[k]))