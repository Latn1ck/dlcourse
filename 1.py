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
knn_classifier = KNN(k=1)
knn_classifier.fit(binary_train_X, binary_train_y)