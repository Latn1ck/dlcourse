import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    accuracy=np.sum(prediction==ground_truth)/len(prediction)
    tp=np.sum(ground_truth & prediction)
    fn=np.sum((prediction==False) & ground_truth)
    fp=np.sum((ground_truth==False) & prediction)
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1=(2*precision*recall)/(precision+recall)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
