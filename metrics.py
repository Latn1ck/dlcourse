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
    tp=np.sum((ground_truth==True)&(prediction==True))
    fp=np.sum((ground_truth==False)&(prediction==True))
    fn=np.sum((ground_truth==True)&(prediction==False))
    precision=0 if (tp+fp)<=0 else tp/(tp+fp)
    recall=0 if (tp+fn)<=0 else tp/(tp+fn)
    f1=0 if precision+recall==0 else (2*precision*recall)/(precision+recall)
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
