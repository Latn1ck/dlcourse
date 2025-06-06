import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    if len(predictions.shape)==1:
      predictions-=np.max(predictions)
      return np.exp(predictions)/np.sum(np.exp(predictions))
    else:
        predictions-=np.max(predictions,axis=1).reshape(-1,1)
        return np.exp(predictions)/(np.sum(np.exp(predictions),axis=1).reshape(-1,1))


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if isinstance(target_index,int):
      return -np.log(probs[target_index])
    else:
        pr=np.zeros(probs.shape)
        pr[np.array(range(probs.shape[0])),target_index[0]]=1
        return -np.sum(pr*np.log(probs))


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loop
    dprediction=predictions.copy()
    softMaxes=softmax(dprediction)
    loss=cross_entropy_loss(softMaxes,target_index)
    if isinstance(target_index,int):
        targets=np.zeros(predictions.shape,float)
        targets[target_index]=1
        dprediction=softMaxes-targets
    else:
      targets=np.zeros(predictions.shape,float)
      targets[np.array(range(softMaxes.shape[0])),target_index[0]]=1
      #targets[target_index,np.array(range(predictions.shape[1]))]=1
      dprediction=softmax(dprediction)-targets
    return loss, dprediction
    


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss=reg_strength*np.sum(W**2)
    grad=2*reg_strength*W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W) # shape=(num_batch,classes)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    loss,dW=softmax_with_cross_entropy(predictions,target_index)
    dW=(X.T).dot(dW)
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            
            for i in batches_indices:
              lossCE,dWCE=linear_softmax(X[i],self.W,y[i])
              lossR,dWR=l2_regularization(self.W,reg)
              loss=lossCE+lossR
              dW=dWCE+dWR
              self.W-=learning_rate*dW
              loss_history.append(loss)
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred=np.argmax(softmax(np.dot(X,self.W)), axis=1)
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
      

        return y_pred



                
                                                          

            

                
