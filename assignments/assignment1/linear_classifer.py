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
    tmp_predictions = predictions.copy()
    if (len(tmp_predictions.shape) == 1):
      tmp_predictions = tmp_predictions.reshape(1, -1)
    tmp_predictions -= np.max(tmp_predictions, axis = 1)[:, np.newaxis]
    exps = np.exp(tmp_predictions)
    inv_sums = 1 / np.sum(exps, axis=1)
    return exps * inv_sums[:, np.newaxis]


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
    # raise Exception("Not implemented!")
    # shift = np.arange(0, target_index.shape[0] * probs.shape[1], probs.shape[1])
    # tind = target_index.ravel() + shift
    # return np.sum(-np.log(probs.ravel()[tind]))
    return np.sum(-np.log(np.take_along_axis(probs, target_index.reshape(-1, 1), axis=1)))


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
    # Your final implementation shouldn't have any loops
    # raise Exception("Not implemented!")
    s_vals = softmax(predictions)
    loss = cross_entropy_loss(s_vals, target_index)
    
    
    true_prob = np.zeros_like(predictions)

    dprediction = np.zeros(predictions.shape)
    np.put_along_axis(true_prob, target_index.reshape(-1, 1), 1, axis=1)
    dprediction = (s_vals - true_prob) / len(true_prob)
    # dprediction += s_vals / len(dprediction)
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

    l2_reg_loss = reg_strength * np.sum(np.power(W, 2))
    grad = 2 * W * reg_strength 

    return l2_reg_loss, grad
    

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
    predictions = np.dot(X, W)

    loss, dL = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dL)
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

            loss = 0
            for bi in batches_indices:
              loss_1, dW_1 = linear_softmax(X[bi], self.W, y[bi])
              loss_2, dW_2 = l2_regularization(self.W, reg)
              self.W -= learning_rate * (dW_1 + dW_2)
              loss += loss_1 + loss_2
            
            loss_history.append(loss)
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            # raise Exception("Not implemented!")

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
        # y_pred = np.zeros(X.shape[0], dtype=np.int)
        y_pred = softmax(np.dot(X, self.W))
        y_pred = np.argmax(y_pred, axis=1)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
