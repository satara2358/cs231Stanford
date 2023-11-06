from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  Una red neuronal completamente conectada de dos capas. La red tiene una dimensión de entrada de N, una dimensión de capa oculta de H y realiza clasificación en C clases. Entrenamos la red con una función de pérdida softmax y regularización L2 en las matrices de pesos. La red utiliza una no linealidad ReLU después de la primera capa completamente conectada.
  En otras palabras, la red tiene la siguiente arquitectura:
    entrada - capa completamente conectada - ReLU - capa completamente conectada - softmax
  Las salidas de la segunda capa completamente conectada son las puntuaciones para cada clase.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Inicializa el modelo. 
    Los pesos se inicializan con pequeños valores aleatorios y los sesgos se inicializan a cero. 
    Los pesos y sesgos se almacenan en la variable self.params, que es un diccionario con las siguientes claves:
    W1: Pesos de la primera capa; tiene forma: (D, H)
    b1: Sesgos de la primera capa; tiene forma: (H,)
    W2: Pesos de la segunda capa; tiene forma: (H, C)
    b2: Sesgos de la segunda capa; tiene forma: (C,)

    Inputs:
    - Tamaño de entrada: La dimensión D de los datos de entrada.
    - Tamaño de capa oculta: El número de neuronas H en la capa oculta.
    - Tamaño de salida: El número de clases C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Calcular la pérdida y los gradientes para una red neuronal completamente conectada de dos capas.

    Inputs:
    - X: Datos de entrada de forma (N, D). Cada X[i] es una muestra de entrenamiento.
    - y: Vector de etiquetas de entrenamiento. y[i] es la etiqueta para X[i], y cada y
      [i] es un número entero en el rango 0 <= y[i] < C. 
      Este parámetro es opcional; si no se pasa, solo se devuelven las puntuaciones, y si se pasa, en su lugar se devuelve 
      la pérdida y los gradientes.
    - reg: Intensidad de la regularización.
    Return
      Si y es None, devuelve una matriz de puntuaciones de forma (N, C) donde scores[i, c] es la puntuación para la 
      clase c en la entrada X[i].
      Si y no es None, en su lugar devuelve una tupla que contiene:

    - loss: Pérdida (pérdida de datos y pérdida de regularización) para este lote de muestras de entrenamiento.
    - grads: Diccionario que mapea nombres de parámetros a gradientes de esos parámetros 
      con respecto a la función de pérdida; tiene las mismas claves que self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


