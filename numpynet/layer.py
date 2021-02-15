# -*- coding: utf-8 -*-

import numpy as np

FloatType = np.float64
IntType = np.int64


class Layer(object):
    """Layer 
    This is the absract class of implementing layer objects    """


    def __init__(self):
        self.cache = None

    def __call__(self):
        raise NotImplementedError

    def bprop(self):
        raise NotImplementedError


class ReLU(Layer):
    """ReLU Numpy implementation of ReLU activation

    """
    def __init__(self):
        """ReLU Constructor
        """
        super(ReLU, self).__init__()

    def __call__(self, x):
        """__call__ Forward propogation through ReLU

        Arguments:
            x {np.ndarray} -- Input of ReLU Layer with shape (B, D)
                B is the batch size, D is the number of dimensions

        Returns:
            np.ndarray -- Output of ReLU Layer
        """
        self.cache = x
        return np.maximum(x, np.zeros_like(x))

    def bprop(self):
        """bprop Backward propogation of ReLU layer

        Returns:
            np.ndarray -- The gradient flowing out of ReLU
        """
        return 1.0 * (self.cache > 0)


class Dense(Layer):
    """Dense Numpy implementation of Dense Layer
    """
    def __init__(self, dim_in, dim_out):
        """__init__ Constructor

        Arguments:
            dim_in {int} -- Number of the input dimensions 
            dim_out {int} -- Number of the output dimensions
        """
        super(Dense, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # The following are the parameters
        self._W = None
        self._b = None

        # The following are the gradients for the parameters
        self.dW = None
        self.db = None

        # Initialize all the parameters for training
        self._parameter_init()

    def zero_grad(self):
        """zero_grad Clear out the previous gradients
        """
        if self.dW is not None:
            self.dW = np.zeros_like(self.dW)

        if self.db is not None:
            self.db = np.zeros_like(self.db)

    def get_weights(self):
        """get_weights Return the parameters

        Returns:
            list -- A list containing the weights and bias
        """
        return [self._W, self._b]

    def set_weights(self, new_W, new_b):
        """set_weights Set the new parameters

        Arguments:
            new_W {np.ndarray} -- new weights
            new_b {np.ndarray} -- new bias
        """

        self._W = new_W
        self._b = new_b

    def _parameter_init(self):
        """_parameter_init Initialize the parameters
        """
        self._W = np.random.normal(loc=0.0, scale = np.sqrt(2/(self.dim_in+self.dim_out)), size = (self.dim_in,self.dim_out))
        self._b = np.zeros(self.dim_out)
        

    def __call__(self, x):
        """__call__ Forward propogation through Dense layer

        Arguments:
            x {np.ndarray} -- Input of Dense layer with shape (B, D)
                B is the batch size, D is the number of dimensions

        Returns:
            np.ndarray -- Output of Dense Layer
        """
        self.x_inp = x.astype('float64')
        self.x_out= np.add(np.matmul(x,self._W),self._b)
        return self.x_out


    def bprop(self, grad):
        """bprop Backward propogation of Dense Layer

        Arguments:
            grad {np.ndarray} -- Gradient comming from the previous layer

        Returns:
            np.ndarray -- The gradient flowing out of Dense Layer
        """
       
        #Find gradient for each Bxdim_in weight matrix, and B such matrices
        #Also return new gradient wrt output of prev layer
        new_grad=np.matmul(grad,np.transpose(self._W))
        B=grad.shape[0]
        #Find dW and dB
        one=np.ones(B)
        self.dW=(np.matmul(np.transpose(self.x_inp),grad))/B
        self.db=(np.matmul(one,grad))/B
        
         

        return np.array(new_grad)   

    
      

    def update(self, lr):
        """update Update the parameters 
        
        Arguments:
            lr {FloatType or float} -- learning rate
        """
        self._W=np.subtract(self._W,lr*self.dW)
        self._b=np.subtract(self._b,lr*self.db)
        

class ELU(Layer):
    """ELU Numpy implementation of ELU activation
    """
    def __init__(self, alpha):
        """ELU Constructor
        """
        super(ELU, self).__init__()
        self.alpha = alpha
        self.x_inp = None

    def __call__(self, x):
        """__call__ Forward propogation through ELU

        Arguments:
            x {np.ndarray} -- Input of ELU Layer with shape (B, D)
                B is the batch size, D is the number of dimensions

        Returns:
            np.ndarray -- Output of ELU Layer
        """

        self.x_inp = x.astype('float64')
        x = x.astype('float64')
        x[x <= 0] = self.alpha*(np.exp(x[x <= 0]) - 1)
        return x

    def bprop(self):
        """bprop Backward propogation of ELU layer

        Returns:
            np.ndarray -- The gradient flowing out of ELU
        """

        grad = self.x_inp
        grad[grad > 0] = 1
        grad[grad <= 0] = self.alpha*np.exp(grad[grad <= 0])
        return grad


class SoftmaxCrossEntropy(Layer):
    """SoftmaxCrossEntropy Numpy implementation of Softmax and Cross Entroppy 
    """
    def __init__(self, axis=-1):
        """__init__ Constructor

        Keyword Arguments:
            axis {int} -- The axis on which to apply the Softmax (default: {-1})
        """
        super(SoftmaxCrossEntropy, self).__init__()
        self.axis = axis

    def __call__(self, logits, labels):
        """__call__ Forward propogation through Softmax

        Arguments:
            logits {np.ndarray} -- Input logits with shape (B, C)
                B is the batch size, D is the number of classes
            labels {np.ndarray} -- Input one-hot encoded labels with shape (B, C)
                B is the batch size, D is the number of classes

        Returns:
            FloatType --  loss per batch
        """
        B=logits.shape[0]
        C=logits.shape[1] 
        y_pred=np.zeros([B,C])
        L=[]
        #Softmax
        for i in range(B):
            sum=np.sum(np.exp(logits[i]))+ 1e-16
            error=0
            for j in range(C):
                y_pred[i][j]=np.exp(logits[i][j])/sum
                error+=labels[i][j]*np.log(y_pred[i][j])
            L.append(error*-1)
        self.y=y_pred
        self.y_hat=labels
        return np.sum(np.array(L))




    def bprop(self):
        """bprop Backward propogation of Softmax layer

        Returns:
            np.ndarray -- The gradient flowing out of SoftmaxCrossEntropy

        Raises:
            NotImplementedError: [description]
        """
        return np.subtract(self.y,self.y_hat)
