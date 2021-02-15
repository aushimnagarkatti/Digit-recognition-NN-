# -*- coding: utf-8 -*-

import numpy as np
from numpynet.layer import Dense, ELU, ReLU, SoftmaxCrossEntropy
from numpynet.function import Softmax
from numpynet.utils import Dataloader, one_hot_encoding, load_MNIST, save_csv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

IntType = np.int64
FloatType = np.float64

class Model(object):
    """Model starts here
    """
    def __init__(self, input_dim, output_dim):
        """__init__ Constructor

        Arguments:
            input_dim {IntType or int} -- Number of input dimensions
            output_dim {IntType or int} -- Number of classes
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_fn = SoftmaxCrossEntropy(axis=-1)
        self.build_model()
        self.accuracy=0
       

    def build_model(self):
        """build_model Build the model using numpynet API
        """
        self.l1=Dense( 784, 256)
        self.l2=ELU(1)
        self.l3=Dense(256,64)
        self.l4=ELU(1)
        self.l5=Dense(64,10)
        self.l6=ELU(1)
        self.l7=SoftmaxCrossEntropy()

    def __call__(self, X):
        """__call__ Forward propogation of the model

        Arguments:
            X {np.ndarray} -- Input batch
           

        Returns:
            np.ndarray -- The output of the model. 
                You can return the logits or probits, 
                which depends on the way how you structure 
                the code.
        """
        op=self.l1.__call__(X)
        op=self.l2.__call__(op)
        op=self.l3.__call__(op)
        op=self.l4.__call__(op)
        op=self.l5.__call__(op)
        op=self.l6.__call__(op)    

        return op

        

    def bprop(self, istraining,X,train_Y):
        """bprop Backward propogation of the model

        Arguments:
            logits {np.ndarray} -- The logits of the model output, 
                which means the pre-softmax output, since you need 
                to pass the logits into SoftmaxCrossEntropy.
            labels {np,ndarray} -- True one-hot lables of the input batch.

        Keyword Arguments:
            istraining {bool} -- If False, only compute the loss. If True, 
                compute the loss first and propagate the gradients through 
                each layer. (default: {True})

        Returns:
            FloatType or float -- The loss of the iteration
            Logits
        """
        output=self.__call__(X) #gets the output of last ELU layer
        loss=self.l7.__call__(output,train_Y)  #calls cross entropy&softmax, and returns loss
        if(istraining==False):
            return loss,output
        
        grad=self.l7.bprop()

        #3rd Dense Layer
        gradelu=self.l6.bprop()
        grad=np.multiply(gradelu,grad)
        grad=self.l5.bprop(grad)

        #2nd Dense Layer
        gradelu=self.l4.bprop()
        grad=np.multiply(gradelu,grad)
        grad=self.l3.bprop(grad)

        #1st Dense Layer
        gradelu=self.l2.bprop()
        grad=np.multiply(gradelu,grad)
        grad=self.l1.bprop(grad)

        return loss,output

        

    def update_parameters(self, lr):
        """update_parameters Update the parameters for each layer.

        Arguments:
            lr {FloatType or float} -- The learning rate
        """
        self.l5.update(lr)
        self.l3.update(lr)
        self.l1.update(lr)


def train(model,
          train_X,
          train_y,
          max_epochs=20,
          lr=1e-3,
          batch_size=16,
          metric_fn=accuracy_score,
          **kwargs):
    """train Train the model

    Arguments:
        model {Model} -- The Model object
        train_X {np.ndarray} -- Training dataset
        train_y {np.ndarray} -- Training labels
        val_X {np.ndarray} -- Validation dataset
        val_y {np.ndarray} -- Validation labels

    Keyword Arguments:
        max_epochs {IntType or int} -- Maximum training expochs (default: {20})
        lr {FloatType or float} -- Learning rate (default: {1e-3})
        batch_size {IntType or int} -- Size of each mini batch (default: {16})
        metric_fn {function} -- Metric function to measure the performance of 
            the model (default: {accuracy_score})
    """
   
    #forward prop is called inside bprop
    loss, output=model.bprop(True,train_X,train_y) #loss is Bx1,

    #Update weights
    model.update_parameters(lr)    

    #Calculate accuracy
    for i in range(batch_size):
        ind=np.where(output[i]==max(output[i]))[0][0]
        true_ind=np.where(train_y[i]==1)[0][0]
        if(ind==true_ind):
            model.accuracy+=1
    

    #Report training error per instance
    train_loss=np.sum(loss) #/49984
    #print("Training Loss:",train_loss)
    return train_loss



    


def inference(model,X, y, batch_size=16, metric_fn=accuracy_score, **kwargs):
    """inference Run the inference on the given dataset

    Arguments:
        model {Model} -- The Neural Network model
        X {np.ndarray} -- The dataset input
        y {np.ndarray} -- The sdataset labels

    Keyword Arguments:
        metric {function} -- Metric function to measure the performance of the model 
            (default: {accuracy_score})

    Returns:
        tuple of (float, float): A tuple of the loss and accuracy
    """
    
    #forward prop is called inside bprop
    loss, output=model.bprop(False,X,y) #loss is Bx1,


    #Calculate accuracy
    for i in range(batch_size):
        ind=np.where(output[i]==max(output[i]))[0][0]
        true_ind=np.where(y[i]==1)[0][0]
        if(ind==true_ind):
            model.accuracy+=1
    

    #Report training error per instance
    val_loss=np.sum(loss)#/9984
    #print("Validation Loss:",val_loss)
    return val_loss


def predict(model,X):
    output=model.__call__(X) #gets the output of last ELU layer
    #Calculate accuracy
    y_pred=[]
    for i in range(len(output)):
        ind=np.where(output[i]==max(output[i]))[0][0]
        y_pred.append(ind)
    return np.array(y_pred)   


       
    




def main():
    print('loading data #####')
    train_X, train_y = load_MNIST(path ='dataset/',name="train")
    val_X, val_y = load_MNIST(path = 'dataset/', name="val")
    test_X = load_MNIST(path = 'dataset/', name="test")
    print('loading data complete #####')
    batchSize = 16
    learningRate = 0.2 
    model = Model(input_dim = 784,output_dim = 10)
    model.build_model()
    print('Model built #####')
    t_loss=[]
    t_acc=[]
    v_loss=[]
    v_acc=[]
    one_hot_train_y = one_hot_encoding(train_y)
    one_hot_val_y = one_hot_encoding(val_y)

    #50k training samples, 200 epochs --> 1562 batches of size 32 samples/batch
    #Start training 
    for k in range(1,201):

        print("\n\nEpoch",k)
        print("********")
        if(k==160):
            learningRate=0.02

        #Training        
        
        train_dataloader = Dataloader(X=train_X, y=one_hot_train_y,batch_size=batchSize)
        model.accuracy=0
        train_loss=0
        #for each batch 
        for i, (features, labels) in enumerate(train_dataloader):
            #call model train
            train_loss += train(model,features, labels, max_epochs=200, lr=learningRate, batch_size=batchSize, metric_fn=accuracy_score)
        t_acc.append(model.accuracy/50000) 
        t_loss.append(train_loss/50000)
        print("Training Loss:",train_loss/50000)
        print("Training Accuracy:",model.accuracy/50000)

        #Validation        
        if(k%2==0):
            model.accuracy=0
            val_loss=0        
            val_dataloader = Dataloader(X=val_X, y=one_hot_val_y,batch_size=batchSize)
            for i, (features, labels) in enumerate(val_dataloader):
                #call validation
                val_loss += inference(model,features, labels, batch_size = batchSize)
                
            v_acc.append(model.accuracy/10000) 
            v_loss.append(val_loss/10000) 
            print("Validation Loss:",val_loss/10000)
            print("Validation Accuracy:",model.accuracy/10000)
         

    print('Training complete #####')
    
    # Plot of train and val accuracy vs iteration
    plt.figure(figsize=(10,7))
    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations')
    plt.title('Accuracy vs number of iterations')
    plt.plot(np.linspace(0,199,200), t_acc, label = 'Train accuracy across iterations')
    plt.plot(np.linspace(0,198,100), v_acc, label = 'Val accuracy across iterations')
    plt.legend(loc = 'lower right')
    plt.show()

    # Plot of train and val loss vs iteration
    plt.figure(figsize=(10,7))
    plt.ylabel('Loss')
    plt.xlabel('Number of iterations')
    plt.title('Loss vs number of iterations')
    plt.plot(np.linspace(0,199,200), t_loss, label = 'Train loss across iterations')
    plt.plot(np.linspace(0,198,100), v_loss, label = 'Val loss across iteration')
    plt.legend(loc='upper right')
    plt.show()
    
    # Inference on test dataset without labels
    test_pred = predict(model,test_X)
    
    save_csv(test_pred)
  

    print("Validation loss: {0}, Validation Acc: {1}%".format(v_loss[-1], 100 * v_acc[-1]))


   
if __name__ == '__main__':
    main()
