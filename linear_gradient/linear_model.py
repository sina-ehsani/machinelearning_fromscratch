from utils import *
from data import *
import math
from operator import add , sub
import matplotlib.pyplot as plt


##############################################################################

class LinearModel(object):

    def __init__(self, dim):
        self.dim = dim
        self.weights = [0] * dim

    def predict(self, features):
        return dot(self.weights, features)

    def train(self, dataset, l, alpha=0.001):
        self.regularization.l = l

        # use these objects to call the loss and gradient evaluation methods
        loss_f = self.loss_f
        reg = self.regularization
        
        n_iter = 0
        training_loss_list=[]
        validation_loss_list=[]

        while True:
            g = [0] * self.dim

            # WRITE YOUR GRADIENT DESCENT CODE HERE!

            # hint: because the size of your validation set is
            # different from the training set, it is more convenient
            # to divide the loss of each point by the size of the
            # appropriate set, interpreting it as an average
            # loss. This lets you compare the training loss and the
            # validation loss more directly.  Make sure to scale the
            # gradient as well: this makes it easy to compare
            # regularization values and learning rates across datasets
            
            # use the following print function to monitor your procedure
            training_loss = 0
            validation_loss = 0
            

            training_loss=0
            for sample in dataset.training_set:
                prediction=self.classify(sample.features)
                training_loss += loss_f.loss(prediction,sample) + reg.loss(self.weights)
                g = list(map(add , g , loss_f.gradient(prediction,sample)))
                g = list(map(add , g , reg.gradient(self.weights)))


            g = list(map(lambda x: x/len(dataset.training_set) , g ))
            # print(g)

            training_loss = training_loss/len(dataset.training_set)
            
            for sample in dataset.validation_set:
                prediction=self.classify(sample.features)
                validation_loss += loss_f.loss(prediction,sample) + reg.loss(self.weights)

            validation_loss = validation_loss/len(dataset.validation_set)

            # update weights:
            galpha = list(map(lambda x: alpha*x , g ))
            
            
            self.weights = list(map(sub , self.weights , galpha))
            # self.weights = list(map(add , self.weights , galpha))


            print("\r  Train: %.2f Validation: %.2f log10(|g|): %.2f          " % (training_loss, validation_loss, math.log10(dot(g, g))), end='')
            n_iter += 1
            
            
            # # Early Stopping:
            # if  n_iter > 100:
            #   if validation_loss > validation_loss_list[-100]:
            #     training_loss_list.append(training_loss)
            #     validation_loss_list.append(validation_loss)
            #     break
            
            training_loss_list.append(training_loss)
            validation_loss_list.append(validation_loss)

            if math.log10(dot(g, g)) < -4.0:
              break


        
        print()
        print("Training Loss: %.2f, Validation Loss: %.2f, Gradient: %s" %
              (training_loss, validation_loss, g))
        print("  n: %s" % n_iter)
        
        print(training_loss_list)
        iters = range(0,n_iter)
        plt.plot(iters, training_loss_list, 'r--')
        plt.plot(iters, validation_loss_list, 'b-')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('iter')
        plt.ylabel('Loss')
        plt.show()


