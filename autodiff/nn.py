#!/usr/bin/env python3

import math
import random
import pickle
import sys
import time
import matplotlib.pyplot as plt


from autodiff import *
from data import *
from utils import *

##############################################################################
def convert_one_hot(label,classes=10):
    final_label = zero_vector(classes)
    final_label.vals[label] = Var(1)
    return final_label


def argmax(lst):
    max_ix = 0
    max_value = lst[0]
    for i, v in enumerate(lst):
        if v > max_value:
            max_ix = i
            max_value = v
    return max_ix


def plot_acc(n_iter,training_acc_list,validation_acc_list):

    iters = range(0,n_iter+1)
    plt.plot(iters, training_acc_list, 'r--')
    plt.plot(iters, validation_acc_list, 'b-')
    plt.legend(['Training acc', 'Validation acc'])
    plt.xlabel('iter')
    plt.ylabel('Accuracy')
    plt.show()


def confmatrixplot(nn,dataset):
    from mlxtend.evaluate import confusion_matrix
    from mlxtend.plotting import plot_confusion_matrix

    def canf_matrix(model,sample_set):
      y_predict=[]
      y_true=[]
      for i, sample in enumerate(sample_set):
        y_predict.append(int(model.classify(sample.features)))
        y_true.append(int(sample.label))
      return y_predict , y_true

    y_predict , y_true = canf_matrix(nn,dataset.testing_set)
    cm = confusion_matrix(y_target=y_true, 
                          y_predicted=y_predict, 
                          binary=False)

    fig, ax = plot_confusion_matrix(conf_mat=cm)
    plt.show()



class NN(object):

    def __init__(self):
        # WRITE THIS
        #
        # Define your layers here
        self.ll1 = LinearLayer(784,16)
        self.ll2 = LinearLayer(16,12)
        self.ll3 = LinearLayer(12,10)
        self.layers = [self.ll1, self.ll2 , self.ll3 ]
        # self.layers = [self.ll1  ]

        # self.ll3 = LinearLayer(16,10)

        self.input_minibatch = zero_vector(28 * 28)
        self.onehot_label = zero_vector(10)
        self.network_prediction = self.apply_nn(self.input_minibatch)
        # self.loss = l2_vec_loss(self.network_prediction, self.onehot_label)
        self.loss = cross_entropy_loss(self.network_prediction, self.onehot_label)        
        
    def apply_nn(self, in_vec):
        # WRITE THIS
        # To apply a layer to an input vector, use the '*' operator:
        # output = layer * input
        output = self.ll1 * in_vec
        output = ReLU(output)
        output = self.ll2 * output
        output = ReLU(output)
        output3 = self.ll3 * output
        output = softmax(output)
        return output



    def gradient_descent_step(self, labeled_sample):
        # WRITE THIS
        # 1) set the values of the input minibatch to the features in labeled_sample
        # 2) set the one-hot version of the label to the correct value
        # 3) evaluate the loss function and gradient with respect to loss
        # 4) for each layer in your network, take a step in direction of the
        #    negative gradient wrt the loss

        # Note that the LinearLayer objects are defined as a list of
        # rows (layer.rows[0], layer.rows[1], etc), and each row is a list of vals:
        # (layer.rows[0].vals[0], layer.rows[0].vals[1], etc)

        # remember that in reverse mode automatic differentiation, the gradient of the
        # evaluated expression wrt the variable is stored together with the value of
        # the variable.

        # in other words,
        # var.value gives the value, var.gradient_value gives the gradient

        # Note, finally, that you won't have to call apply_nn()
        # here. That computation graph has already been defined on
        # __init__(). All you have to do is set the variable values
        # and call clear_eval() followed by evaluate() and backward()
        
        alpha = 0.1

        for i , ( val_mini , feature) in enumerate(zip(self.input_minibatch.vals , labeled_sample.features)):
          self.input_minibatch.vals[i].value=feature
        self.onehot_label = convert_one_hot(labeled_sample.label)
        self.network_prediction = self.apply_nn(self.input_minibatch)
        self.loss = cross_entropy_loss(self.network_prediction, self.onehot_label)

        self.loss.clear_eval()
        v = self.loss.evaluate()
        self.loss.backward()



        for x , layer in enumerate(self.layers):
          for i,  row in enumerate(layer.rows):
            for j , val in enumerate(row.vals):
              self.layers[x].rows[i][j].value -= val.gradient_value * alpha
        



    def evaluate_sample(self, features):
        for var, feature_value in zip(self.input_minibatch, features):
            var.value = feature_value
        self.network_prediction.clear_eval()
        return self.network_prediction.evaluate()

    def predict(self, features):
        return self.evaluate_sample(features)

    def classify(self, features):
        return argmax(self.predict(features))

if __name__ == '__main__':
    dataset = pickle.load(open("mnist-digits.pickle", "rb"))
    nn = NN()
    # epochs
    validation_acc_list=[]
    training_acc_list=[]
    n_iter = 0
    

    for i in range(25):

        start_time = time.time()
        # shuffle the training set between epochs
        random.shuffle(dataset.training_set)
        for sample in dataset.training_set:
            nn.gradient_descent_step(sample)
            print(".", end='')
            sys.stdout.flush()
        print("Epoch done")

        
        eval_dict = evaluate_model(nn, dataset)
        
        print("--- %s seconds ---" % (time.time() - start_time))
        
        # Early Stopping:
        n_iter += 1
        training_acc_list.append(eval_dict['training'])
        validation_acc_list.append(eval_dict['validation'])

        if  n_iter > 3:
          print(eval_dict['validation'] , validation_acc_list[-3])
          if eval_dict['validation'] < validation_acc_list[-3]:
            training_acc_list.append(eval_dict['training'])
            validation_acc_list.append(eval_dict['validation'])
            break

    plot_acc(n_iter,training_acc_list,validation_acc_list)
    evaluate_test(nn, dataset)
    confmatrixplot(nn,dataset)





