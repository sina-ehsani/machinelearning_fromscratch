from utils import *
import perceptron

import pickle
import sys
import random
import pylab
import copy

def label_stats(dataset):
    '''returns number of each label in dataset
    '5': 25,
    '11': 14,'''
    class_dict=dict()
    for datapoint in dataset:
        class_dict = quant_label(datapoint , class_dict)
    return class_dict 

def quant_label(datapoint , quant_dict):
    '''update label_stats based on the datapoint'''
    if datapoint.label in quant_dict.keys():
        quant_dict[datapoint.label]+=1
    else: quant_dict[datapoint.label]=1
    return quant_dict


def convert_to_pos(dataset,label):
  dataset_test=copy.deepcopy(dataset)
  pos_dataset=list()
  for datapoint in dataset_test:
    if datapoint.label == label:
      datapoint.label = +1
      pos_dataset.append(datapoint)
  return(pos_dataset)

def convert_to_neg(dataset,label):
  dataset_test=copy.deepcopy(dataset)
  neg_dataset=list()
  for datapoint in dataset_test:
    if datapoint.label == label:
      datapoint.label = -1
      neg_dataset.append(datapoint)
  return(neg_dataset)


##############################################################################

class ava(object):

    # Write this code! It should return a model with the same API as your
    # previous models, but should use models obtained from `perceptron.train`

    # your general strategy will be to create transformations that
    # will convert your multiclass dataset to a number of different
    # two-class datasets, train these, and then at test time, you
    # will need to run the binary classifiers, combine their results
    # appropriately, and produce a final preduction.
  
  def __init__(self, k):
    self.k=k

  def ava_train(self,multiclass_dataset):
    self.labels=label_stats(multiclass_dataset.training_set)
    self.ava_models=dict()
    
    seen_list=list()
    for i in self.labels:
      seen_list.append(i)
      pos_dataset=convert_to_pos(multiclass_dataset.training_set,i)
      for j in self.labels:
        if j not in seen_list:
          neg_dataset=convert_to_neg(multiclass_dataset.training_set,j)
          new_dataset = pos_dataset + neg_dataset
          self.ava_models[i,j]= perceptron.train(new_dataset,self.k)

  def classify(self,features):
    score=dict()
    seen_list=list()
    for i in self.labels:
      seen_list.append(i)
      for j in self.labels:
        if j not in seen_list:
          y = self.ava_models[i,j].classify(features)
          if i in score:
            score[i]=score[i]+y
          else: score[i]=y
          if j in score:
            score[j]=score[j]-y
          else: score[j]=y
    return(max(score,key=score.get))


    
def main():
    if len(sys.argv) < 2:
        print("Usage: %s dataset" %
              sys.argv[0])
        sys.exit(1)

    # notice that `dataset` will not have numerical +1/-1 labels, and
    # different reductions will require different transformations.
    dataset = (pickle.load(open(sys.argv[1], "rb")).
               convert_features_to_numerical())

    model = ava(10)
    model.ava_train(dataset)

    # note that this only evaluates accuracy! You'll need to write your own code
    # to compute the confusion matrix and report that separately
    evaluate_model(model, dataset)

if __name__ == '__main__':
    main()
