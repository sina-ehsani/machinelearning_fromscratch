from utils import *
import perceptron

import pickle
import sys
import random
import pylab
import copy


##############################################################################

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
  return pos_dataset 

def convert_to_neg(dataset,label):
  dataset_test=copy.deepcopy(dataset)
  neg_dataset=list()
  for datapoint in dataset_test:
    if datapoint.label == label:
      datapoint.label = -1
      neg_dataset.append(datapoint)
  return neg_dataset

def equalspilit(sorted_label):
  left=list()
  right=list()
  while sorted_label:
    pops=sorted_label.pop()
    if len(left)<=len(right):
      left.append(pops)
    else:
      right.append(pops)
  return left , right



class Tree:
    def __init__(self , model , left_labels, right_labels):
        self.left = None
        self.right = None
        self.model = model
        self.left_label=left_labels
        self.right_label=right_labels

class btt(object):

    # Write this code! It should return a model with the same API as your
    # previous models, but should use models obtained from `perceptron.train`

    # your general strategy will be to create transformations that
    # will convert your multiclass dataset to a number of different
    # two-class datasets, train these, and then at test time, you
    # will need to run the binary classifiers, combine their results
    # appropriately, and produce a final preduction.
  
  def __init__(self, k):
    self.k=k

  def _grow_tree(self,sorted_labels):
    left, right=equalspilit(sorted_labels)
    left_labels=[i[0] for i in left]
    right_labels=[i[0] for i in right]
    

    left_data = [item for sublist in [convert_to_pos(self.training_set,i[0]) for i in left] for item in sublist]
    right_data = [item for sublist in [convert_to_neg(self.training_set,i[0]) for i in right] for item in sublist]
    merge_data = left_data + right_data
    # print(merge_data)


    node = Tree(perceptron.train(merge_data,self.k),left_labels, right_labels)

    if len(left) >= 2:
      node.left=self._grow_tree(left)
    if len(right) >= 2:
      node.right=self._grow_tree(right)
    return node

  def btt_train(self,multiclass_dataset):
    self.training_set=multiclass_dataset.training_set
    self.labels=label_stats(multiclass_dataset.training_set)
    sorted_d = sorted(self.labels.items(), key=lambda x: x[1])
    
    self.tree = self._grow_tree(sorted_d)
    return self.tree


  def classify(self, features):
      """Predict class for a single sample."""
      node = self.tree
      answer = node.right_label + node.left_label
      while len(answer)>1:
        if node.model.classify(features)==+1:
          answer=node.left_label
          node=node.left
        else:
          answer=node.right_label
          node=node.right 
      return answer[0]


    
def main():
    if len(sys.argv) < 2:
        print("Usage: %s dataset" %
              sys.argv[0])
        sys.exit(1)

    # notice that `dataset` will not have numerical +1/-1 labels, and
    # different reductions will require different transformations.
    dataset = (pickle.load(open(sys.argv[1], "rb")).
               convert_features_to_numerical())

    model = btt(5)
    tree=model.btt_train(dataset)

    # note that this only evaluates accuracy! You'll need to write your own code
    # to compute the confusion matrix and report that separately
    evaluate_model(model, dataset)

if __name__ == '__main__':
    main()
