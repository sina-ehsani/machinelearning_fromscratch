from utils import *
import perceptron

import pickle
import sys
import random
import pylab

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

class ova(object):

  def __init__(self, k):
    self.k=k

  def ova_train(self,multiclass_dataset):
    self.labels=label_stats(multiclass_dataset.training_set)
    self.ova_models=dict()

    for label in self.labels.keys():
      dataset=multiclass_dataset.convert_labels_to_numerical(label)
      self.ova_models[label]=perceptron.train(dataset,self.k)

  def classify(self,features):
    score=dict()
    for label in self.labels.keys():
      score[label]=self.ova_models[label].classify(features)
    if len([True for value in  score.values() if value==+1]) > 2 :
      # when there are two or more labels with +1: return the one that is more common in the training dataset
      ones_with_labled=[key for key, value in score.items() if value == +1]
      ones_with_labled_training_dist={key:self.labels[key] for key in ones_with_labled}
      return max(ones_with_labled_training_dist,key=ones_with_labled_training_dist.get) 
      [True for keyvalue in  score.values() if value==+1]
    elif all(value == -1 for value in score.values()):
      #When there is no +1 lebel, chose the one that is more common in the training set as the result
      return(max(self.labels,key=self.labels.get))
      # return(random.choice(sorted(self.labels, key=self.labels.get, reverse=True)[:3]))
    else: return(max(score,key=score.get))
    # return(max(score,key=score.get))
    
def main():
    if len(sys.argv) < 2:
        print("Usage: %s dataset" %
              sys.argv[0])
        sys.exit(1)

    # notice that `dataset` will not have numerical +1/-1 labels, and
    # different reductions will require different transformations.
    dataset = (pickle.load(open(sys.argv[1], "rb")).
               convert_features_to_numerical())
    
    model = ova(10)
    model.ova_train(dataset)

    # note that this only evaluates accuracy! You'll need to write your own code
    # to compute the confusion matrix and report that separately
    evaluate_model(model, dataset)

if __name__ == '__main__':
    main()
