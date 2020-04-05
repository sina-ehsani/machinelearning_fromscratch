from utils import *
from data import *
import pickle
import sys
import random
import pylab
from transform import FeatureTransform
from perceptron import train
import numpy as np

##############################################################################

import matplotlib.pyplot as plt
def plotdata(dataset):
  x=[]
  y=[]
  for data in dataset.training_set:
    x.append(data.features)
    y.append(data.label)
  y = np.asanyarray(y)
  x = np.asanyarray(x)
  plt.scatter(x[:,0],x[:,1],c=y)
  plt.show()
  plt.scatter(x[:,0],x[:,2],c=y)
  plt.show()
  plt.scatter(x[:,1],x[:,2],c=y)
  plt.show()

class MysteryTransform(FeatureTransform):

    def transform_features(self, features):
        # implement this - `features` is a list of numbers representing
        # a vector to be transformed
        power =np.absolute(features)

        return power
        
def main():
    # notice that, for your convenience, this script hard-codes
    # the loading of "mystery-dataset.pickle", since you'll need
    # to hardcode the transformation in MysteryTransform above
    # anyway.
    if len(sys.argv) < 2:
        print("Usage: %s number_of_passes" %
              sys.argv[0])
        sys.exit(1)
    dataset = (pickle.load(open("mystery-dataset.pickle", "rb")).
               convert_labels_to_numerical("y").
               convert_features_to_numerical())
    # this is implemented in `transform.py`
    dataset = MysteryTransform().transform_dataset(dataset)
    print(dataset.training_set[10].features)
    plotdata(dataset)
    k = int(sys.argv[1])
    model = train(dataset, k)
    evaluate_model(model, dataset)
    
if __name__ == '__main__':
    main()
