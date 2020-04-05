from utils import *
import pickle
import sys
import random
import pylab
import numpy as np

##############################################################################

class Perceptron(object):

    def __init__(self, dim):
        self.weights = [0] * dim
        self.bias = 0

    def predict(self, features):
      activation =  np.dot(self.weights , features) + self.bias
      return activation
        # implement this
        
    def classify(self, features):
        # implement this
        prediction = self.predict(features)
        return sign(prediction)

    # NB: this assumes that the label for labeled_sample is -1 or +1!
    def update(self, labeled_sample):
      activation = np.dot(self.weights , labeled_sample.features) + self.bias
      if activation*labeled_sample.label <= 0:
        self.weights += np.dot(labeled_sample.label,labeled_sample.features)
        self.bias += labeled_sample.label
        # implement this

##############################################################################

def train(dataset, k):
    try: labeled_samples = dataset.training_set.copy()
    except: labeled_samples=dataset
    model = Perceptron(len(labeled_samples[0].features))
    for i in range(k):
        random.shuffle(labeled_samples)
        for sample in labeled_samples:
            model.update(sample)
    return model
        
def main():
    if len(sys.argv) < 4:
        print("Usage: %s dataset positive_label number_of_passes" %
              sys.argv[0])
        sys.exit(1)
    dataset = (pickle.load(open(sys.argv[1], "rb")).
               convert_labels_to_numerical(sys.argv[2]).
               convert_features_to_numerical())

    k = int(sys.argv[3])
    model = train(dataset, k)
    evaluate_model(model, dataset)
    
if __name__ == '__main__':
    main()
