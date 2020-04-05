from utils import *
import pickle
import sys
import math
import time


class kNNClassification(object):

    def __init__(self, training_set, k):
        self.training_set = training_set
        len_n=len(self.training_set)
        self.k = k

    def classify(self, instance):
        # implement this
        distances=list()
        for train_inst in self.training_set:
            distances.append((self.euclidist(train_inst.features,instance),train_inst))
        distances.sort(key = lambda x: x[0])  
        
        neighbors = list()
        for k in range(self.k):
            neighbors.append(distances[k][1])
        
        labels = [neighbor.label for neighbor in neighbors] 
        prediction = max(set(labels), key=labels.count)
        return prediction
            
        
    def euclidist(self,array_a,array_b):
        distance = 0.0
        for i, j in zip(array_a,array_b):
            distance += (i-j)**2
        return math.sqrt(distance)

#############################################################################

def train(labeled_samples, k):
    return kNNClassification(labeled_samples, k)

if __name__ == '__main__':
    dataset = pickle.load(open(sys.argv[1], "rb")).convert_to_numerical()
    k = int(sys.argv[2])

    model = train(dataset.training_set, k)

    start_time = time.time()
    evaluate_model(model, dataset)
    print("--- %s seconds ---" % (time.time() - start_time))
