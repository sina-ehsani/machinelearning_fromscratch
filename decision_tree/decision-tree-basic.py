import pickle
import sys
# from utils import *

def accuracy(model, sample_set):
    accuracy = 0
    for sample in sample_set:
        # print (model._predict(sample) , sample.label)
        if model._predict(sample) == sample.label:
            accuracy += 1
    return accuracy

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def _best_split(self, dataset):

        self.score_dict=dict()
        for feat in self.features_.keys():
            self.score_dict[feat]=0
            for value in self.features_[feat]:
                sub_class_dict=dict()
                for datapoint in dataset:
                     if datapoint.features[feat]==value:
                            if datapoint.label in sub_class_dict.keys():
                                sub_class_dict[datapoint.label]+=1
                            else: sub_class_dict[datapoint.label]=1
                if sub_class_dict:
                    # print(sub_class_dict)
                    highest_value = sub_class_dict[self.guess(sub_class_dict)]
                    # print('highest_value',highest_value)
                else: highest_value = 0
                self.score_dict[feat] += int(highest_value)
        best_idx =max(self.score_dict, key=self.score_dict.get)
        
        return best_idx
    
    
    def fit(self, dataset):
        """Build decision tree classifier."""
        self.n_classes_ = self.class_size(dataset)  # c
        self.features_=dataset.metadata.feature_values
        self.n_features_ = len(self.features_)
        self.num_splits_ = max([len(value) for key , value in dataset.metadata.feature_values.items()])

        self.tree_ = self._grow_tree(dataset.training_set)
        return self.tree_
        
    def _grow_tree(self, dataset, depth=0):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with
        # largest population.
        num_samples_per_class = self.label_stats(dataset)
        predicted_class = self.guess(num_samples_per_class)
        
        node = Node(
            num_samples=len(dataset),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
            num_splits=self.num_splits_,
            
        )
        
        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            idx = self._best_split(dataset)
            if idx is not None:
                spilit_dict=self.split_data(dataset,idx)
                node.feature_index = idx
                for key , value in spilit_dict.items():
                    node.spilit[key] = self._grow_tree(value, depth + 1)
        return node    
    
    def split_data(self, dataset,idx):
        split_dict=dict()
        for datapoint in dataset:
            for value in self.features_[idx]:
                if datapoint.features[idx] == value:
                    if value in split_dict:
                        split_dict[value].append(datapoint)
                    else: 
                        split_dict[value]=[datapoint]
        return split_dict
    
    
    
    def label_stats(self,dataset):
        '''returns number of each label in dataset
        '5': 25,
        '11': 14,'''
        class_dict=dict()
        for datapoint in dataset:
            class_dict = self.quant_label(datapoint , class_dict)
        return class_dict 


    def quant_label(self ,datapoint , quant_dict):
        '''update label_stats based on the datapoint'''
        if datapoint.label in quant_dict.keys():
            quant_dict[datapoint.label]+=1
        else: quant_dict[datapoint.label]=1
        return quant_dict
        
    def class_size(self,dataset):
        a=[]
        for data in dataset.training_set:
            a.append(data.label)
        return len(set(a))
    
    def guess(self, part_dict):
        return max(part_dict, key=part_dict.get)
    
    
    
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.spilit:
            if inputs.features[node.feature_index] in node.spilit.keys():
                node = node.spilit[inputs.features[node.feature_index]]
            else: return node.predicted_class
        return node.predicted_class

class Node:
    def __init__(self, num_samples, num_samples_per_class, predicted_class, num_splits):
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
#         splits
        self.spilit=dict()
#         for num in range (num_splits):
#             self.spilit[num]=None       


if __name__ == '__main__':
    # read command-line parameters
    dataset = pickle.load(open(sys.argv[1], "rb"))
    max_depth = int(sys.argv[2])
    print("Training...")
    
    clf = DecisionTreeClassifier(max_depth)
    clf.fit(dataset)

    print("Training complete.")

    tr_n = len(dataset.training_set)
    va_n = len(dataset.validation_set)
    te_n = len(dataset.testing_set)
    

    print("\nEvaluating...")
    tr_acc = accuracy(clf, dataset.training_set)
    va_acc = accuracy(clf, dataset.validation_set)
    te_acc = accuracy(clf, dataset.testing_set)
    print("Evaluation complete:")
    print("  Training:    %s/%s: %.2f%%" % (tr_acc, tr_n, 100 * tr_acc / tr_n))
    print("  Validation:  %s/%s: %.2f%%" % (va_acc, va_n, 100 * va_acc / va_n))
    print("  Testing:     %s/%s: %.2f%%" % (te_acc, te_n, 100 * te_acc / te_n))


