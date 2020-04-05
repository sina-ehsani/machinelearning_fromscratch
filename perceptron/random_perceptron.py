import random 
import sys
from perceptron import *


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

def main():
    
    if len(sys.argv) < 5:
        print("Usage: %s dataset number_of_itr_for_each_label k_random_range_s k_random_range_e" %
              sys.argv[0])
        sys.exit(1)

    dataset = (pickle.load(open(sys.argv[1], "rb")))

    for label in sorted(label_stats(dataset.training_set).keys()):
      dataset2 = dataset.convert_labels_to_numerical(label).convert_features_to_numerical()
      for i in range(int(sys.argv[2])):
        k = random.randrange(int(sys.argv[3]), int(sys.argv[4]))
        print ('\n\n\n------------label= ',label , 'k= ',k, '-----------------')
        model = train(dataset2, k)
        evaluate_model(model, dataset2)


if __name__ == '__main__':
    main()




