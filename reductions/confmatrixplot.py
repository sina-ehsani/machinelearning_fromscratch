import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import ova

def canf_matrix(model,sample_set):
  y_predict=[]
  y_true=[]
  for i, sample in enumerate(sample_set):
    y_predict.append(int(model.classify(sample.features)))
    y_true.append(int(sample.label))
  return y_predict , y_true





def main():

	dataset = (pickle.load(open(sys.argv[1], "rb")).
	           convert_features_to_numerical())

	model = ova.ova(200)
	model.ova_train(dataset)

	evaluate_model(model, dataset)

	y_predict , y_true = canf_matrix(model,dataset.testing_set)
	cm = confusion_matrix(y_target=y_true, 
	                      y_predicted=y_predict, 
	                      binary=False)

	fig, ax = plot_confusion_matrix(conf_mat=cm)
	plt.show()


    
if __name__ == '__main__':
    main()
