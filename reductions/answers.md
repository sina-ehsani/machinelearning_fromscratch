**1. What are the accuracies you obtain for the primary-tumor dataset for AVA, OVA, and binary tree tournament?**

### OVA:
For the OVA we achieved the following:

It should be mentioned that, for these algorithms, we used two innovations:

1. If all the scores were -1 (meaning none of the classifiers were able to identify the label of the dataset), we return the label of the most frequent training label as the output.
2. If there were more than two scores of +1, meaning two or more classifiers assigned the data for their label, we returned the more common label between these positive labels (based on the training set).

Following is our result based on different k. (This classifier confidence range is wide since even if one classifier make mistake it will affect the classifier)

| **OVA**     |k=5   |  k=10| k=20 |k=200 |
| ----------- | -----| -----| -----| -----|
| Training    |43.20%|53.85%|53.85%|55.03%| 
| Validation: |38.82%|45.88%|41.18%|34.12%| 
| Testing:    |38.82%|47.06%|37.65%|37.65%| 
  

Based on this table, when K increases the model tends to overfit more in training data.  (however, it is very uncommon for this model to pass 60% accuracy for the training data)
  

### AVA:

For the AVA we achived the following (based on diffrent k):

| **AVA**     |k=5   |  k=10| k=20 |k=200 |
| ----------- | -----| -----| -----| -----|
| Training    |57.99%|69.23%|76.92%|84.02%| 
| Validation: |36.47%|44.71%|45.88%|44.71%| 
| Testing:    |37.65%|43.53%|37.65%|31.76%| 

Based on this table, when K increases the model tends to overfit more, compared to the OVA model. (this model is a bit more stable regarding the confidence level range)


### Binary Tree Tournament:

For the BTT we achived the following (based on diffrent k):

| **BTT**     |k=5   |  k=10| k=20 |k=200 |
| ----------- | -----| -----| -----| -----|
| Training    |40.83%|46.75%|47.34%|54.44%| 
| Validation: |34.12%|37.65%|35.29%|29.41%| 
| Testing:    |42.35%|40.00%|38.82%|29.41%| 

This model, compared to the other models, has less overfitting when k increases. Also, the results seem to be worst than the other two.


**2. Report the confusion matrix of these methods. Are they comparable?**

For these reports, we used k=10 for all models, and we used the models that we're able to achieve higher than 40% on testing:

### OVA:
Here is the OVA with k=10 confusion matrix for the test set. (47.06% accuracy on test)

![](/plots/ova10.png "OVA , k=10, a=47.06%")

As you can see our innovation for the OVA (mentioned in question 1) led to having more miss classification for the most common label ('1' in this dataset). However, it also helped to increase the overall accuracy and the True Positive and Recall for this label. 

### AVA:
And the following is the AVA with k=10 confusion matrix for the test set. (43.53% accuracy on test)

![](/plots/ava10.png "AVA , k=10, a=43.53%")

### Binary Tree Tournament:

BTT with k=10 confusion matrix for the test set. (40.00% accuracy on test)

![](/plots/btt10.png "Binary Tree Tournament , k=10, a=40.00%")

This matrix is kind of similar to the one achieved with the AVA.


**3. Given the semantics of the labels for the primary-tumor dataset, is “accuracy” a good measure of model quality? If not, what are the problems and possible alternatives?**

Since the data is not evenly between the classes the accuracy might not be a good metric. You can see the data distribution in the following:

![](/plots/labels.png "Training set labels histogram")
![](/plots/label_val.png "Validation set labels histogram")
![](/plots/label_test.png "Test set labels histogram")

In addition, as you can see from the above histogram, the distribution for the training, dev, and test sets dose not seem to be the same. Some labels that are in the test/dev sets do not even exist in the training set. These all result in poor classification performance (especially when overfitting too much on the training set).


It should be also mentioned since this data is hospital related data, having a good precision or recall based on the type of tumor is important. You do not want to missclasify someone that has a tumor as a healthy person, and also you want to make sure that you do not stress people by giving a false alarm. Therefore in this case having a good precision-recall ratio (or f1) would be better than accuracy. 

One way that we can improve this for the OVA model, is when the model cannot decide on the classification (predicts all -1, or has two or more +1) we can give weight to the more critical classification as the tie breaker.
