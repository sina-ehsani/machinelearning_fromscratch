
For the primary-tumor.pickle dataset the following were achived (for depth 3):

  Training:    79/169: 46.75% 
  Validation:  35/85: 41.18% 
  Testing:     34/85: 40.00%
  
 
And for the agaricus-lepiota with a depth of 3 the following:

  Training:    4056/4062: 99.85%
  Validation:  2022/2031: 99.56%
  Testing:     2022/2031: 99.56%
  
  There are two reasons two say that the agaricus-lepiota is performing better:
  
  1. It has more data, which enables the model to learn more.
  
  2. Since it only has two labels it easier to perform better results. (binary classification is easier than multilabel classification)


The id3 seems not to work as well as the basid. It seems to overfit on the training better, but decreases the test performance.
