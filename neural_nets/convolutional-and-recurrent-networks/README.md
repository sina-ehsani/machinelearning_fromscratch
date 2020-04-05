# Objectives

1. learn the Keras APIs for convolutional and recurrent neural networks.
2. explore the space of hyper-parameters for convolutional and recurrent
   networks.

# Setup your environment


* [git](https://git-scm.com/downloads)
* [Python (version 3.7 or higher)](https://www.python.org/downloads/)
* [Keras (version 2.1 or higher)](https://keras.io/)
* [numpy](http://www.numpy.org/)
* [h5py](https://www.h5py.org/)
* [pytest](https://docs.pytest.org/)


# Test your code

The tests show how each of the methods is expected to be used.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.

you should see something like:
```
============================= test session starts ==============================
platform darwin -- Python 3.7.4, pytest-5.0.1, py-1.8.0, pluggy-0.12.0
rootdir: .../convolutional-and-recurrent-<your-username>
plugins: timeout-1.3.3
collected 4 items

test_nn.py
1.3 RMSE for RNN on toy problem
.
87.0% accuracy for CNN on MNIST sample
.
88.9% accuracy for RNN on Youtube comments
.
83.2% accuracy for CNN on Youtube comments
.                                                          [100%]

========================== 4 passed in 39.93 seconds ===========================
```
