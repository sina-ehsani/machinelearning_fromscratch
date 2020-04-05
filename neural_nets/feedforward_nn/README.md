# Objectives

1. get familiar with the Keras framework for training neural networks.
2. experiment with the various hyper-parameter choices of feedforward networks.

# Setup your environment

* [git](https://git-scm.com/downloads)
* [Python (version 3.7 or higher)](https://www.python.org/downloads/)
* [Keras (version 2.1 or higher)](https://keras.io/)
* [numpy](http://www.numpy.org/)
* [h5py](https://www.h5py.org/)
* [pytest](https://docs.pytest.org/)


# Test your code

Tests have been provided in the `test_nn.py` file.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.

```
============================= test session starts ==============================
platform darwin -- Python 3.7.4, pytest-5.0.1, py-1.8.0, pluggy-0.12.0
rootdir: .../feedforward-networks-<your-username>
plugins: timeout-1.3.3
collected 4 items

test_nn.py
8.2 RMSE for baseline on Auto MPG
6.7 RMSE for deep on Auto MPG
4.7 RMSE for wide on Auto MPG
.
65.0% accuracy for baseline on del.icio.us
68.6% accuracy for relu on del.icio.us
66.8% accuracy for tanh on del.icio.us
.
18.2% accuracy for baseline on UCI-HAR
90.0% accuracy for dropout on UCI-HAR
86.8% accuracy for no dropout on UCI-HAR
.
75.4% accuracy for baseline on census income
77.8% accuracy for early on census income
76.5% accuracy for late on census income
.                                                          [100%]

========================== 4 passed in 22.03 seconds ===========================
```
**Warning**: The performance of your models may change somewhat from run to run,
since neural network models are randomly initialized.
A correct solution should pass the tests almost all of the time, but due to randomness may occasionally fail.

