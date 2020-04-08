from utils import *
from data import *

##############################################################################

class L2Regularization(object):
    def __init__(self, l):
        self.l = l
    def loss(self, weight_vector):
        return (self.l/2)*dot(weight_vector,weight_vector)
    def gradient(self, weight_vector):
        return  list(map(lambda x: x * self.l, weight_vector))
        