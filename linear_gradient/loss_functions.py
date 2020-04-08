from utils import *
from data import *
import math

##############################################################################

class L2RegressionLoss(object):
    def loss(self, prediction, sample):
        # WRITE THIS CODE
        return( 0.5*pow(sample.label-prediction,2) )
    def gradient(self, prediction, sample):
        # WRITE THIS CODE
        # return(2 * sample.features * (prediction - sample.label))
        return list(map( lambda x : x * (prediction - sample.label) , sample.features ))

class LogisticLoss(object):
    def loss(self, prediction, sample):
        return (1/math.log(2))*(1+math.exp(-prediction*sample.label))
        # WRITE THIS CODE
    def gradient(self, prediction, sample):
        bottom=math.log(2)* (1+math.exp(prediction*sample.label))
        zarib = -sample.label/bottom
        return list(map( lambda x : x* zarib ,sample.features ))
        # WRITE THIS CODE


class HingeLoss(object):
    def loss(self, prediction, sample):
        return max(0 , 1-prediction*sample.label )
        # WRITE THIS CODE
    def gradient(self, prediction, sample):
        # WRITE THIS CODE
        # smoothed version
        y=prediction
        t=sample.label
        if t*y >= 1:
            return list(map(lambda x : x * 0 ,sample.features ))
        else: return list(map(lambda x : -x * prediction ,sample.features ))

