import pandas as pd
import tensorflow as tf
import numpy as np
from math import log 
from sklearn.model_selection import train_test_split

class AdaBoost:
    def __init__(self, training_data, L, M):
        self.training_data = training_data.copy()
        self.M = M
        self.w = [1.0 / len(training_data)]
        self.h = [None] * M
        self.z = [0.0] * M
        self.L = L.copy()