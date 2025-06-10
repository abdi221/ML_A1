from abc import ABC, abstractmethod
import numpy as np
from sklearn import datasets as ds
from collections import Counter
import math
class MachineLearningModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, y_true, y_predicted):
        pass

def euclidean_distance(point_x, point_y):
    distance = 0.0
    for i in range(len(point_x)):
        distance += (point_x[i]-point_y[i]) ** 2
    return math.sqrt(distance)

class KNNRegressionModel(MachineLearningModel):
    """
    Class for KNN regression model.
    """

    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X 
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            dists = [euclidean_distance(x, x_i) for x_i in self.X_train]
            #find indicies of the k nearest
            k_nearest = np.argsort(dists)[:self.k]
            #avgerage their y
            preds = np.mean(self.y_train[k_nearest])
            predictions.append(preds)
        return np.array(predictions)

    def evaluate(self, y_true, y_predicted):
        return np.mean((y_true - y_predicted) ** 2)
    

class KNNClassificationModel(MachineLearningModel):
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            dists = [euclidean_distance(x, x_i) for x_i in self.X_train]
            k_nearest = np.argsort(dists)[:self.k]
            #majority vote
            labels = self.y_train[k_nearest]
            most_common = Counter(labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

    def evaluate(self, y_true, y_predicted):
        return np.sum(y_true == y_predicted)
