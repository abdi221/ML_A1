from abc import ABC, abstractmethod
import numpy as np
from sklearn import datasets as ds
from collections import Counter
import math
class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, y_true, y_predicted):
        """
        Evaluate the model on the given data.

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
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
        """
        Initialize the model with the specified instructions.

        Parameters:
        k (int): Number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Train the model using the given training data.
        In this case, the training data is stored for later use in the prediction step.
        The model does not need to learn anything from the training data, as KNN is a lazy learner.
        The training data is stored in the class instance for later use in the prediction step.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        self.X_train = X 
        self.y_train = y

    def predict(self, X):
        """
        Make predictions on new data.
        The predictions are made by averaging the target variable of the k nearest neighbors.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
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
        """
        Evaluate the model on the given data.
        You must implement this method to calculate the Mean Squared Error (MSE) between the true and predicted values.
        The MSE is calculated as the average of the squared differences between the true and predicted values.        

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        #--- Write your code here ---#
        return np.mean((y_true - y_predicted) ** 2)
    

class KNNClassificationModel(MachineLearningModel):
    """
    Class for KNN classification model.
    """

    def __init__(self, k):
        """
        Initialize the model with the specified instructions.

        Parameters:
        k (int): Number of neighbors.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Train the model using the given training data.
        In this case, the training data is stored for later use in the prediction step.
        The model does not need to learn anything from the training data, as KNN is a lazy learner.
        The training data is stored in the class instance for later use in the prediction step.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Make predictions on new data.
        The predictions are made by taking the mode (majority) of the target variable of the k nearest neighbors.
        
        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        #--- Write your code here ---#
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
        """
        Evaluate the model on the given data.
        You must implement this method to calculate the total number of correct predictions only.
        Do not use any other evaluation metric.

        Parameters:
        y_true (array-like): True target variable of the data.
        y_predicted (array-like): Predicted target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        #--- Write your code here ---#
        return np.sum(y_true == y_predicted)
