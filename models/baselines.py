from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np 




class SVM():
    def __init__(self, X, y, **kwargs):
        self.clf = SVC(**kwargs)
        self.X = X
        self.y = y

    def fit(self, train_mask):
        try:
            train_mask = train_mask.detach().numpy()
        except AttributeError:
            train_mask = np.array(train_mask)

        self.clf.fit(self.X[train_mask], self.y[train_mask])

    def predict(self, test_mask):
        try:
            test_mask = test_mask.detach().numpy()
        except AttributeError:
            test_mask = np.array(test_mask)

        return self.clf.predict(self.X[test_mask])


        
class RandomForest():
    def __init__(self, X,y, **kwargs):
        self.clf = RandomForestClassifier(**kwargs)
        self.X = X
        self.y = y

    def fit(self, train_mask):
        try:
            train_mask = train_mask.detach().numpy()
        except AttributeError:
            train_mask = np.array(train_mask)

        self.clf.fit(self.X[train_mask], self.y[train_mask])

    def predict(self, test_mask):
        try:
            test_mask = test_mask.detach().numpy()
        except AttributeError:
            test_mask = np.array(test_mask)

        return self.clf.predict(self.X[test_mask])


class MLP():
    def __init__(self, X,y, **kwargs):
        self.clf = MLPClassifier(**kwargs)
        self.X = X
        self.y = y

    def fit(self, train_mask):
        try:
            train_mask = train_mask.detach().numpy()
        except AttributeError:
            train_mask = np.array(train_mask)

        self.clf.fit(self.X[train_mask], self.y[train_mask])

    def predict(self, test_mask):
        try:
            test_mask = test_mask.detach().numpy()
        except AttributeError:
            test_mask = np.array(test_mask)

        return self.clf.predict(self.X[test_mask])
