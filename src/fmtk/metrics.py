import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score

def get_mae(y_test, y_pred):
    if len(y_test.shape) > 2:
        y_test = y_test.reshape(-1, y_test.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    return mean_absolute_error(y_test, y_pred)

def get_accuracy(y_test, y_pred):
    def normalize(x):
        if isinstance(x, str):
            return x.lower()
        return x

    y_test = [normalize(y) for y in y_test]
    y_pred = [normalize(y) for y in y_pred]

    return accuracy_score(y_test, y_pred)