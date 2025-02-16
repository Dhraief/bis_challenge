import numpy as np
from sklearn.svm import OneClassSVM

def run_one_class_svm(data, nu=0.05, kernel="rbf", gamma="scale"):
    """
    Run One-Class SVM to detect anomalies.
    :param data: NumPy array of shape (n_samples, n_features)
    :param nu: An upper bound on the fraction of training errors.
    :param kernel: Kernel type.
    :param gamma: Kernel coefficient.
    :return: anomaly predictions (1 for normal, -1 for anomaly) and decision function values.
    """
    ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    preds = ocsvm.fit_predict(data)
    scores = ocsvm.decision_function(data)
    return preds, scores
