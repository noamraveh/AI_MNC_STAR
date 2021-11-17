from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from DL_Tuning import LSTM_model, CNN_model
import numpy as np
import pandas as pd
from joblib import dump


def set_weights(ML_acc, DL_acc):
    accuracies = list(ML_acc.values())
    accuracies.extend(list(DL_acc.values()))
    # accuracies = list(ML_acc.values()) + list(DL_acc.values())
    total_score = sum(accuracies)
    weights = [weight / total_score for weight in accuracies]
    with open("voting_classifier_weights.txt", 'w') as myfile:
        myfile.write(', '.join(str(item) for item in weights) + '\n')
    return weights


class EnsembleClassifier():
    def __init__(self, ML_models_params, ML_models_acc, DL_models_params, DL_models_acc, vocab_size, vector_size, input_length):
        self.ComplementNB = ComplementNB(**ML_models_params['NB'])
        self.LinearSVM = CalibratedClassifierCV(
            base_estimator=SGDClassifier(**ML_models_params['SVM'], class_weight='balanced'))
        self.LogisticRegression = LogisticRegression(**ML_models_params["LR"])
        self.AdaBoost = AdaBoostClassifier(**ML_models_params["AdaBoost"])
        LSTM = LSTM_model(vocab_size, vector_size, input_length)
        self.LSTM = tf.keras.wrappers.scikit_learn.KerasClassifier(LSTM.build_network, **DL_models_params['LSTM'])
        self.LSTM._estimator_type = "classifier"
        CNN = CNN_model(vocab_size, vector_size, input_length)
        self.CNN = tf.keras.wrappers.scikit_learn.KerasClassifier(CNN.build_network, **DL_models_params['CNN'])
        self.CNN._estimator_type = "classifier"
        self.classifiers = [self.ComplementNB, self.LinearSVM, self.LogisticRegression, self.AdaBoost, self.LSTM,
                            self.CNN]
        self.ML_classifiers = self.classifiers[:3]
        self.DL_classifiers = self.classifiers[4:]
        self.weights = set_weights(ML_models_acc, DL_models_acc)

    def fit(self, X_ML_train, X_Ada_train, X_DL_train, y_train):
        for classifier in self.ML_classifiers:
            classifier.fit(X_ML_train, y_train)
        self.AdaBoost.fit(X_Ada_train, y_train)
        for classifier in self.DL_classifiers:
            classifier.fit(X_DL_train, y_train)

    def predict_proba(self, X_ML_val, X_Ada_val, X_DL_val):
        self.predictions_ = []
        for classifier in self.ML_classifiers:
            self.predictions_.append(classifier.predict_proba(X_ML_val))
            dump(classifier, f'{classifier.__class__.__name__}.joblib')
        self.predictions_.append(self.AdaBoost.predict_proba(X_Ada_val))
        dump(self.AdaBoost, f'{self.AdaBoost.__class__.__name__}.joblib')
        self.predictions_.append(self.LSTM.predict_proba(X_DL_val))
        self.LSTM.model.save('LSTM_model.h5')
        self.predictions_.append(self.CNN.predict_proba(X_DL_val))
        self.CNN.model.save('CNN_model.h5')
        # for classifier in self.DL_classifiers:
        #     self.predictions_.append(classifier.predict_proba(X_DL_val))

        return np.average(self.predictions_, axis=0, weights=self.weights)
