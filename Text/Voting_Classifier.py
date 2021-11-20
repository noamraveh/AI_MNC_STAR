from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from DL_Tuning import LSTM_model, CNN_model
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns


def set_weights(ML_acc, DL_acc):
    accuracies = list(ML_acc.values())
    accuracies.extend(list(DL_acc.values()))
    total_score = sum(accuracies)
    weights = [weight / total_score for weight in accuracies]
    with open("voting_classifier_weights.txt", 'w') as myfile:
        myfile.write(', '.join(str(item) for item in weights) + '\n')
    return weights


def convert_y_values(predictions):
    return [np.argmax(predictions[i], axis=0) for i in range(len(predictions))]


def get_accuracy(predicted_values, real_y):
    predicted_values = convert_y_values(predicted_values)
    num_hits = sum(1 for pred, true_val in zip(predicted_values, real_y) if
                   pred == true_val)
    return num_hits / float(len(real_y))

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
        self.emotions = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
        self.base_models_accuracies_on_test = None
        self.voting_clf_accuracy_on_test = None

    def fit(self, X_ML_train, X_Ada_train, X_DL_train, y_train):
        for classifier in self.ML_classifiers:
            classifier.fit(X_ML_train, y_train)
        self.AdaBoost.fit(X_Ada_train, y_train)
        for classifier in self.DL_classifiers:
            classifier.fit(X_DL_train, y_train)

    def predict_proba_on_test(self, X_ML_test, X_Ada_test, X_DL_test, y_test):
        predictions_ = []
        accuracies = []
        for classifier in self.ML_classifiers:
            dump(classifier, f'{classifier.__class__.__name__}.joblib')
            predict = classifier.predict_proba(X_ML_test)
            predictions_.append(predict)
            accuracies.append(get_accuracy(predict, y_test))

        dump(self.AdaBoost, f'{self.AdaBoost.__class__.__name__}.joblib')
        predict = self.AdaBoost.predict_proba(X_Ada_test)
        predictions_.append(predict)
        accuracies.append(get_accuracy(predict, y_test))

        self.LSTM.model.save('LSTM_model.h5')
        predict = self.LSTM.predict_proba(X_DL_test)
        predictions_.append(predict)
        accuracies.append(get_accuracy(predict, y_test))

        self.CNN.model.save('CNN_model.h5')
        predict = self.CNN.predict_proba(X_DL_test)
        predictions_.append(predict)
        accuracies.append(get_accuracy(predict, y_test))

        self.base_models_accuracies_on_test = accuracies

        y_pred = np.average(predictions_, axis=0, weights=self.weights)
        self.voting_clf_accuracy_on_test = get_accuracy(y_pred, y_test)

        return y_pred

    def create_multi_label_confusion_matrix(self, data, labels):
        sns.set(color_codes=True)
        plt.figure(1, figsize=(9, 6))

        plt.title("Final Model - Confusion Matrix")

        sns.set(font_scale=1.4)
        ax = sns.heatmap(data, annot=True, cbar_kws={'label': 'Scale'})

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.set(ylabel="True Label", xlabel="Predicted Label")

        plt.savefig("confusion_matrix.png")
        #plt.show()
        plt.close()

    def plot_accuracies(self):
        x = ['ComplementNB', 'LinearSVM', 'LogisticRegression', 'AdaBoost', 'LSTM', 'CNN', 'Voting Classifier']
        y = self.base_models_accuracies_on_test
        y.append(self.voting_clf_accuracy_on_test)

        sns.set(color_codes=True)
        plt.figure(1, figsize=(12, 9))
        plt.title("Models Accuracies")
        ax = sns.barplot(x=x, y=y)
        ax.set(ylabel="Accuracy", xlabel="Model")
        ax.xaxis.set_ticklabels(x)
        plt.savefig("accuracies.png", bbox_inches='tight', dpi=400)
        #plt.show()