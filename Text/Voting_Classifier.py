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
    """
    Calculates the weight of each base model in the voting classifier based on its average accuracy on the validaiton set.
    :param ML_acc: dictionary of the accuracies of the ML models (keys: models' names | values: avg accuracy on val set).
    :param DL_acc: dictionary of the accuracies of the DL models (keys: models' names | values: avg accuracy on val set).
    :return: list of weights.
    """
    accuracies = list(ML_acc.values())
    accuracies.extend(list(DL_acc.values()))
    total_score = sum(accuracies)
    weights = [weight / total_score for weight in accuracies]
    with open("voting_classifier_weights.txt", 'w') as myfile:
        myfile.write(', '.join(str(item) for item in weights) + '\n')
    return weights


def convert_y_values(predictions):
    """
    Convert the y values from probabilities to categorical.
    :param predictions: list of lists of probability predictions.
    :return:list of categorical predictions.
    """
    return [np.argmax(predictions[i], axis=0) for i in range(len(predictions))]


def get_accuracy(predicted_values, real_y):
    """
    Calculate the accuracy.
    :param predicted_values: list of predicted values.
    :param real_y: list of the real values
    :return: the accuracy of the prediction.
    """
    predicted_values = convert_y_values(predicted_values)
    num_hits = sum(1 for pred, true_val in zip(predicted_values, real_y) if
                   pred == true_val)
    return num_hits / float(len(real_y))

class EnsembleClassifier:
    """
    Class which holds all the base models and the needed parameters for the voting classifier. 
    """
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
        """
        Fit each base model on the respective train set
        :param X_ML_train: the X values of the ML train set
        :param X_Ada_train: the X values of the AdaBoost train set
        :param X_DL_train: the X values of the DL train set
        :param y_train: the labels of the samples in the train set.
        """
        for classifier in self.ML_classifiers:
            classifier.fit(X_ML_train, y_train)
        self.AdaBoost.fit(X_Ada_train, y_train)
        for classifier in self.DL_classifiers:
            classifier.fit(X_DL_train, y_train)
            
    def calc_acc_on_val(self, X_val_ML, X_val_AdaBoost, X_val_DL, y_val):
        """
        Calculate the accuracy of the voting classifier on the validation set by combining the predictions of each
        base model using weighted sum and check the voting classifier accuracy.
        :param X_val_ML: the X values of the ML validation set
        :param X_val_AdaBoost: the X values of the AdaBoost validation set
        :param X_val_DL: the X values of the DL validation set
        :param y_val: the labels of the samples in the validation set.
        :return: the voting classifier's accuracy on the validation set.
        """
        predictions_ = []
        for classifier in self.ML_classifiers:
            predict = classifier.predict_proba(X_val_ML)
            predictions_.append(predict)

        predict = self.AdaBoost.predict_proba(X_val_AdaBoost)
        predictions_.append(predict)

        predict = self.LSTM.predict_proba(X_val_DL)
        predictions_.append(predict)

        predict = self.CNN.predict_proba(X_val_DL)
        predictions_.append(predict)

        y_pred = np.average(predictions_, axis=0, weights=self.weights)
        acc = get_accuracy(y_pred, y_val)
        return acc
    
    def predict_proba_on_test(self, X_ML_test, X_Ada_test, X_DL_test, y_test):
        """
        Calculate the prediction of the voting classifier by combining the prediction of each 
        base model using weighted sum.
        Update the base_models_accuracies_on_test and the voting_clf_accuracy_on_test class variables. 
        :param X_ML_test:  the X values of the ML test set
        :param X_Ada_test:  the X values of the AdaBoost test set
        :param X_DL_test:  the X values of the DL test set
        :param y_test: the labels of the samples in the test set.
        :return: the voting classifer's prediction on the test set.
        """
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

    @staticmethod
    def create_multi_label_confusion_matrix(data, labels):
        """
        Create a multi label confusion matrix based on test set.
        :param data: sklearn.metrices' confusion matrix.
        :param labels: list of the emotions in a particular order.
        """
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
        """
        Plot a bar plot of the accuracy of each base model and the voting classifier accuracy on the test set.
        """
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
        plt.close()