from joblib import load
from scipy import sparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import load_model
from preprocess_text import PreProcess
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import pickle


def load_tokenizer(filename):
    with open(filename, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def convert_y_values(predictions):
    return [np.argmax(predictions[i], axis=0) for i in range(len(predictions))]


def get_accuracy(predicted_values, real_y):
    predicted_values = convert_y_values(predicted_values)
    num_hits = sum(1 for pred, true_val in zip(predicted_values, real_y) if
                   pred == true_val)
    return num_hits / float(len(real_y))


def test_preprocess(test_text_sample): #todo: change the maxlen parameter!!!!
    df = pd.DataFrame({'Text': [test_text_sample]})
    pre_process = PreProcess(df)
    pre_process.pre_process_text()
    pre_process.Tfidf(is_train=False)
    pre_process.add_features()

    tokenizer = load_tokenizer("tokenizer.pickle")
    clean_text = pre_process.clean_text_df
    encoded_text = tokenizer.texts_to_sequences(clean_text)
    vectorized_text = pad_sequences(encoded_text, maxlen=32, padding='pre')

    X_ML = pre_process.X
    X_AdaBoost = pre_process.added_features
    X_DL = vectorized_text
    return X_ML, X_AdaBoost, X_DL



class FinalVotingClf:
    def __init__(self):
        self.ComplementNB = load('ComplementNB.joblib')
        self.LinearSVM = load('CalibratedClassifierCV.joblib')
        self.LogisticRegression = load('LogisticRegression.joblib')
        self.AdaBoost = load('AdaBoostClassifier.joblib')
        self.LSTM = load_model('LSTM_model.h5')
        self.CNN = load_model('CNN_model.h5')
        with open("voting_classifier_weights.txt", 'r') as myfile:
            self.weights = [float(x) for x in myfile.readline().split(',')]
        self.basic_classifiers = [self.ComplementNB, self.LinearSVM, self.LogisticRegression, self.AdaBoost, self.LSTM, self.CNN]
        self.ML_classifiers = self.basic_classifiers[:3]
        self.DL_classifiers = self.basic_classifiers[4:]
        self.emotions = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
        self.voting_clf_accuracy = None
        self.models_accuracies = []

    def create_multi_label_confusion_matrix(self, real_y, predicted_y):
        predicted_y = convert_y_values(predicted_y)
        cm = confusion_matrix(real_y, predicted_y, normalize='true')
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            cm[i][j] = "{:0.2f}".format(cm[i, j])
        ax = sns.heatmap(cm, annot=True)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Emotions')
        ax.set_ylabel('Actual Emotions')
        ax.xaxis.set_ticklabels(self.emotions)
        ax.yaxis.set_ticklabels(self.emotions)
        plt.show()

    def plot_accuracies(self):
        x = ['ComplementNB', 'LinearSVM', 'LogisticRegression', 'AdaBoost', 'LSTM', 'CNN', 'Voting Classifier']
        y = self.models_accuracies
        y.append(self.voting_clf_accuracy)
        ax = sns.barplot(x=x, y=y)
        ax.set_ylabel('Accuracies')
        ax.set_title('Models Accuracies')
        ax.xaxis.set_ticklabels(x)
        plt.show()

    def predict_proba(self, X_ML_test, X_Ada_test, X_DL_test, y_test=None, is_test=False):
        predictions_ = []
        accuracies = []
        for classifier in self.ML_classifiers:
            predict = classifier.predict_proba(X_ML_test)
            predictions_.append(predict)
            if is_test:
                accuracies.append(get_accuracy(predict, y_test))
        predict = self.AdaBoost.predict_proba(X_Ada_test)
        predictions_.append(predict)
        if is_test:
            accuracies.append(get_accuracy(predict, y_test))
        for classifier in self.DL_classifiers:
            predict = classifier.predict_proba(X_DL_test)
            predictions_.append(predict)
            if is_test:
                accuracies.append(get_accuracy(predict, y_test))
        self.models_accuracies = accuracies
        y_pred = np.average(predictions_, axis=0, weights=self.weights)
        if is_test:
            self.voting_clf_accuracy_on_test = get_accuracy(y_pred, y_test)
        return y_pred


def test_voting_clf_on_text(voting_clf: FinalVotingClf):
    text = input() # todo: get an iput from the user in jupyter
    X_ML, X_AdaBoost, X_DL = test_preprocess(text)
    proba = voting_clf.predict_proba(X_ML, X_AdaBoost, X_DL)
    emotion_number = np.argmax(proba)
    print(voting_clf.emotions[emotion_number])
    # for prob in proba: # if sent multiple sentences to classify and not one
    #     emotion_number = np.argmax(prob)
    #     print(voting_clf.emotions[emotion_number])


def main():
    X_ML = sparse.load_npz("X_test_ML.npz")
    X_DL = np.loadtxt("X_test_DL", delimiter=',')
    X_AdaBoost = np.loadtxt("X_test_AdaBoost", delimiter=',')
    y_real= np.loadtxt("y_test_ML", delimiter=',')

    final_voting_clf = FinalVotingClf()
    y_pred = final_voting_clf.predict_proba(X_ML, X_AdaBoost, X_DL, y_real, is_test=True)

    # convert y values from one hot dictionary to a value (0-6)
    final_voting_clf.create_multi_label_confusion_matrix(y_real, y_pred)
    final_voting_clf.plot_accuracies()


if __name__ == '__main__':
    main()