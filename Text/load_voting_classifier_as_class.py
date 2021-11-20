from joblib import load
from scipy import sparse
import numpy as np
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


def test_preprocess(test_text_sample):  # todo: change the maxlen parameter!!!!
    df = pd.DataFrame({'Text': [test_text_sample]})
    pre_process = PreProcess(df)
    pre_process.pre_process_text()
    pre_process.Tfidf(is_train=False)
    pre_process.add_features()

    tokenizer = load_tokenizer("tokenizer.pickle")
    clean_text = pre_process.clean_text_df
    encoded_text = tokenizer.texts_to_sequences(clean_text)
    vectorized_text = pad_sequences(encoded_text, maxlen=22, padding='pre')

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

    def predict_proba(self, X_ML_test, X_Ada_test, X_DL_test):
        predictions_ = []
        for classifier in self.ML_classifiers:
            predict = classifier.predict_proba(X_ML_test)
            predictions_.append(predict)
        predict = self.AdaBoost.predict_proba(X_Ada_test)
        predictions_.append(predict)
        for classifier in self.DL_classifiers:
            predict = classifier.predict_proba(X_DL_test)
            predictions_.append(predict)

        y_pred = np.average(predictions_, axis=0, weights=self.weights)
        return y_pred


def test_voting_clf_on_text(text, voting_clf: FinalVotingClf):
    X_ML, X_AdaBoost, X_DL = test_preprocess(text)
    proba = voting_clf.predict_proba(X_ML, X_AdaBoost, X_DL)
    emotion_number = np.argmax(proba)
    print(voting_clf.emotions[emotion_number])
    # for prob in proba: # if sent multiple sentences to classify and not one
    #     emotion_number = np.argmax(prob)
    #     print(voting_clf.emotions[emotion_number])


def main():
    final_voting_clf = FinalVotingClf()
    sentence = "I feel so good about this project!"  # todo: get input from the user
    test_voting_clf_on_text(sentence, final_voting_clf)


if __name__ == '__main__':
    main()