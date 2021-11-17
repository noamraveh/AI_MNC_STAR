from joblib import load
from scipy import sparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import load_model


def convert_y_values(predictions):
    return [np.argmax(predictions[i], axis=0) for i in range(len(predictions))]


def create_multi_label_confusion_matrix(real_y, predicted_y, emotions):
    predicted_y = convert_y_values(predicted_y)
    cm = confusion_matrix(real_y, predicted_y, normalize='true')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cm[i][j] = "{:0.2f}".format(cm[i, j])
    ax = sns.heatmap(cm, annot=True)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Emotions')
    ax.set_ylabel('Actual Emotions')
    ax.xaxis.set_ticklabels(emotions)
    ax.yaxis.set_ticklabels(emotions)
    plt.show()


def get_accuracy(predicted_values, real_y):
    predicted_values = convert_y_values(predicted_values)
    num_hits = sum(1 for pred, true_val in zip(predicted_values, real_y) if
                   pred == true_val)
    return num_hits / float(len(real_y))


def plot_accuracies(accuracies):
    x = ['ComplementNB', 'LinearSVM', 'LogisticRegression', 'AdaBoost', 'LSTM', 'CNN', 'Voting Classifier']
    ax = sns.barplot(x=x, y=accuracies)
    ax.set_ylabel('Accuracies')
    ax.set_title('Models Accuracies')
    ax.xaxis.set_ticklabels(x)
    plt.show()


def predict_proba_voting_classifier(ML_classifiers, DL_classifiers, AdaBoost, weights, X_ML_val, X_Ada_val, X_DL_val, y):
    predictions_ = []
    accuracies = []
    for classifier in ML_classifiers:
        predict = classifier.predict_proba(X_ML_val)
        predictions_.append(predict)
        accuracies.append(get_accuracy(predict, y))
    predict = AdaBoost.predict_proba(X_Ada_val)
    predictions_.append(predict)
    accuracies.append(get_accuracy(predict, y))
    for classifier in DL_classifiers:
        predict = classifier.predict_proba(X_DL_val)
        predictions_.append(predict)
        accuracies.append(get_accuracy(predict, y))
    return accuracies, np.average(predictions_, axis=0, weights=weights)


def main():
    X_ML = sparse.load_npz("X_test_ML.npz")
    X_DL = np.loadtxt("X_test_DL", delimiter=',')
    X_AdaBoost = np.loadtxt("X_test_AdaBoost", delimiter=',')
    y = np.loadtxt("y_test_ML", delimiter=',')

    ComplementNB = load('ComplementNB.joblib')
    LinearSVM = load('CalibratedClassifierCV.joblib')
    LogisticRegression = load('LogisticRegression.joblib')
    AdaBoost = load('AdaBoostClassifier.joblib')
    LSTM = load_model('LSTM_model.h5')
    CNN = load_model('CNN_model.h5')
    with open("voting_classifier_weights.txt", 'r') as myfile:
        weights = [float(x) for x in myfile.readline().split(',')]
    classifiers = [ComplementNB, LinearSVM, LogisticRegression, AdaBoost, LSTM, CNN]
    ML_classifiers = classifiers[:3]
    DL_classifiers = classifiers[4:]

    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    models_accuracies, predicted_values_voting_classifier = predict_proba_voting_classifier(ML_classifiers, DL_classifiers, AdaBoost,
                                                                                            weights, X_ML, X_AdaBoost, X_DL, y)
    # convert y values from one hot dictionary to a value (0-6)
    create_multi_label_confusion_matrix(y, predicted_values_voting_classifier, emotions)
    voting_classifier_accuracy = get_accuracy(predicted_values_voting_classifier, y)
    models_accuracies.append(voting_classifier_accuracy)
    plot_accuracies(models_accuracies)


if __name__ == '__main__':
    main()