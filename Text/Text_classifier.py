import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from ML_Tuning import Tune
from DL_Tuning import LSTM_model, CNN_model
import matplotlib.pyplot as plt
from tabulate import tabulate
from preprocess_text import PreProcess
from sklearn.preprocessing import LabelBinarizer
from scipy import sparse
from joblib import dump
from sklearn.utils import shuffle
from sklearn.calibration import CalibratedClassifierCV
from Voting_Classifier import EnsembleClassifier
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


def save_tokenizer_to_file(tokenizer):
    """
    Save the given tokenizer to a pickle file.
    :param tokenizer: trained tokenizer.
    """
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def preprocessing_data():
    """
    Merge the three datasets together into one large dataset.
    Convert emotions and encode them.
    Use the PreProcess class for the preprocessing section.
    :return: dictionary where the keys are the type of the data and the values are the dataframes/csr_matrix.
    """
    bert = pd.read_csv('../Data/original_data/bert.csv')
    isear = pd.read_csv('../Data/original_data/isear.csv')
    tweets = pd.read_csv('../Data/original_data/tweets.csv', usecols=['Text', 'Emotion'], names=['Tweet', 'Emotion', 'Text']).drop([0])
    all_data = pd.concat([tweets, bert, isear])
    all_data = shuffle(all_data, random_state=42)

    # convert emotions
    to_remove = ['empty', 'relief', 'boredom', 'shame', 'guilt']
    to_happy = ['enthusiasm', 'love', 'fun', 'happy', 'joy']
    all_data = all_data[~all_data['Emotion'].isin(to_remove)]
    all_data = all_data[~all_data['Emotion'].isna()]
    all_data.loc[all_data['Emotion'].isin(to_happy), 'Emotion'] = 'happiness'
    all_data.loc[all_data['Emotion'] == 'worry', 'Emotion'] = 'fear'
    all_data.loc[all_data['Emotion'] == 'hate', 'Emotion'] = 'anger'

    # encode emotions
    emotions_dict = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "neutral": 4,
                     "sadness": 5, "surprise": 6}

    # analysis
    # sns.countplot(data=all_data, x="Emotion")
    # plt.show()
    # print("##### ORIGINAL DATA SAMPLE#####")
    # print(tabulate(all_data.sample(n=5), headers='keys', tablefmt='psql'))

    all_data = all_data.replace(to_replace=emotions_dict)

    pre_process = PreProcess(all_data)
    pre_process.pre_process_text()
    pre_process.Tfidf()
    pre_process.data_visualisation()
    pre_process.add_features()
    pre_process.save_csvs()

    data = {'X': pre_process.X,
            'y': pre_process.y,
            'added_features_df': pre_process.added_features,
            'clean_text': pre_process.clean_text_df,
            }
    return data


def tokenizer_modifications(df, num_words):
    """
    Perform tokenization on the clean sentences.
    :param df: dataframe of clean text
    :param num_words: the maximum number of words for the tokenizer to keep
    :return: max length of a sentence in the dataframe and the tokenizied sentences in a dataframe form.
    """
    text = df.values.tolist()
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(text)
    save_tokenizer_to_file(tokenizer)
    encoded_text = tokenizer.texts_to_sequences(text)
    max_length = max([len(s.split()) for s in text])
    data = pad_sequences(encoded_text, maxlen=max_length, padding='pre')
    return max_length, data


def tune_machine_learning_models(X_train_ML, y_train_ML, X_train_Adaboost, y_train_Adaboost):
    """
    Tune each of the following models: Complement Naive Bayes, LinearSVM, Logistic Regression and AdaBoost.
    :param X_train_ML: X values of ML train dataset
    :param y_train_ML: y values of ML train dataset
    :param X_train_Adaboost: X values of AdaBoost train dataset
    :param y_train_Adaboost: y values of AdaBoost train dataset
    :return: 3 dictionaries: 1) keys: models' names | values: best models.
                             2) keys: models' names | values: best hyperparameters for the model.
                             3) keys: models' names | values: best average accuracy of the model.
    """
    print("starts to tune Naive Bayes")
    NB_clf = ComplementNB()
    NB_hyperparams_dict = {
        'alpha': np.linspace(0.01, 5, num=50)}  # todo: after setting the logspace, check linspace for better scaling
    print("CV Naive Bayes")
    NB_model = Tune(NB_clf, NB_hyperparams_dict, X_train_ML, y_train_ML)
    NB_best_model_params = NB_model.tune()

    print("starts to tune Linear SVM")
    # Linear SVM (SGDclassifier)  # when the loss is "hinge" the SGD is LinearSVM
    SVM_clf = SGDClassifier(class_weight='balanced')
    SVM_hyperparams_dict = {
        'alpha': np.logspace(-6, -4, num=50)}  # todo: after setting the logspace, check linspace for better scaling
    print("CV SVM")
    SVM_model = Tune(SVM_clf, SVM_hyperparams_dict, X_train_ML, y_train_ML)
    SVM_best_model_params = SVM_model.tune()

    print("starts to tune Logistic Regression")
    # linear regression
    LR = LogisticRegression(max_iter=5000, class_weight='balanced')
    LR_hyperparams_dict = {
        'C': np.linspace(0.01, 10, num=50)}  # todo: after setting the logspace, check linspace for better scaling
    print("CV LR")
    LR_model = Tune(LR, LR_hyperparams_dict, X_train_ML, y_train_ML)
    LR_best_model_params = LR_model.tune()

    print("starts to tune Adaboost")
    # Adaboost
    AdaBoost = AdaBoostClassifier()
    AdaBoost_hyperparams_dict = {'n_estimators': [50, 100, 200, 300, 500],
                                 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1]}
    print("CV Adaboost")
    Adaboost_model = Tune(AdaBoost, AdaBoost_hyperparams_dict, X_train_Adaboost, y_train_Adaboost)
    Adaboost_best_model_params = Adaboost_model.tune()

    best_models = {'NB': NB_model.best_model, 'SVM': SVM_model.best_model, 'LR': LR_model.best_model,
                   "AdaBoost": Adaboost_model.best_model}
    best_parameters = {'NB': NB_best_model_params, 'SVM': SVM_best_model_params, 'LR': LR_best_model_params,
                       'AdaBoost': Adaboost_best_model_params}
    accuracies = {'NB': NB_model.val_acc, 'SVM': SVM_model.val_acc, 'LR': LR_model.val_acc,
                  'AdaBoost': Adaboost_model.val_acc}
    return best_models, best_parameters, accuracies


def tune_deep_learning_models(vocab_size, max_length, X_train, y_train):
    """
    Tune each of the following models: LSTM and CNN neutral networks.
    :param vocab_size: size of vocabulary (10,000) for the embedded layer.
    :param max_length: length of the longest cleaned sentence in the dataset.
    :param X_train: X values of DL train dataset
    :param y_train: y values of DL train dataset
    :return:  3 dictionaries: 1) keys: models' names | values: best models.
                             2) keys: models' names | values: best hyperparameters for the model.
                             3) keys: models' names | values: best average accuracy of the model.
    """
    print("starts to tune LSTM")
    LSTM = LSTM_model(vocab_size, 100, max_length)
    LSTM.tune(X_train, y_train)

    print("starts to tune CNN")
    CNN = CNN_model(vocab_size, 100, max_length)
    CNN.tune(X_train, y_train)

    best_models = {'LSTM': LSTM.best_model, 'CNN': CNN.best_model}
    best_parameters = {'LSTM': LSTM.best_params, 'CNN': CNN.best_params}
    accuracy = {'LSTM': LSTM.val_acc, 'CNN': CNN.val_acc}
    return best_models, best_parameters, accuracy


def main():
    data = preprocessing_data()
    X = data['X']
    y = data['y']
    added_features_df = data['added_features_df']
    clean_text = data['clean_text']

    """
    if data already exists, comment out the lines above and use the 4 below:
    """
    # y = pd.read_csv("../Data/processed_data/y.csv").values.ravel()
    # added_features_df = pd.read_csv("../Data/processed_data/added_features.csv")
    # clean_text = pd.read_csv("../Data/processed_data/clean_text.csv").astype(str).iloc[:,0]
    # X = sparse.load_npz("../Data/processed_data/X.npz")

    X_train_ML, X_test_ML, y_train_ML, y_test_ML = train_test_split(X, y, test_size=0.2, random_state=11, shuffle=True)
    sparse.save_npz("X_test_ML.npz", X_test_ML)
    np.savetxt("y_test_ML", y_test_ML, delimiter=',')

    X_train_AdaBoost, X_test_AdaBoost, y_train_AdaBoost, y_test_AdaBoost = train_test_split(added_features_df, y,
                                                                                            test_size=0.2,
                                                                                            random_state=11,
                                                                                            shuffle=True)
    X_train_AdaBoost.reset_index(drop=True, inplace=True)
    X_test_AdaBoost.reset_index(drop=True, inplace=True)
    np.savetxt("X_test_AdaBoost", X_test_AdaBoost, delimiter=',')
    np.savetxt("y_test_AdaBoost", y_test_AdaBoost, delimiter=',')

    # training ML models
    best_ML_models, ML_models_params, ML_accuracies = tune_machine_learning_models(X_train_ML, y_train_ML,
                                                                                   X_train_AdaBoost, y_train_AdaBoost)

    # training DL models
    vocab_size = 10000
    max_length, X_DL = tokenizer_modifications(clean_text, vocab_size)
    print(f"Max Length for DL models: {max_length}")
    y_DL = LabelBinarizer().fit_transform(y)
    X_train_DL, X_test_DL, y_train_DL, y_test_DL = train_test_split(X_DL, y_DL, test_size=0.2, random_state=11,
                                                                    shuffle=True)
    np.savetxt("X_test_DL", X_test_DL, delimiter=',')
    np.savetxt("y_test_DL", y_test_DL, delimiter=',')

    best_DL_models, DL_models_params, DL_accuracies = tune_deep_learning_models(vocab_size, max_length, X_train_DL,
                                                                                y_train_DL)

    # the section below is used just for printing the average validation accuracy of the voting_clf - comment out this section afterwards

    # voting_clf = EnsembleClassifier(ML_models_params, ML_accuracies, DL_models_params, DL_accuracies, vocab_size, 100, max_length)
    # kfold = StratifiedKFold(n_splits=5, random_state=11, shuffle=True)
    # sum_acc = 0
    # for train_group_idx, val_group_idx in kfold.split(X=X_train_ML, y=y_train_ML):
    #     X_train_ML_voting = X_train_ML[train_group_idx]
    #     X_train_AdaBoost_voting = X_train_AdaBoost.iloc[train_group_idx]
    #     X_train_DL_voting = X_train_DL[train_group_idx]
    #     X_val_ML_voting = X_train_ML[val_group_idx]
    #     X_val_AdaBoost_voting = X_train_AdaBoost.iloc[val_group_idx]
    #     X_val_DL_voting = X_train_DL[val_group_idx]
    #     y_train_voting = y_train_ML[train_group_idx]
    #     y_val_voting = y_train_ML[val_group_idx]
    #     voting_clf.fit(X_train_ML_voting, X_train_AdaBoost_voting, X_train_DL_voting, y_train_voting)
    #     cur_accuracy = voting_clf.calc_acc_on_val(X_val_ML_voting, X_val_AdaBoost_voting, X_val_DL_voting, y_val_voting)
    #     sum_acc += cur_accuracy
    # avg_acc = sum_acc / float(5)
    # print(f"Voting classifier average accuracy on validation set: {avg_acc}")


    # FINAL VOTING CLASSIFIER
    voting_clf = EnsembleClassifier(ML_models_params, ML_accuracies, DL_models_params, DL_accuracies, vocab_size, 100,
                                    max_length)
    voting_clf.fit(X_train_ML, X_train_AdaBoost, X_train_DL, y_train_ML)
    y_pred = voting_clf.predict_proba_on_test(X_test_ML, X_test_AdaBoost, X_test_DL, y_test_ML)
    y_pred = np.argmax(y_pred, axis=1)
    cf_matrix = confusion_matrix(y_test_ML, y_pred)
    cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

    # create confusion matrix
    labels = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
    voting_clf.create_multi_label_confusion_matrix(cf_matrix, labels)
    voting_clf.plot_accuracies()

    print(f"Voting classifier accuracy on test set: {voting_clf.voting_clf_accuracy_on_test}")


if __name__ == '__main__':
    main()
