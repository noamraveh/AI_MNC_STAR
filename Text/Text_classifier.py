import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import shuffle
from sklearn.calibration import CalibratedClassifierCV
from Voting_Classifier import EnsembleClassifier


def preprocessing_data():
    bert = pd.read_csv('bert.csv')
    isear = pd.read_csv('isear.csv')
    tweets = pd.read_csv('tweets.csv', usecols=['Text', 'Emotion'], names=['Tweet', 'Emotion', 'Text']).drop([0])
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
    sns.countplot(data=all_data, x="Emotion")
    plt.show()
    print("##### ORIGINAL DATA SAMPLE#####")
    print(tabulate(all_data.sample(n=5), headers='keys', tablefmt='psql'))

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
    text = df.values.tolist()
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(text)
    encoded_text = tokenizer.texts_to_sequences(text)
    max_length = max([len(s.split()) for s in text])
    data = pad_sequences(encoded_text, maxlen=max_length, padding='pre')
    return max_length, data


def tune_machine_learning_models(X_train_ML, y_train_ML, X_train_Adaboost, y_train_Adaboost):
    print("starts to tune Naive Bayes")
    NB_clf = ComplementNB()
    NB_hyperparams_dict = {'alpha': np.linspace(0.01, 2, num=50)} # todo: after setting the logspace, check linspace for better scaling
    print("CV Naive Bayes")
    NB_model = Tune(NB_clf, NB_hyperparams_dict, X_train_ML, y_train_ML)
    NB_best_model_params = NB_model.tune()
    NB_model.save_model()

    print("starts to tune Linear SVM")
    # Linear SVM (SGDclassifier)  # when the loss is "hinge" the SGD is LinearSVM
    SVM_clf = SGDClassifier(class_weight='balanced')
    SVM_hyperparams_dict = {'alpha': np.logspace(-4, -3, num=50)} # todo: after setting the logspace, check linspace for better scaling
    print("CV SVM")
    SVM_model = Tune(SVM_clf, SVM_hyperparams_dict, X_train_ML, y_train_ML)
    SVM_best_model_params = SVM_model.tune()
    SVM_model.save_model()

    print("starts to tune Linear Regression")
    # linear regression
    LR = LogisticRegression(max_iter=5000, class_weight='balanced')
    LR_hyperparams_dict = {'C': np.linspace(0.1, 5, num=50)} # todo: after setting the logspace, check linspace for better scaling
    print("CV LR")
    LR_model = Tune(LR, LR_hyperparams_dict, X_train_ML, y_train_ML)
    LR_best_model_params = LR_model.tune()
    LR_model.save_model()

    print("starts to tune Adaboost")
    # Adaboost
    AdaBoost = AdaBoostClassifier()
    AdaBoost_hyperparams_dict = {'base_estimator': [DecisionTreeClassifier(max_depth=x) for x in [3, 5, 7, 9]],
                                 'n_estimators': [50, 100, 200, 300, 500],
                                 'learning_rate': [0.1, 0.2, 0.4, 0.6, 0.8, 1]}
    print("CV Adaboost")
    Adaboost_model = Tune(AdaBoost, AdaBoost_hyperparams_dict, X_train_Adaboost, y_train_Adaboost)
    Adaboost_best_model_params = Adaboost_model.tune()
    Adaboost_model.save_model()

    best_models = {'NB': NB_model.best_model, 'SVM': SVM_model.best_model, 'LR': LR_model.best_model, "AdaBoost": Adaboost_model.best_model}
    best_parameters = {'NB': NB_best_model_params, 'SVM': SVM_best_model_params, 'LR': LR_best_model_params, 'AdaBoost': Adaboost_best_model_params}
    accuracies = {'NB': NB_model.val_acc, 'SVM': SVM_model.val_acc, 'LR': LR_model.val_acc, 'AdaBoost': Adaboost_model.val_acc}
    return best_models, best_parameters, accuracies


def tune_deep_learning_models(vocab_size, max_length, X_train, y_train):
    print("starts to tune LSTM")
    LSTM = LSTM_model(vocab_size, 100, max_length)
    LSTM.tune(X_train, y_train)
    LSTM.save_model()

    print("starts to tune CNN")
    CNN = CNN_model(vocab_size, 100, max_length)
    CNN.tune(X_train, y_train)
    CNN.save_model()

    best_models = {'LSTM': LSTM.best_model, 'CNN': CNN.best_model}
    best_parameters = {'LSTM': LSTM.best_params, 'CNN': CNN.best_params}
    accuracy = {'LSTM': LSTM.val_acc, 'CNN': CNN.val_acc}
    return best_models, best_parameters, accuracy


# def build_ML_voting_classifier(parameters, accuracies, X, X_adaboost, y):
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=11, shuffle=True)
#
#     weights = set_weights(accuracies)
#     calibrated_clf = CalibratedClassifierCV(base_estimator=SGDClassifier(**parameters['SVM'], class_weight='balanced'), cv='prefit')
#     calibrated_clf.fit(X, y)
#
#     voting_clf = VotingClassifier(
#         estimators=[('ComplementNB', ComplementNB(**parameters['NB'])),
#                     ('LinearSVM', calibrated_clf),
#                     ('Logistic Regression', LogisticRegression(**parameters['LR'], class_weight='balanced')),
#                     ],
#         voting='soft',
#         flatten_transform=True,
#         weights=weights)
#     combined_clf = combine_ML_clfs(voting_clf,AdaBoostClassifier(**parameters['AdaBoost']), X, X_adaboost, y)
#     voting_clf.fit(X_train, y_train)
#     y_pred = voting_clf.predict(X_val)
#     voting_clf.predict(X_val)
#     val_accuracy = balanced_accuracy_score(y_val, y_pred)
#
#     return voting_clf, val_accuracy


# def build_DL_voting_classifier(parameters, accuracies, X, y):
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=11, shuffle=True)
#     weights = set_weights(accuracies)
#     LSTM_sklearn = tf.keras.wrappers.scikit_learn.KerasClassifier(LSTM_model.build_network(**parameters['LSTM']))
#     LSTM_sklearn._estimator_type = "classifier"
#
#     CNN_sklearn = tf.keras.wrappers.scikit_learn.KerasClassifier(CNN_model.build_network(**parameters['CNN']))
#     CNN_sklearn._estimator_type = "classifier"
#
#     voting_clf = VotingClassifier(
#         estimators=[('lstm', LSTM_sklearn),
#                     ('cnn', CNN_sklearn)],
#         voting='soft',
#         flatten_transform=True,
#         weights=weights)
#
#     voting_clf.fit(X_train, y_train)
#     y_pred = voting_clf.predict(X_val)
#     voting_clf.predict(X_val)
#     val_accuracy = balanced_accuracy_score(y_val, y_pred)
#     return voting_clf, val_accuracy


# def choose_best_model(models, accuracies):
#     model_name = max(accuracies, key=accuracies.get)
#     return models[model_name], model_name, max(accuracies.values())


def main():
    """
    note that the 3 y's (ML, Adaboost and DL) are the same - saving all three
     in case we want to use different random state in the train-test split
    """
    # data = preprocessing_data()
    # X = data['X']
    # y = data['y']
    # added_features_df = data['added_features_df']
    # clean_text = data['clean_text']

    # # todo: if data already exists, comment out the lines above and use the 4 below:
    y = pd.read_csv("y.csv").values.ravel()
    added_features_df = pd.read_csv("added_features.csv")
    clean_text = pd.read_csv("clean_text.csv")
    X = sparse.load_npz("X.npz")


    X_train_ML, X_test_ML, y_train_ML, y_test_ML = train_test_split(X, y, test_size=0.2, random_state=11, shuffle=True)
    sparse.save_npz("X_test_ML.npz", X_test_ML)
    np.savetxt("y_test_ML", y_test_ML, delimiter=',')

    X_train_AdaBoost, X_test_AdaBoost, y_train_AdaBoost, y_test_AdaBoost = train_test_split(added_features_df, y, test_size=0.2, random_state=11, shuffle=True)
    np.savetxt("X_test_AdaBoost", X_test_AdaBoost, delimiter=',')
    np.savetxt("y_test_AdaBoost", y_test_AdaBoost, delimiter=',')

    # training ML models
    best_ML_models, ML_models_params, ML_accuracies = tune_machine_learning_models(X_train_ML, y_train_ML, X_train_AdaBoost, y_train_AdaBoost)

    # training DL models
    vocab_size = 10000
    max_length, X_DL = tokenizer_modifications(clean_text, vocab_size)
    y_DL = LabelBinarizer().fit_transform(y)
    X_train_DL, X_test_DL, y_train_DL, y_test_DL = train_test_split(X_DL, y_DL, test_size=0.2, random_state=11, shuffle=True)
    np.savetxt("X_test_DL", X_test_DL, delimiter=',')
    np.savetxt("y_test_DL", y_test_DL, delimiter=',')

    best_DL_models, DL_models_params, DL_accuracies = tune_deep_learning_models(vocab_size, max_length, X_train_DL, y_train_DL)

    voting_clf = EnsembleClassifier(ML_models_params, ML_accuracies, DL_models_params, DL_accuracies, vocab_size, 100, max_length)

    voting_clf.fit(X_train_ML, X_train_AdaBoost, X_train_DL, y_train_ML)
    y_pred = voting_clf.predict_proba(X_test_ML, X_test_AdaBoost, X_test_DL)
    acc = sum(1 for pred,actual in zip(y_pred, y_test_ML) if np.argmax(pred) == actual) / len(y_test_ML)

    print(f"Voting CLF ACC: {acc} ")


    # best_ML_model, model_name, best_ML_model_accuracy = choose_best_model(best_ML_models, ML_accuracies)
    # best_DL_model, model_name, best_DL_model_accuracy = choose_best_model(best_DL_models, DL_accuracies)
    #
    # # create ML ensemble
    # ML_voting_clf, ML_voting_clf_accuracy = build_ML_voting_classifier(ML_models_params, ML_accuracies, X_train_ML,
    #                                                                    X_train_AdaBoost, y_train_ML)
    #
    # if best_ML_model_accuracy > ML_voting_clf_accuracy:
    #     dump(best_ML_model, f'BestMLClassifier_{model_name}.joblib')
    # else:
    #     dump(ML_voting_clf, f'BestMLClassifier_VotingClf.joblib')
    #
    #
    #
    # # create DL ensemble
    # DL_voting_clf, DL_voting_clf_accuracy = build_DL_voting_classifier(DL_models_params, DL_accuracies, X_train_DL, y_train_DL)
    # if best_DL_model_accuracy > DL_voting_clf_accuracy:
    #     best_DL_model.save(f'BestDLClassifier_{model_name}.h5')
    # else:
    #     dump(DL_voting_clf, f'BestDLClassifier_VotingClf.joblib')


if __name__ == '__main__':
    main()
