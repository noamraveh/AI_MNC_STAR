import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from joblib import dump
import numpy as np
import pandas as pd
from tabulate import tabulate


class Tune:
    """
    Class for performing all the hyperparameters tuning of the ML models.
    """
    def __init__(self, clf, hyperparams_dict, X_train, y_train):
        self.clf = clf
        self.hyperparams_dict = hyperparams_dict
        self.X_train = X_train
        self.y_train = y_train
        self.val_acc = 0
        self.best_model = None

    def tune(self):
        """
        Perform GridSearchCV in order to find the best hyperparmaters for the model.
        Print accuracies and plot graphs of the model accuracies.
        :return: the hyperparameters of the best model.
        """
        check_clf = GridSearchCV(estimator=self.clf, param_grid=self.hyperparams_dict, verbose=True, return_train_score=True)
        check_clf.fit(self.X_train, self.y_train)
        train_acc = check_clf.cv_results_["mean_train_score"]
        val_acc = check_clf.cv_results_["mean_test_score"]
        print(f"Best hyperparams are: {check_clf.best_params_}"
              f"Best Accuracy is: {check_clf.best_score_}")
        if self.clf.__class__.__name__ == 'AdaBoostClassifier':
            table = pd.concat([pd.DataFrame(check_clf.cv_results_["params"]),
                               pd.DataFrame(train_acc, columns=["Train Accuracy"]),
                               pd.DataFrame(val_acc, columns=["Validation Accuracy"])], axis=1)
            table.to_csv("AdaBoost_tuning_results.csv", index=False)
            self.plot_adaboost_graph(table)
        else:
            self.plot_graph(train_acc, val_acc)
        self.val_acc = check_clf.best_score_
        self.best_model = check_clf.best_estimator_
        return check_clf.best_params_

    def get_graph_labels(self):
        """
        :return: the type of the model and its hyperparameter.
        """
        my_class = self.clf.__class__.__name__
        if my_class == 'ComplementNB':
            return 'ComplementNB', 'Alpha'
        elif my_class == 'SGDClassifier':
            return 'Linear SVM', 'Alpha'
        elif my_class == 'LogisticRegression':
            return 'LogisticRegression', 'C'
        elif my_class == 'AdaBoostClassifier':
            return 'AdaBoostClassifier', ''

    def plot_graph(self, train_acc, val_acc):
        """
        Plot the train and validation accuracy of the different generated AdaBoost models.
        :param train_acc: list of the resulted train accuracies from the GridSearchCV.
        :param val_acc: list of the resulted validation accuracies from the GridSearchCV.
        """
        classifier, xlabel = self.get_graph_labels()
        params = []
        for key, value in self.hyperparams_dict.items():
            params.append(value)
        params = params[0]
        plt.figure()
        plt.title(f"{classifier} Classifier Accuracy")
        plt.xlabel(f"{xlabel}")
        plt.ylabel("Accuracy")
        plt.plot(params, val_acc, label='Validation')
        plt.plot(params, train_acc, label='Training')
        plt.legend()
        #plt.show()
        plt.savefig(f"{classifier}_Accuracy.png")
        plt.close()

    def save_model(self):
        filename = self.get_graph_labels()[0]
        dump(self.clf, f'{filename}.joblib')

    def plot_adaboost_graph(self, df):
        """
        Plot the train and validation accuracy of the different generated AdaBoost models.
        :param df: dataframe containing all the different hyperparameters and the resulted accuracies from the GridSearchCV.
        """
        for datatype_ in ["Train", "Validation"]:
            plt.figure()
            plt.title(f'AdaBoost {datatype_} Accuracy')
            plt.xlabel("Learning Rate")
            plt.ylabel(f"{datatype_} Accuracy")
            for value in self.hyperparams_dict["n_estimators"]:
                df1 = df[df['n_estimators'] == value]
                x_lr = df1["learning_rate"].to_numpy()
                y_acc = df1[f"{datatype_} Accuracy"].to_numpy()
                plt.plot(x_lr, y_acc)
            plt.legend(self.hyperparams_dict["n_estimators"], title="Num Estimators")
            plt.savefig(f'AdaBoost_{datatype_}_Accuracy.png')
            plt.close()