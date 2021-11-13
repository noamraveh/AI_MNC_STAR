import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from joblib import dump
import numpy as np
import pandas as pd
from tabulate import tabulate


class Tune:
    def __init__(self, clf, hyperparams_dict, X_train, y_train):
        self.clf = clf
        self.hyperparams_dict = hyperparams_dict
        self.X_train = X_train
        self.y_train = y_train
        self.val_acc = 0
        self.best_model = None

    def tune(self):
        check_clf = GridSearchCV(estimator=self.clf, param_grid=self.hyperparams_dict, verbose=True, return_train_score=True)
        check_clf.fit(self.X_train, self.y_train)
        train_acc = check_clf.cv_results_["mean_train_score"]
        val_acc = check_clf.cv_results_["mean_test_score"]
        self.plot_graph(train_acc, val_acc)
        if self.clf.__class__.__name__ == 'AdaBoost':
            table = pd.concat([pd.DataFrame(check_clf.cv_results_["params"]),
                               pd.DataFrame(train_acc, columns=["Train Accuracy"]),
                               pd.DataFrame(val_acc, columns=["Validation Accuracy"])], axis=1)
            print(tabulate(table, headers='keys', tablefmt='psql'))

        self.val_acc = check_clf.best_score_
        print(f"Best hyperparams are: {check_clf.best_params_}"
              f"Best Accuracy is: {check_clf.best_score_}")
        self.best_model = check_clf.best_estimator_.model
        return check_clf.best_params_

    def get_graph_labels(self):
        my_class = self.clf.__class__.__name__
        if my_class == 'ComplementNB':
            return 'ComplementNB', 'Alpha'
        elif my_class == 'SGDClassifier':
            return 'Linear SVM', 'Alpha'
        elif my_class == 'LogisticRegression':
            return 'LogisticRegression', 'C'
        elif my_class == 'AdaBoost':
            return 'AdaBoost', ""

    def plot_graph(self, train_acc, val_acc):
        classifier, xlabel = self.get_graph_labels()
        if classifier == "AdaBoost":
            self.plot_4d_graph(train_acc, "Train")
            self.plot_4d_graph(val_acc, "Validation")
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
        plt.show()

    def plot_4d_graph(self, acc, datatype):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_depth = np.array([3, 5, 7, 9])
        y_num_estimators = np.array(self.hyperparams_dict["num estimators"])
        z_lr = np.array(self.hyperparams_dict["learning rate"])
        c = acc
        ax.set_xlabel('Num Estimators')
        ax.set_ylabel('Learning Rate')
        ax.set_zlabel(f'{datatype} Accuracy')

        ax.set_title(f'AdaBoost {datatype} Accuracy')
        img = ax.scatter(x_depth, y_num_estimators, z_lr, c=c, cmap=plt.hot())
        fig.colorbar(img)
        # plt.show()
        plt.savefig(f"AdaBoost_{datatype}_acc.png")

    def save_model(self):
        filename = self.get_graph_labels()[0]
        dump(self.clf, f'{filename}.joblib')
