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
        check_clf = GridSearchCV(estimator=self.clf, param_grid=self.hyperparams_dict, scoring='balanced_accuracy', verbose=True, return_train_score=True)
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
            self.plot_4d_graph(table)
        else:
            self.plot_graph(train_acc, val_acc)
        self.val_acc = check_clf.best_score_
        self.best_model = check_clf.best_estimator_
        return check_clf.best_params_

    def get_graph_labels(self):
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
        plt.savefig(f"{classifier}_Accuracy.png")
        plt.show()

    def save_model(self):
        filename = self.get_graph_labels()[0]
        dump(self.clf, f'{filename}.joblib')

    @staticmethod
    def plot_4d_graph(df):
        pre_extract = df["base_estimator"].tolist()
        extracted = [x.max_depth for x in pre_extract]
        # for sample in pre_extract:
        #     digit = [c for c in sample if c.isdigit()]
        #     if len(digit):
        #         extracted.append(int(digit[0]))
        df["base_estimator"] = np.array(extracted)
        for datatype_ in ["Train", "Validation"]:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x_depth = df['base_estimator'].to_numpy()
            y_num_estimators = df["n_estimators"].to_numpy()
            z_lr = df["learning_rate"].to_numpy()
            c = df[f'{datatype_} Accuracy'].to_numpy()
            ax.set_xlabel('Max Depth')
            ax.set_ylabel('Num Estimators')
            ax.set_zlabel('Learning Rate')

            ax.set_title(f'AdaBoost {datatype_} Accuracy')
            img = ax.scatter(x_depth, y_num_estimators, z_lr, c=c, cmap='Wistia')
            fig.colorbar(img, pad=0.1, aspect=30)
            plt.savefig(f'AdaBoost_{datatype_}_Accuracy.png')
