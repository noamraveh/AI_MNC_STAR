import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from joblib import dump


def load_dataset():
    pass


class Tune:
    def __init__(self, clf, hyperparams_dict, X_train, y_train):
        self.clf = clf
        self.hyperparams_dict = hyperparams_dict
        self.X_train = X_train
        self.y_train = y_train

    def tune(self):
        check_clf = GridSearchCV(estimator=self.clf, param_grid=self.hyperparams_dict, verbose=True)
        check_clf.fit(self.X_train, self.y_train)
        train_acc = check_clf.cv_results_["mean_train_score"]
        val_acc =   check_clf.cv_results_["mean_test_score"]
        params = self.hyperparams_dict.values()[0]
        self.plot_graph(self.clf, train_acc, val_acc, params)
        print(f"Best hyperparams are: {check_clf.best_params_}"
              f"Best Accuracy is: {check_clf.best_score_}")
        return check_clf.best_estimator_

    def get_graph_labels(self):
        if self.clf is MultinomialNB:
            return 'MultinomialNB', 'Alpha'
        elif self.clf is SGDClassifier:
            return 'Linear SVM', 'Alpha'
        elif self.clf is LogisticRegression:
            return 'LogisticRegression', 'C'

    def plot_graph(self, clf, train_acc, val_acc, params):
        classifier, xlabel = self.get_graph_labels()
        plt.figure()
        plt.title(f"{classifier} Classifier Accuracy")
        plt.xlabel(f"{xlabel}")
        plt.ylabel("Accuracy")
        plt.plot(params, val_acc, label='Validation')
        plt.plot(params, train_acc, label='Training')
        plt.legend()
        plt.show()

    def save_model(self):
        filename = self.get_graph_labels()[0]
        dump(self.clf, f'{filename}.joblib')

def main():
    X_train, y_train = load_dataset()
    # Naive Bayes
    """
    MultinomialNB implements the naive Bayes algorithm for multinomially distributed data,
    and is one of the two classic naive Bayes variants used in text classification
    (where the data are typically represented as word vector counts,
     although tf-idf vectors are also known to work well in practice).
    """
    print("starts to tune Naive Bayes")
    NB_clf = MultinomialNB()
    NB_hyperparams_dict = {'alpha': np.logspace(-2, 3, num=400)} # todo: after setting the logspace, check linspace for better scaling
    NB_model = Tune(NB_clf, NB_hyperparams_dict, X_train, y_train)

    print("starts to tune Linear SVM")
    # Linear SVM (SGDclassifier)  # when the loss is "hinge" the SGD is LinearSVM
    SVM_clf = SGDClassifier()
    SVM_hyperparams_dict = {'alpha': np.logspace(-3, 3, num=500)} # todo: after setting the logspace, check linspace for better scaling
    SVM_model = Tune(SVM_clf, NB_hyperparams_dict, X_train, y_train)

    print("starts to tune Linear Regression")
    # linear regression
    LR = LogisticRegression(max_iter=1000)
    LR_hyperparams_dict = {'C': np.logspace(-3, 3, num=500)} # todo: after setting the logspace, check linspace for better scaling
    LR_model = Tune(LR, LR_hyperparams_dict, X_train, y_train)


if __name__ == '__main__':
    main()
