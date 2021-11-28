from keras.layers import Conv1D, Dense, Embedding, Flatten, MaxPooling1D, LSTM, Bidirectional
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import tensorflow as tf
import pandas as pd


class TuneNetwork:
    def __init__(self, vocab_size, vector_size, input_length):
        self.vocab_size = vocab_size
        self.vector_size = vector_size
        self.input_length = input_length
        self.model = None
        self.val_acc = 0
        self.params = {}
        self.best_model = None
        self.best_params = None

    def build_model(self, inner_layers):
        """
        Build and compile the model given the inner layers as a list.
        :param inner_layers: the inner layers of the model.
        :return: compiled Sequential model.
        """
        all_layers = []
        embedded_layer = Embedding(self.vocab_size, self.vector_size, input_length=self.input_length)
        all_layers.append(embedded_layer)
        all_layers.extend(inner_layers)
        all_layers.append(Dense(7, activation='softmax'))
        model = Sequential(all_layers)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_network(self, units, filters=None):
        pass

    def save_model(self):
        """
        Save the model to an .h5 file.
        """
        self.best_model.save(f'{self.__class__.__name__}.h5')

    def plot_graph(self, train_acc=None, val_acc=None, df=None):
        pass

    def tune(self, X_train, y_train):
        """
        Perform GridSearchCV in order to find the best hyperparmaters for the network.
        Print accuracies and plot graphs of the model accuracies.
        :param X_train: the tokenized clean text.
        :param y_train: the labels.
        """
        batch_size = 256
        epochs = 10
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3)
        model = KerasClassifier(build_fn=self.build_network)
        check_model = GridSearchCV(estimator=model, param_grid=self.params, cv=5, verbose=1, return_train_score=True)
        check_model = check_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[callback], validation_split=0.2)
        train_acc = check_model.cv_results_["mean_train_score"]
        val_acc = check_model.cv_results_["mean_test_score"]
        self.val_acc = check_model.best_score_
        self.best_params = check_model.best_params_
        print(f'{self.__class__.__name__.replace("_model", "")} Validation Accuracy: { self.val_acc}')
        print(f'{self.__class__.__name__.replace("_model", "")} Best Parameters: {self.best_params}')
        self.best_model = check_model.best_estimator_.model
        if self.__class__.__name__ == 'CNN_model':
            table = pd.concat([pd.DataFrame(check_model.cv_results_["params"]),
                               pd.DataFrame(train_acc, columns=["Train Accuracy"]),
                               pd.DataFrame(val_acc, columns=["Validation Accuracy"])], axis=1)
            table.to_csv("CNN_tuning_results.csv", index=False)
            self.plot_graph(df=table)
        else:
            self.plot_graph(train_acc=train_acc, val_acc=val_acc)


class LSTM_model(TuneNetwork):
    """
    Derives from TuneNetwork class.
    """
    def __init__(self, vocab_size, vector_size, input_length):
        super(LSTM_model, self).__init__(vocab_size=vocab_size, vector_size=vector_size, input_length=input_length)
        self.params = {"units": [32, 64, 128, 256]}

    def build_network(self, units, filters=None):
        """
        Set the inner layers of the model according to the units.
        :param units: number of units for the LSTM layer.
        :param filters: Not in use in LSTM.
        :return: the compiled model with the specified inner layers.
        """
        inner_layers = [Bidirectional(LSTM(units=units, dropout=0.25, recurrent_dropout=0.25))]
        self.model = self.build_model(inner_layers)
        return self.model

    def plot_graph(self, train_acc=None, val_acc=None, df=None):
        """
        Plot the train and validation accuracy of the different generated LSTM models.
        :param train_acc: list of the resulted train accuracies from the GridSearchCV.
        :param val_acc: list of the resulted train accuracies from the GridSearchCV.
        :param df: not in use.
        """
        units = self.params["units"]
        plt.figure()
        plt.title(f'{self.__class__.__name__.replace("_model", "")} Classifier Accuracy')
        plt.xlabel("units")
        plt.ylabel("Accuracy")
        plt.plot(units, train_acc, label='Training')
        plt.plot(units, val_acc, label='Validation')
        plt.legend()
        #plt.show()
        plt.savefig("LSTM_Accuracy.png")
        plt.close()


class CNN_model(TuneNetwork):
    def __init__(self, vocab_size, vector_size, input_length):
        super(CNN_model, self).__init__(vocab_size=vocab_size, vector_size=vector_size, input_length=input_length)
        self.params = {'filters': [32, 64, 128, 256], "units": [32, 64, 128, 256]}

    def build_network(self, units, filters=None):
        """
        Set the inner layers of the model according to the units and filters.
        :param units: number of units for the dense layer.
        :param filters: number of filters for the convolution layer.
        :return: the compiled model with the specified inner layers.
        """
        inner_layers = [Conv1D(filters=filters, kernel_size=3, activation='relu'),
                        MaxPooling1D(pool_size=2),
                        Flatten(),
                        Dense(units, activation='relu')]
        self.model = self.build_model(inner_layers)
        return self.model

    def plot_graph(self, train_acc=None, val_acc=None, df=None):
        """
        Plot the train and validation accuracy of the different generated CNN models.
        :param train_acc: not in use.
        :param val_acc: not in use.
        :param df: dataframe containing all the different hyperparameters and the resulted accuracies from the GridSearchCV.
        """
        for datatype_ in ["Train", "Validation"]:
            plt.figure()
            plt.title(f'CNN {datatype_} Accuracy')
            plt.xlabel("Filters")
            plt.ylabel(f"{datatype_} Accuracy")
            for value in self.params["units"]:
                df1 = df[df['units'] == value]
                x_lr = df1["filters"].to_numpy()
                y_acc = df1[f"{datatype_} Accuracy"].to_numpy()
                plt.plot(x_lr, y_acc)
            plt.legend(self.params["units"], title="Units")
            plt.savefig(f'CNN_{datatype_}_Accuracy.png')
            plt.close()

