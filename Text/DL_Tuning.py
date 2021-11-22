from keras.layers import Conv1D, Dense, Embedding, Flatten, MaxPooling1D, LSTM, Bidirectional
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import tensorflow as tf
import pandas as pd


def load_vocab(vocab_filename):
    file = open(vocab_filename, 'r')
    text = file.read()
    file.close()
    vocabulary = text.split()
    return set(vocabulary)


def load_dataset():
    pass


def clean_text_based_on_vocab(df, vocabulary):
    clean_data = df['clean text'].tolist()
    text = []
    for sample in clean_data:
        words = sample.split()
        tokens = [w for w in words if w in vocabulary]
        tokens = ' '.join(tokens)
        text.append(tokens)
    return text

# def plot_3d_graph(acc, datatype_, params):
#     fig = plt.figure()
#     x_filters = params["filters"]
#     x_filters = np.array(x_filters)
#     y_units = params["units"]
#     y_units = np.array(y_units)
#     x, y = np.meshgrid(x_filters, y_units)
#     c = np.array(acc)
#     plt.xlabel('Filters')
#     plt.ylabel('Units')
#     plt.title(f'CNN {datatype_} Accuracy')
#     img = plt.scatter(x, y, c=c, cmap='Wistia')
#     fig.colorbar(img, pad=0.1, aspect=30)
#     plt.savefig(f"CNN_{datatype_}_Accuracy.png")
#     plt.close()


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
        self.best_model.save(f'{self.__class__.__name__}.h5')

    def plot_graph(self, train_acc=None, val_acc=None, df=None):
        pass

    def tune(self, X_train, y_train):
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
    def __init__(self, vocab_size, vector_size, input_length):
        super(LSTM_model, self).__init__(vocab_size=vocab_size, vector_size=vector_size, input_length=input_length)
        self.params = {"units": [32, 64, 128, 256]}

    def build_network(self, units, filters=None):
        inner_layers = [Bidirectional(LSTM(units=units, dropout=0.25, recurrent_dropout=0.25))]
        self.model = self.build_model(inner_layers)
        return self.model

    def plot_graph(self, train_acc=None, val_acc=None, df=None):
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
        inner_layers = [Conv1D(filters=filters, kernel_size=3, activation='relu'),
                        MaxPooling1D(pool_size=2),
                        Flatten(),
                        Dense(units, activation='relu')]
        self.model = self.build_model(inner_layers)
        return self.model

    def plot_graph(self, train_acc=None, val_acc=None, df=None):
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

