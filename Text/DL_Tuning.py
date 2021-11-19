from keras.layers import Conv1D, Dense, Embedding, Flatten, MaxPooling1D, LSTM, Bidirectional
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import tensorflow as tf


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


def plot_3d_graph(acc, datatype, params):
    x_filters = params["filters"]
    x_filters = np.array(x_filters)
    y_units = params["units"]
    y_units = np.array(y_units)
    x, y = np.meshgrid(x_filters, y_units)
    z = acc
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pnt3d = ax.scatter(x, y, z, c=z)
    plt.colorbar(pnt3d)
    ax.set_xlabel('Filters')
    ax.set_ylabel('Units')
    ax.set_zlabel(f'{datatype} Accuracy')
    ax.set_title(f'CNN {datatype} Accuracy')
    # plt.show()
    plt.savefig(f"CNN_{datatype}_Accuracy.png")


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

    def plot_graph(self, train_acc, val_acc):
        pass

    def tune(self, X_train, y_train):
        batch_size = 256
        epochs = 10
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3)
        model = KerasClassifier(build_fn=self.build_network)
        check_model = GridSearchCV(estimator=model, param_grid=self.params, cv=5, verbose=1, scoring='accuracy', return_train_score=True)
        check_model = check_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[callback], validation_split=0.2)
        train_acc = check_model.cv_results_["mean_train_score"]
        val_acc = check_model.cv_results_["mean_test_score"]
        self.val_acc = check_model.best_score_
        self.best_params = check_model.best_params_
        print(f'{self.__class__.__name__.replace("_model", "")} Validation Accuracy: { self.val_acc}')
        print(f'{self.__class__.__name__.replace("_model", "")} Best Parameters: {self.best_params}')
        self.best_model = check_model.best_estimator_.model
        self.plot_graph(train_acc, val_acc)


class LSTM_model(TuneNetwork):
    def __init__(self, vocab_size, vector_size, input_length):
        super(LSTM_model, self).__init__(vocab_size=vocab_size, vector_size=vector_size, input_length=input_length)
        self.params = {"units": [32, 64, 128, 256]}

    def build_network(self, units, filters=None):
        inner_layers = [Bidirectional(LSTM(units=units, dropout=0.25, recurrent_dropout=0.25))]
        self.model = self.build_model(inner_layers)
        return self.model

    def plot_graph(self, train_acc, val_acc):
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

    def plot_graph(self, train_acc, val_acc):
        plot_3d_graph(train_acc, "Train", self.params)
        plot_3d_graph(val_acc, "Validation", self.params)

