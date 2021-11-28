import pandas as pd
import matplotlib.pyplot as plt
import neattext.functions as nfx
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from textblob import Word
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle


class PreProcess:
    """
    Class for performing the preprocessing of the dataset.
    """
    def __init__(self, data):
        self.all_data = data
        self.NRC_lex = pd.read_csv('NRC_Emotion_Lexicon.csv')[
            ['English (en)', 'Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']]
        self.NRC_lex = self.NRC_lex.rename(columns={'Joy': 'Happy', 'Sadness': 'Sad', 'Anger': 'Angry'})
        self.X = None
        self.y = None
        self.clean_text_df = None
        self.added_features = pd.DataFrame()

    @staticmethod
    def avg_word(sentence):
        """
        :param sentence: string
        :return: the average length of the words in the sentence
        """
        words = sentence.split()
        return sum(len(word) for word in words) / len(words)

    @staticmethod
    def is_repeated_letters(sentence):
        """
        :param sentence: string
        :return: 0 if none of the words in the sentence have more than 2 repeated letters. 1- otherwise.
        """
        is_repeated = 0
        words = sentence.split()
        alphabet_lowercase = list(string.ascii_lowercase)
        for index in range(26):
            for word in words:
                if alphabet_lowercase[index] * 3 in word:
                    is_repeated = 1
                    break
        return is_repeated

    @staticmethod
    def save_vectorizer_to_file(vectorizer):
        """
        Given a vectorizer, save it to a pickle file for future use.
        :param vectorizer: the vectorizer
        """
        with open('vectorizer.pickle', 'wb') as handle:
            pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_vectorizer(filename):
        """
        Given a path to a pickle file saving vectorizer, load the vectorizer from the file.
        :param filename: path to the saved vectorizer file in pickle format.
        """
        with open(filename, 'rb') as handle:
            vectorizer = pickle.load(handle)
        return vectorizer

    def NRC_lex_scores_per_line(self, line):
        """
        calculates the nrclex score using the NRC_lex library.
        :param line: sentence.
        :return: dictionary of the score of each emotion in the sentence.
        """
        emotions_scores_dict = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Sad": 0, "Surprise": 0}
        line = line.split()
        for word in line:
            scores = self.NRC_lex.loc[self.NRC_lex['English (en)'] == word]
            if scores.size == 0:
                continue
            for emotion in emotions_scores_dict.keys():
                emotions_scores_dict[emotion] += scores.iloc[0][emotion]
        emotions_scores_dict_new = {"Score" + str(key): val for key, val in emotions_scores_dict.items()}
        return emotions_scores_dict_new

    def pre_process_text(self):
        """
        Performs cleaning of the text in order for it to be ready to be vectorized and tokenized.
        """
        self.all_data['clean text'] = self.all_data['Text'].str.lower()
        self.all_data['clean text'] = self.all_data['clean text'].apply(nfx.remove_userhandles)
        self.all_data['clean text'] = self.all_data['clean text'].apply(nfx.remove_stopwords).apply(
            nfx.remove_punctuations).apply(nfx.remove_special_characters).apply(nfx.remove_numbers)
        st = PorterStemmer()
        self.all_data['clean text'] = self.all_data['clean text'].apply(
            lambda x: " ".join([st.stem(word) for word in x.split()]))
        self.all_data['clean text'] = self.all_data['clean text'].apply(
            lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        self.clean_text_df = self.all_data['clean text']
        self.clean_text_df.reset_index(drop=True, inplace=True)

    def add_features(self):
        """
        Add more features to the dataframe - Used for the AdaBoost features.
        """
        self.added_features['num ex mark'] = self.all_data['Text'].map(lambda x: x.count("!")).astype('int8')
        self.added_features['num words'] = self.all_data['Text'].apply(lambda x: len(str(x).split(" "))).astype('int32')
        self.added_features['num chars'] = self.all_data['Text'].apply(len).astype('int64')
        self.added_features['avg_word'] = self.all_data['Text'].apply(lambda x: self.avg_word(x)).astype('float16')
        self.added_features['stopwords'] = self.all_data['Text'].apply(
            lambda x: len([x for x in x.split() if x in STOPWORDS])).astype('int8')
        self.added_features['upper'] = self.all_data['Text'].apply(
            lambda x: len([x for x in x.split() if x.isupper()])).astype('int32')
        self.added_features['repeated letters'] = self.all_data['Text'].apply(
            lambda x: self.is_repeated_letters(x)).astype('int8')

        self.added_features['Emotions_list'] = self.all_data['clean text'].apply(self.NRC_lex_scores_per_line)
        self.added_features = pd.concat(
             [self.added_features.drop(['Emotions_list'], axis=1), self.added_features['Emotions_list'].apply(pd.Series)], axis=1)

    def Tfidf(self, is_train=True):
        """
        If: is_train=True --> Creates TFIDF vectorizier and fit it on the cleaned train sentences.
        Else: load a saved TDIDF vectorizer and transforms the cleaned test sentences.
        :param is_train: Boolean value whether the function is used on train set or test set.
        """
        vectorizer = TfidfVectorizer()
        if is_train:
            self.X = vectorizer.fit_transform(self.all_data['clean text'])
            self.save_vectorizer_to_file(vectorizer)
            self.y = self.all_data["Emotion"]
            self.y.reset_index(drop=True, inplace=True)
        else:
            vectorizer = self.load_vectorizer('vectorizer.pickle')
            self.X = vectorizer.transform(self.all_data['clean text'])

    def data_visualisation(self):
        """
        Create wordcloud of the train set words and plot it.
        """
        words = ''
        for i in self.all_data['clean text'].values:
            words += '{} '.format(i.lower())
        wd = pd.DataFrame(Counter(words.split()).most_common(200), columns=['word', 'frequency'])
        data = dict(zip(wd['word'].tolist(), wd['frequency'].tolist()))
        wc = WordCloud(background_color='white',
                       max_words=200).generate_from_frequencies(data)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.savefig("WordCloud.png")
        # plt.show()
        plt.close()

    def save_csvs(self):
        """
        Save the cleaned and generated data to files for future use.
        """
        self.all_data.to_csv('../Data/processed_data/text_dataset.csv', index=False)
        sparse.save_npz("../Data/processed_data/X.npz", self.X)
        self.y.to_csv("../Data/processed_data/y.csv", index=False)
        self.added_features.to_csv('../Data/processed_data/added_features.csv', index=False)
        self.clean_text_df.to_csv("../Data/processed_data/clean_text.csv", index=False)

