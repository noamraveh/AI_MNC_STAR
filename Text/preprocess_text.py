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


class PreProcess:
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
        words = sentence.split()
        return sum(len(word) for word in words) / len(words)

    @staticmethod
    def remove_three_letters_words(sentence):
        words = sentence.split()
        alphabet_lowercase = list(string.ascii_lowercase)
        for index in range(26):
            for word in words:
                if alphabet_lowercase[index] * 3 in word:
                    words.remove(word)
        return ' '.join(word for word in words)

    def NRC_lex_scores_per_line(self, line):
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
        self.all_data['clean text'] = self.all_data['Text'].str.lower()
        self.all_data['clean text'] = self.all_data['clean text'].apply(nfx.remove_userhandles)
        self.all_data['clean text'] = self.all_data['clean text'].apply(nfx.remove_stopwords).apply(
            nfx.remove_punctuations).apply(nfx.remove_special_characters).apply(nfx.remove_numbers)
        # self.all_data['clean text'] = self.all_data['clean text'].apply(self.remove_three_letters_words)
        st = PorterStemmer()
        self.all_data['clean text'] = self.all_data['clean text'].apply(
            lambda x: " ".join([st.stem(word) for word in x.split()]))
        self.all_data['clean text'] = self.all_data['clean text'].apply(
            lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        self.clean_text_df = self.all_data['clean text']
        self.clean_text_df.reset_index(drop=True, inplace=True)

    def add_features(self):
        self.added_features['num ex mark'] = self.all_data['Text'].map(lambda x: x.count("!")).astype('int8')
        self.added_features['num words'] = self.all_data['Text'].apply(lambda x: len(str(x).split(" "))).astype('int32')
        self.added_features['num chars'] = self.all_data['Text'].apply(len).astype('int64')
        self.added_features['avg_word'] = self.all_data['Text'].apply(lambda x: self.avg_word(x)).astype('float16')
        self.added_features['stopwords'] = self.all_data['Text'].apply(
            lambda x: len([x for x in x.split() if x in STOPWORDS])).astype('int8')
        self.added_features['upper'] = self.all_data['Text'].apply(
            lambda x: len([x for x in x.split() if x.isupper()])).astype('int32')

        self.added_features['Emotions_list'] = self.all_data['clean text'].apply(self.NRC_lex_scores_per_line)
        self.added_features = pd.concat(
            [self.added_features.drop(['Emotions_list'], axis=1), self.added_features['Emotions_list'].apply(pd.Series)], axis=1)

    def Tfidf(self, is_train=True):
        vectorizer = TfidfVectorizer()
        self.X = vectorizer.fit_transform(self.all_data['clean text'])
        if is_train:
            self.y = self.all_data["Emotion"]
            self.y.reset_index(drop=True, inplace=True)

    def data_visualisation(self):
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

    def save_csvs(self):
        self.all_data.to_csv('text_dataset.csv', index=False)
        sparse.save_npz("X.npz", self.X)
        self.y.to_csv("y.csv", index=False)
        self.added_features.to_csv('added_features.csv', index=False)
        self.clean_text_df.to_csv("clean_text.csv", index=False)

