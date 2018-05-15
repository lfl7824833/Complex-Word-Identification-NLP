from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import svm

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint

from wordfreq import word_frequency
from nltk.corpus import wordnet
from itertools import chain
import nltk, string, re
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import pyphen

# import logging
# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
# import gensim
# from gensim.models import word2vec
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

class Baseline(object):

    def __init__(self, language, type):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            self.lang = 'en'

        else:  # spanish
            self.avg_word_length = 6.2
            self.lang = 'es'

        # self.model = LogisticRegression()
        if type == 'classify':
            self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, ), random_state=1)
            # GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
            # MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, ), random_state=1)
        else:
            self.model = MLPRegressor(hidden_layer_sizes=(3), activation='tanh', solver='lbfgs')
            # MLPRegressor(hidden_layer_sizes=(3), activation='tanh', solver='lbfgs')
            # GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
            # svm.SVR()

        self.type = type
        # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        # sentences = word2vec.Text8Corpus(u"datasets/text8")
        # self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(r'C:/Users/lfl78/Documents/MScDataAnalytics/COM6513 - Natural Language Processing/classProject/cwisharedtask2018-teaching/GoogleNews-vectors-negative300.bin', binary=True)
        # print("word2vec success")

    def extract_features(self, word, sentence):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))

        wordRE = re.compile(r'[a-zA-Z]+')
        if self.lang == 'en':
            tokens = wordRE.findall(sentence.replace("n't", " n't").
                                replace("n’t", " n't").lower())
        else:
            tokens = wordRE.findall(sentence.replace("fina", " fina ").
                                    replace("derrotar"," derrotar ").lower())

        ls_freq_sent = [word_frequency(w, self.lang) * 1e4 for w in tokens]

        freq_word = 0
        for w1 in word.split(' '):
            freq_word += word_frequency(w1, self.lang) * 1e4

        sort = sorted(ls_freq_sent)
        if len(ls_freq_sent)>2:
            if freq_word == sort[0] or freq_word == sort[1] or freq_word == sort[2]:
                signal = 1
            else:
                signal = 0
        else:
            if freq_word == sort[0]:
                signal = 1
            else:
                signal = 0

        if self.lang == 'en':
            pos_dict = dict(nltk.pos_tag(tokens))
            pos = []
            stop_words = set(stopwords.words('english'))
            if len(wordRE.findall(word))>0:
                # total_simi = 0
                num_syn = 0
                num_syll = 0
                dic = pyphen.Pyphen(lang='en')
                for each in [w for w in wordRE.findall(word) if not w in stop_words]:
                    pos.append(pos_dict[each.lower()])
                    num_syn = num_syn + len(set(chain.from_iterable([word.lemma_names() for word in wordnet.synsets(each)])))
                    num_syll = num_syll + len(dic.inserted(each).split("-"))
                    # try:
                    #    top10_simi = self.word2vec_model.most_similar(each, topn=10)
                    #    for item in top10_simi:
                    #        total_simi = total_simi + item[1]
                    # except KeyError as err:
                        # print(err.args)

            else:
                pos.append("Other")
                num_syn = 0
                num_syll = 0
                #total_simi=0

            encoder_pos = {"CC":0, "CD":1, "DT":2, "EX":3, "FW":4, "IN":5,
                           "JJ":6, "JJR":7, "JJS":8, "LS":9, "MD":10,
                           "NN":11, "NNS":12, "NNP":13, "NNPS":14, "PDT":15,
                           "POS":16, "PRP":17, "PRP$":18, "RB":19, "RBR":20,
                           "RBS":21, "RP":22, "TO":23, "UH":24, "VB":25,
                           "VBD":26, "VBG":27, "VBN":28, "VBP":29, "VBZ":30,
                           "WDT":31, "WP":32, "WP$":33, "WRB":34, "Other":35}

            # print(filtered_sentence)
            # y4 = self.word2vec_model.doesnt_match(filtered_sentence)
            # print(y4)
            # print(total_simi , encoder_pos[pos[0]])
            if len(pos) == 0:
                pos = ["Other"]

            # print(len_chars, len_tokens, signal, freq_word, encoder_pos[pos[0]]/10)
            return [len_chars, len_tokens, signal, freq_word, encoder_pos[pos[0]], num_syn, num_syll]

        else:
            stop_words = set(stopwords.words('spanish'))
            num_syll = 0
            num_syn = 0
            dic = pyphen.Pyphen(lang='es')
            for each in [w for w in wordRE.findall(word) if not w in stop_words]:
                num_syll = num_syll + len(dic.inserted(each).split("-"))
                num_syn = num_syn + len(
                    set(chain.from_iterable([word.lemma_names() for word in wordnet.synsets(each, lang='spa')])))

            return [len_chars, len_tokens, signal, freq_word, num_syll]

    def train(self, trainset):
        X = []
        y = []
        title = "Learning Curves (GradientBoostingClassifier)"
        for sent in trainset:
            X.append(self.extract_features(sent['target_word'], sent['sentence']))

            if self.type == 'classify':
                y.append(sent['gold_label'])
            else:
                y.append(sent['gold_prob'])

        if self.type == 'classify':
            self.model.fit(X, y)
            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            plot_learning_curve(self.model, title, X, y, ylim=(0.76, 0.813), cv=cv, n_jobs=4)

            plt.show()
        else:
            self.model.fit(np.array(X).astype(float), np.array(y).astype(float))


    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'], sent['sentence']))

        return self.model.predict(X)
