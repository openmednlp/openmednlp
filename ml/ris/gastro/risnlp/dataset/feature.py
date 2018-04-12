import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Phrases, Word2Vec
from ml.ris.gastro.risnlp.dataset import common


def train_tfidf_vectorizer(X, persist_path=None):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X)

    common.save_pickle(tfidf_vectorizer, persist_path)

    return tfidf_vectorizer


def train_word2vec_vectorizer(tokenized_sentences,
                              group_by_phrases=False,
                              persist_path=None):
    # tokenized_sentences :
    #   input format is a list of sentences representet as list of tokens
    #   e.g. [
    #           [['sentence'],['one']],
    #           [['sentence'],['two']]
    #        ]

    # TODO: this could be also streamed directly from disk
    # TODO: if not tokenized
    # Try detecting phrases, not very good for our purpose atm
    if group_by_phrases:
        bigram_transformer = Phrases(tokenized_sentences)
        tokenized_sentences = bigram_transformer[tokenized_sentences]

    model = Word2Vec(tokenized_sentences, window=5, size=200)
    print('Learned vocab len: ', len(model.wv.vocab))

    common.save_pickle(model, persist_path)

    return model


def load_pickle(pickle_path):
    return pickle.load(pickle_path)