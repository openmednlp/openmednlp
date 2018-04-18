from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

import tensorflow as tf
import keras
import numpy as np
import text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

max_sentence_length = 50
_count_vectorizer = None


def get_count_vectorizer(tokenized_sentences):
    cv = CountVectorizer()
    cv.fit(tokenized_sentences)

    global _count_vectorizer
    _count_vectorizer = cv

    return cv


def get_sentence_count_vector(sentence):
    return _count_vectorizer.transform(sentence).indices


def get_indexed_data(df):
    X, word_dict = text.index_tokenized_sentences(
        df[text.PROCESSED_TOKENS]
    )
    y = df[text.BINARY_CLASS]

    return X, y, word_dict


def get_indexed_and_padded_data():
    np.random.seed(42)

    df_processed = text.process_df(
        df,
        X_column_name='sentence',
        y_column_name='sentence_sentiment',
        do_tokenize_sentences=False,
        do_stem=True,
        do_compound_split=True
    )

    X, y, word_dict = get_indexed_data(df_processed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=df_processed[text.BINARY_CLASS]
    )

    X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)

    return X_train, X_test, y_train, y_test, word_dict


def run_lstm():
    print('Start LSTM')

    X_train, X_test, y_train, y_test, word_dict = get_indexed_and_padded_data()

    total_words = len(word_dict) + 1  # we added zero padding, where dict ints start from 1

    embedding_vector_length = 10
    model = Sequential()
    model.add(Embedding(total_words, embedding_vector_length, input_length=max_sentence_length))
    model.add(Dropout(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


def run_convolutional():
    print('Convolutional model')

    X_train, X_test, y_train, y_test, word_dict = get_indexed_and_padded_data()

    # Run Convolutional model
    embedding_vector_length = 10
    model = Sequential()
    model.add(Embedding(len(word_dict) + 1, embedding_vector_length, input_length=max_sentence_length))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=10, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, epochs=10, batch_size=1)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    print('Done')

    return model


def experiment():
    pass


if __name__ == '__main__':
    '''This example demonstrates the use of Convolution1D for text classification.
        Gets to 0.89 test accuracy after 2 epochs.
        90s/epoch on Intel i5 2.4Ghz CPU.
        10s/epoch on Tesla K40 GPU.
        '''



    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # set parameters:
    max_features = 5000
    maxlen = 400
    batch_size = 32
    embedding_dims = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 2




    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
