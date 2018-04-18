from text import collection, viz, common, process, feature
from ml.ris.gastro.risnlp.models.standard import bnb


import pandas as pd
import numpy as np


config = common.config_to_namedtuple()


def get_tfidf_data(pickle_path, limit=None):
    vectorizer_pickle_path = None
    if pickle_path:
        vectorizer_pickle_path = common.create_pickle_path(
            pickle_path, 'tfidf_vectorizer'
        )

    # Load data
    sentence_granularity = config.GRANULARITY.sentences
    split_level = collection.IMPRESSION_ID

    (X, X_validate,
     y, y_validate) = load_default_train_test(
        sentence_granularity,
        split_level,
        limit=limit
    )

    # Create new vectorizer

    vectorizer = feature.train_tfidf_vectorizer(
        X,
        persist_path=vectorizer_pickle_path
    )
    X_vec = vectorizer.transform(X)
    X_validate_vec = vectorizer.transform(X_validate)

    return X_vec, X_validate_vec, y, y_validate


def load_default_train_test(granularity_key, split_by_field, limit=None):
    # TODO: make returns uniform, this returns series, something returns lists.
    granularity_field_map = {
        config.INPUT.sentences: collection.SENTENCE,
        config.INPUT.impressions: collection.IMPRESSION
    }

    csv_path = getattr(config.INPUT, granularity_key)
    input_field_name = granularity_field_map[csv_path]
    target_field_name = collection.GROUND_TRUTH

    df = collection.file_to_df(csv_path)

    # Randomly subsample
    if limit is not None:
        df = df[
            df[split_by_field]
            .isin(
                np.random.choice(
                    df[split_by_field].unique(),
                    limit,
                    replace=False
                )
            )
        ]

    do_tokenize = config.TEXT_OPERATIONS.word_tokenize.lower() == 'true'
    do_stem = config.TEXT_OPERATIONS.stem.lower() == 'true'
    shortest_word = int(config.TEXT_OPERATIONS.shortest_word)
    chars_to_remove = config.TEXT_OPERATIONS.remove_chars
    replacement_char = str(config.TEXT_OPERATIONS.replacement_char)
    do_compound_split = config.TEXT_OPERATIONS.compound_split.lower() == 'true'
    return_as_tokens = config.TEXT_OPERATIONS.return_tokens.lower() == 'true'

    process.pipeline_df(
        df,
        input_field_name,
        do_tokenize=do_tokenize,
        min_word_len=shortest_word,
        do_stem=do_stem,
        chars_to_remove=chars_to_remove,
        replacement_char=replacement_char,
    )

    return collection.train_validate_split_df(
        df,
        config.FIELD_NAME.processed,
        target_field_name,
        split_by_field
    )


def create_model():
    pickles_dir = 'models'

    (X_train_vectorized,
     X_test_vectorized,
     y_train,
     y_test) = get_tfidf_data(pickles_dir)

    pickle_path = common.create_pickle_path(pickles_dir, 'rfe_bnb_v1')
    model = bnb(
        X_train_vectorized,
        X_test_vectorized,
        y_train,
        y_test,
        persist_path=pickle_path
    )

    # Test
    y_hat = model.predict(X_test_vectorized)

    df_test = pd.DataFrame(
        {
            'X': X_test_vectorized,
            'y': y_test,
            'y_hat': y_hat
        }
    )
    print(df_test.sample(5))

    viz.show_stats(y_test, y_hat)
    viz.plot_confusion_reports(y_test, y_hat)

    csv_path = getattr(config.INPUT, config.GRANULARITY.sentences)
    df = collection.file_to_df(csv_path)
    viz.word_clouds(df, 'misc')


import spacy
import string
from spacy.lang.de.stop_words import STOP_WORDS
import time
nlp = spacy.load('de')


# Clean text before feeding it to spaCy
punctuations = string.punctuation


# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
#TODO: add to risnlp
def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in STOP_WORDS and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


def nn_experiment(x,x_val, y, y_val):
    # Parse documents and print some info

    x_clean = cleanup_text(x)

    print('Parsing documents...')
    start = time()
    train_vec = []
    for doc in nlp.pipe(x, batch_size=500):
        if doc.has_vector:
            train_vec.append(doc.vector)
        # If doc doesn't have a vector, then fill it with zeros.
        else:
            train_vec.append(np.zeros((128,), dtype="float32"))

    # train_vec = [doc.vector for doc in nlp.pipe(train_cleaned, batch_size=500)]
    train_vec = np.array(train_vec)



if __name__ == '__main__':
    create_model()