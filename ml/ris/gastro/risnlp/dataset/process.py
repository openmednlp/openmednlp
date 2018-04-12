from nltk.tokenize import word_tokenize
import nltk.data
from nltk import SnowballStemmer

# TODO: make different backends - so far only NLTK is supported
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

PROCESSED = 'processed'

_stemmer = SnowballStemmer('german')
_sentence_tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')


def _error_on_not_list(input):
    # TODO: make on iterable
    if type(input) is not list:
        raise ValueError(
            'Expected type list, but got ' + str(type(input))
        )


def replace_chars(texts, chars_to_replace, replacement_char=' '):
    # Replace chars in thext with another char.
    _error_on_not_list(texts)

    if not chars_to_replace:
        return texts

    processed_texts = []
    for text in texts:
        for unwanted_char in chars_to_replace:
            text.replace(unwanted_char, replacement_char)
        processed_texts.append(text)
    return processed_texts


def tokenize(texts):
    # Tokenize texts.
    _error_on_not_list(texts)

    processed_texts = []
    for text in texts:
        tokenized_text = word_tokenize(text)
        processed_texts.append(tokenized_text)
    return processed_texts


def remove_short(texts, min_word_len):
    # Remove short words.
    _error_on_not_list(texts)
    if not min_word_len:
        return texts

    processed_text = []
    texts = tokenize(texts)
    for text in texts:
        processed_text.append(
            ' '.join(
                [word for word in text if len(word) >= min_word_len]
            )
        )

    return processed_text


def stem(texts):
    # Stem texts (default language is German)
    _error_on_not_list(texts)
    processed_text = []
    texts = [_stemmer.stem(text) for text in texts]
    return texts


def viperizer(texts, vip_words):
# Adds vip words to the end of the text if found there is a match.
    if vip_words is None or not vip_words:
        return texts

    for text in texts:
        vip_words_found = []
        for vip_word in vip_words:
            if vip_word in text:
                vip_words_found.append(vip_word)
        text += ' '.join(vip_words_found)
    return texts


def pipeline(
        texts,
        min_word_len=5,
        do_stem=True,
        chars_to_remove=None,
        replacement_char=' ',
        vip_words=None,
        do_tokenize=False):
    # texts:
    #   list of texts
    # min_word_len:
    #   Removes words shorter than given value.
    # chars_to_remove:
    #   chars_to_remove=['-', '.'] and chars_to_remove='-.'
    #   does the same, removes '-' and '.' from text.
    # replacement_char:
    #   If anything is given for remove chars it will
    # replacement_char:
    #   If anything is given for remove chars it will
    #   replace them with this character.
    # vip_words:
    #   Word patterns that will be appended to the end
    #   of the text if found.
    #   e.g.
    #   ['karzinom', 'tumor', 'metastas', 'krebs', 'sarkom', 'malign']
    # do_tokenize:
    #   do_tokenize=True performs word tokenization and returns a
    #   list of tokens for each text, making a function return
    #   list of lists.

    texts = replace_chars(texts, chars_to_remove, replacement_char)

    texts = remove_short(texts, min_word_len)

    texts = viperizer(texts, vip_words)

    if do_stem:
        texts = stem(texts)

    if do_tokenize:
        texts = tokenize(texts)

    return texts


def pipeline_df(
        df,
        field_name,
        persist_path=None,
        min_word_len=5,
        do_stem=True,
        chars_to_remove=None,
        replacement_char=' ',
        vip_words=None,
        do_tokenize=False):
    # df:
    #   DataFrame containing the texts.
    # filed_name:
    #   Filed that contains texts in the DataFrame.

    df[PROCESSED] = pipeline(
        list(df[field_name]),
        min_word_len=min_word_len,
        do_stem=do_stem,
        chars_to_remove=chars_to_remove,
        replacement_char=replacement_char,
        vip_words=vip_words,
        do_tokenize=do_tokenize
    )

    if persist_path is not None:
        df.to_csv(persist_path)


def sentence_tokenize(text):
    _sentence_tokenizer.tokenize(text)

