from os import path, listdir
import pandas as pd
from nltk import edit_distance
import csv
from sklearn.model_selection import train_test_split

from ml.ris.gastro.risnlp.dataset import process

IMPRESSION_ID = 'impression_id'
IMPRESSION = 'impression'
SENTENCE_ID = 'sentence_id'
SENTENCE = 'sentence'
GROUND_TRUTH = 'ground_truth'
PROCESSED = 'processed'


def _get_file_type(data_path):
    _, file_type = path.splitext(data_path)
    return str.lower(file_type)


def file_to_df(data_path):
    file_type = _get_file_type(data_path)

    if file_type == '.csv':
        return pd.read_csv(data_path)
    else:
        raise NotImplementedError('This file type is not supported')


def _get_report_paths(dir_path):
    return [
        path.join(dir_path, file_name)
        for file_name in listdir(dir_path)
        if path.isfile(
            path.join(dir_path, file_name)
        )
           and (
                   path.splitext(file_name)[1] == '.txt'
           )
    ]


def extract_impression_from_text(text):
    impression_found = False
    impression_extracted = False
    impression_lines = []

    for line in text.splitlines():
        if not impression_found:
            # TODO: maybe use similarity measure in case there is a typo
            if edit_distance(line.strip().lower(), 'beurteilung') < 3:
                impression_found = True
            continue

        if impression_extracted:
            break

        # TODO: maybe use similarity measure in case there is a typo
        line_is_empty = line.strip().lower() == ''
        new_text_segment_detected = edit_distance(
            line.strip().lower(), 'beilagen zum befund'
        )
        if new_text_segment_detected < 3 or line_is_empty:
            impression_extracted = True
            continue

        impression_lines.append(line)

    impression = None
    if len(impression_lines) > 0:
        impression = '\n'.join(impression_lines)

    return impression


def extract_impression_from_file(file_path):
    with open(file_path, "r", encoding='utf8') as f:
        text = f.read()

    return extract_impression_from_text(text)


def extract_impressions_from_files(
        reports_dir_path,
        persist_path=None):

    report_paths = _get_report_paths(reports_dir_path)

    impression_ids = []
    impressions = []

    for file_path in report_paths:
        impression = extract_impression_from_file(file_path)
        if impression is None:
            continue
        impressions.append(impression)
        impression_ids.append(path.basename(file_path).split('-')[0])

    df = pd.DataFrame(
        {
            IMPRESSION_ID: impression_ids,
            IMPRESSION: impressions
        }
    )

    if persist_path is not None:
        df.to_csv(persist_path, quoting=csv.QUOTE_NONNUMERIC)

    return df


def balance_df(df, balance_type='random_upsample'):
    if balance_type is None:
        print('balance: no type defined, exiting...')
        return df

    positive_count = sum(df.binary_class)
    negative_count = len(df) - positive_count
    difference = abs(negative_count - positive_count)

    if difference < 2:
        print('balance: nothing to do, exiting...')
        return df

    if balance_type.lower() == 'random_upsample':
        is_positive_dominant = positive_count > negative_count
        if is_positive_dominant:
            minority = df[df.binary_class]
        else:
            minority = df[~df.binary_class]

        upsampled = minority.sample(n=difference, replace=True)
        return df.append(upsampled, ignore_index=True)

    raise ValueError('Wrong balancing type')


def _impression_to_sentences(impression_id, impression, ground_truth):
    result = []

    enumerated_sentences = enumerate(process.sentence_tokenize(impression))

    for sentence_id, sentence in enumerated_sentences:
        result.append((impression_id, sentence_id, sentence, ground_truth))
    return result


def impressions_to_sentences(impressions, impression_ids=None, ground_truths=None, persist_path=None):
    sentences_matrix = []
    # TODO: Possible changes - 1. use real impression ids, 2. use df as input. Probably need another function.

    if ground_truths is None:
        ground_truths = [''] * len(impressions)

    if impression_ids is None:
        impression_ids = range(len(impressions))

    inputs = zip(impression_ids, impressions, ground_truths)

    for impression_id, impression, ground_truth in inputs:
        sentences_matrix.extend(
            _impression_to_sentences(
                impression_id,
                impression,
                ground_truth
            )
        )

    columns = [
        IMPRESSION_ID,
        SENTENCE_ID,
        SENTENCE,
        GROUND_TRUTH
    ]
    df = pd.DataFrame(sentences_matrix, columns=columns)

    if persist_path is not None:
        df.to_csv(persist_path, quoting=csv.QUOTE_NONNUMERIC)

    return df


def index_tokenized_sentences(tokenized_sentences):
    # TODO: I needed this for word2vec, but maybe not needed anymroe
    word_dict = dict()
    X = []
    i = 0
    for tokenized_sentence in tokenized_sentences:
        indexed_sentence = []
        for word in tokenized_sentence:
            if word not in word_dict:
                i += 1
                word_dict[word] = i
            indexed_sentence.append(word_dict[word])
        X.append(indexed_sentence)
    return X, word_dict


def train_validate_split_df(
        df,
        input_field_name,
        target_field_name,
        split_by_field,
        test_size=0.3,
        random_state=None):

    df_grouped = df.groupby(split_by_field)
    input_ids = list(df_grouped.groups.keys())
    group_max_targets = list(df_grouped[target_field_name].max())

    train_ids, test_ids, _, _ = train_test_split(
        input_ids,
        group_max_targets,
        test_size=test_size,
        random_state=random_state,
        # stratify=group_max_targets # Assumption: data is balanced
        # TODO: cannot stratify if there is only one sample
    )

    train_dataset = df[
        df[split_by_field].isin(train_ids)
    ]
    test_dataset = df[
        df[split_by_field].isin(test_ids)
    ]

    return (train_dataset[input_field_name],
            test_dataset[input_field_name],
            train_dataset[target_field_name],
            test_dataset[target_field_name]
            )


def reports_to_train_test(reports_dir, persist_path=None):
    # This does not make sense, because there are no targets given,
    # but is a good example of the whole process.
    from ml.ris.gastro.risnlp.dataset import process

    df_impressions = extract_impressions_from_files(reports_dir, persist_path)

    # TODO: Needs new persist path
    df_sentences = impressions_to_sentences(
        df_impressions[IMPRESSION],
        persist_path
    )

    # TODO: Needs new persist path
    process.pipeline_df(
        df_sentences,
        SENTENCE,
        persist_path
    )

    return train_validate_split_df(
        df_sentences,
        PROCESSED,
        GROUND_TRUTH,
        SENTENCE_ID
    )