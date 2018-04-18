from ml.ris.gastro import playground
from text import common
import json


def predict(x):
    path = common.get_latest_file('/home/giga')
    model = common.load_pickle(path)
    return model.predict(x)


def validate(texts, labels):
    result = {}

    result['segments'] = [
        {
            'text': text,
            'label': label,
            'label_text': label2text(label)
        }
        for text, label in zip(texts, labels)
    ]

    result['overall_label'] = max(labels)
    if result['overall_label'] == 3:
        if [1, 2] in labels:
            pass

    result['overall_label_text'] = label2text(result['overall_label'])

    return json.dump(result)


def label2text(label):
    label_dict = {
        0: 'Void',
        1: 'Negative',
        2: 'Somewhat Negative',
        3: 'Ambiguous',
        4: 'Somewhat Positive',
        5: 'Positive'
    }
    return label_dict[label]


if __name__ == '__main__':
    print('Running playground, not meant for production.')

    playground.create_model()
    predict(['abc es gibt ein gorsses tumor', '123 123 aaa vvv kein tumor'])
