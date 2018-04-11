import random
import dill
import os


def predict(x):
    return random.randint(0,6)


if __name__ == '__main__':
    print('Creating model...')

    dill_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'models/model.dill'
    )

    with open(dill_file_path, 'wb') as f:
        dill.dump(predict, f)

    with open(dill_file_path, 'rb') as f:
        model_fun = dill.load(f)

    print(
        'The magic ball predicts {}'
        .format(
            model_fun(None)
        )
    )
