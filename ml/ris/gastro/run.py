import random
import dill
import os
from ml.ris.gastro import playground


def predict(x):
    return random.randint(0,6)


def demo():
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


if __name__ == '__main__':
    print('Running playground, not meant for production.')

    playground.create_model()
