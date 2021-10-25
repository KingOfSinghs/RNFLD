import os
import numpy as np
from collections import Counter
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# load filepaths with labels
def load_data(config):
    pos_path = config.POS_PATH
    neg_path = config.NEG_PATH

    pos = sorted([os.path.join(pos_path, i) for i in os.listdir(pos_path)])
    neg = sorted([os.path.join(neg_path, i) for i in os.listdir(neg_path)])

    pos, neg = np.array(pos), np.array(neg)

    pos_y = np.ones(len(pos))
    neg_y = np.zeros(len(neg))

    x = np.concatenate((pos, neg))
    y = np.concatenate((pos_y, neg_y))

    print('[INFO] Original dataset shape %s' % Counter(y))

    x, y = oversampler(x, y) # class imbalance exists

    print('[INFO] Resampled dataset shape %s' % Counter(y))

    x, y = shuffle(x, y)

    data = split_ds(x, y, split=config.SPLIT)
    return data


def split_ds(x, y, split):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split[2]) # train - test split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=split[1]) # val split

    print(f'\n[INFO] TRAINING IMAGES: {len(x_train)}')
    print(f'[INFO] TESTING IMAGES: {len(x_test)}')
    print(f'[INFO] VALIDATION IMAGES: {len(x_val)}\n')

    assert len(x_train) / len(x_test) <= 8

    # organize into dictionary
    data = dict(
        x_train = x_train, y_train = y_train,
        x_test = x_test, y_test = y_test,
        x_val = x_val, y_val = y_val
    )

    return data

def oversampler(X, y):
    X = X.reshape(-1,1) # for oversampler

    ros = RandomOverSampler(sampling_strategy='minority', random_state=25)
    X_res, y_res = ros.fit_resample(X, y)

    return X_res.flatten(), y_res


