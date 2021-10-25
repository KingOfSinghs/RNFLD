import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from generator import DataGenerator
from cam import cam

def load_cam(n=10):
    cam_dir = 'CAM\\test'
    cam_set = [os.path.join(cam_dir, i) for i in os.listdir(cam_dir)]
    return cam_set[:n]

def disp_confusion(conf_matrix):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap='Blues', alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    plt.close()

def disp_metrics(name, y_test, y_pred):
    # conf_matrix = confusion_matrix(y_test, y_pred)
    # disp_confusion(conf_matrix)

    print(30 * '-')
    print(f'[INFO]Report for {name} model on binary classification')

    print('Accuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    metrics = []
    metrics.append(name)
    metrics.append(accuracy_score(y_test, y_pred))
    metrics.append(precision_score(y_test, y_pred, average='weighted'))
    metrics.append(recall_score(y_test, y_pred, average='weighted'))
    metrics.append(f1_score(y_test, y_pred, average='weighted'))

    df = pd.DataFrame(data=[metrics], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    print(df.to_string(index=False))

    print('\nBASIC ANALYTICS')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f'SENSITIVITY SCORE (actual positive): {sensitivity}')
    print(f'SPECIFICITY SCORE (actual negative): {specificity}')
    print(30 * '-')

def test(model, data, config):
    x_test, y_test = data['x_test'], data['y_test']

    print(f'[INFO] TESTING MODEL ON {len(x_test)} IMAGES...\n')

    test_gen = DataGenerator(x_test, y_test, config=config, do_augment=False)

    predictions = model.predict(test_gen)

    vals = predictions >= 0.5
    zvals = predictions < 0.5
    predictions[vals] = 1
    predictions[zvals] = 0
    y_pred = predictions

    y_test = np.array(y_test)

    disp_metrics(model.name, y_test, y_pred)

    if config.SAVE_HEATMAPS and model.run == 3:
        print('\n[INFO] Generating Heatmaps...')
        try:
            cam(load_cam(), model, config)
        except Exception as e:
            print('[ERROR] COULD NOT GENERATE CAM: ' + str(e))
