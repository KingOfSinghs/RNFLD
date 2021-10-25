from datetime import datetime as dt
import tensorflow.keras.callbacks as tfcallbacks

# ------------------------------- CONFIG ------------------------------- #

MODEL = 'Xception'
TIMESTAMP = dt.now().strftime("%y-%m-%d_%H-%M-%S")

# --------- FILE --------- #
FILEPATH = 'D:\\Jatin\\RNFLD\\ORG-Data\\'
# NEG_PATH = FILEPATH + 'neg-norm'
# POS_PATH = FILEPATH + 'pos-norm'

NEG_PATH = FILEPATH + 'neg'
POS_PATH = FILEPATH + 'pos'

LOG_DIR = "logs/fit/" + TIMESTAMP + '_' + MODEL + 'nonnorm'
CSV_DIR = "csv/fit/" + TIMESTAMP + '_' + MODEL + 'nonnorm' +'.log'
WEIGHTS_DIR = 'weights/' + TIMESTAMP + '_' + MODEL + 'nonnorm' + '.h5'
CAM_DIR = 'CAM/' + TIMESTAMP + '_' + MODEL + 'nonnorm'

# --------- PARAMETERS --------- #
CLASSES = 1
CHANNELS = 3
IMG_DIM = (256, 256)
IMG_SHAPE = (256, 256, CHANNELS)
SPLIT = (0.70, 0.10, 0.20) # train, val, test

EPOCHS = 100
BATCH_SIZE = 16
STEPS_FACTOR = 1
LRS = (1, 1e-3), (2, 1e-4), (3, 1e-5) # (run,lr)

LOSS = 'binary_crossentropy'
METRICS = ['accuracy', 'Precision', 'Recall', 'AUC']

AUGMENT = True
WEIGHTS = None
SAVE_WEIGHTS = True
SAVE_HEATMAPS = True

# --------- CALLBACKS --------- #

TB = tfcallbacks.TensorBoard(
    log_dir=LOG_DIR)

ES = tfcallbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=20,
    verbose=1,
    restore_best_weights=True)

RLR = tfcallbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1,
    min_delta=0.0001)

CSV = tfcallbacks.CSVLogger(CSV_DIR)

CALLBACKS = [TB, ES, RLR, CSV]