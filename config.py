import utils.initutils as initutils


class Config:
    TRAINING_DATA_DIR = './download/training_data/'
    C_DS_PATH = './sample_data/consumption.csv'
    G_DS_PATH = './sample_data/generation.csv'
    SCALER_DIR = './'
    SCALER_NAME = 'scaler.pkl'
    MODEL_NAME = 'seq2seq.hdf5'

    DEVICE = initutils.get_device()

    HOUR_OF_DAY = 24
    DAY_OF_WEEK = 7
    INCOMING_DAYS = 7
    OUTPUT_DAYS = 1
    TEST_SIZE = 0.2

    BATCH_SIZE = 8
    LR = 0.0002
    EPOCH = 300

    INPUT_DIM = 5
    OUTPUT_DIM = 5
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
