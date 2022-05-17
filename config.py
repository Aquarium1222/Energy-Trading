import utils.initutils as initutils


class Config:
    TRAINING_DATA_DIR = './download/training_data/'
    C_DS_PATH = './sample_data/consumption.csv'
    G_DS_PATH = './sample_data/generation.csv'
    SCALER_DIR = './'
    SCALER_NAME = 'scaler.pkl'

    DEVICE = initutils.get_device()

    HOUR_OF_DAY = 24
    DAY_OF_WEEK = 7
    INCOMING_DAYS = 7
    OUTPUT_DAYS = 1
    TEST_SIZE = 0.2

    BATCH_SIZE = 8
    LR = 0.0002
