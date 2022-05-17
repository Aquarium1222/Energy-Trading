import utils.initutils as initutils


class Config:
    TRAINING_DATA_DIR = './download/training_data/'
    C_DS_PATH = './sample_data/consumption.csv'
    G_DS_PATH = './sample_data/generation.csv'
    DEVICE = initutils.get_device()

    INCOMING_DAYS = 7
    HOUR_OF_DAY = 24
    OUTPUT_DAYS = 1
    TEST_SIZE = 0.2

    BATCH_SIZE = 8
    LR = 0.0002
