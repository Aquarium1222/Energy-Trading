import utils.initutils as initutils


class Config:
    TRAINING_DATA_DIR = './download/training_data/'
    C_DS_PATH = './sample_data/consumption.csv'
    G_DS_PATH = './sample_data/generation.csv'
    SCALER_DIR = './'
    SCALER_NAME = 'scaler.pkl'
    CHECKPOINT_DIR = './checkpoint/'
    MODEL_NAME = 'checkpoint_200.hdf5'

    DEVICE = initutils.get_device()

    HOUR_OF_DAY = 24
    DAY_OF_WEEK = 7
    INCOMING_DAYS = 7
    OUTPUT_DAYS = 1
    TEST_SIZE = 0.2

    BATCH_SIZE = 64
    LR = 0.002
    EPOCH = 300

    INPUT_DIM = 5
    OUTPUT_DIM = 5
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    GEN_LOSS = 0.00067
    CON_LOSS = 0.00067

    BUY_INIT_PRICE = 2.52
    SELL_INIT_PRICE = 2.52

    THRESHOLD = 0.7
