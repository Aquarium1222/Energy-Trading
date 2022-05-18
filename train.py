import torch
import torch.nn as nn
import torch.optim as optim

import utils.datautils as datautils
import utils.initutils as initutils
from config import Config
from dataset.energy_dataset import EnergyDataset
from network.encoder import Encoder
from network.decoder import Decoder
from network.seq2seq import Seq2Seq


def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(loader):
        x = x.to(torch.float32).to(Config.DEVICE).permute(1, 0, 2)
        y = y.to(torch.float32).to(Config.DEVICE).permute(1, 0, 2)

        optimizer.zero_grad()
        output = model(x, y)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def test(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(torch.float32).to(Config.DEVICE).permute(1, 0, 2)
            y = y.to(torch.float32).to(Config.DEVICE).permute(1, 0, 2)

            output = model(x, y, 0)
            loss = criterion(output, y)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


if __name__ == '__main__':
    ds = EnergyDataset(Config.TRAINING_DATA_DIR)
    train_indices, test_indices = datautils.make_train_test_indices(ds)
    train_loader = datautils.make_dataloader(ds, train_indices, Config.BATCH_SIZE)
    test_loader = datautils.make_dataloader(ds, test_indices, Config.BATCH_SIZE)

    encoder = Encoder(Config.INPUT_DIM, Config.ENC_EMB_DIM, Config.HID_DIM, Config.N_LAYERS, Config.ENC_DROPOUT)
    decoder = Decoder(Config.OUTPUT_DIM, Config.DEC_EMB_DIM, Config.HID_DIM, Config.N_LAYERS, Config.DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder).to(Config.DEVICE)
    model.apply(initutils.weights_init)
    print(model)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for e in range(Config.EPOCH):
        print('Epoch: {}'.format(e))
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = test(model, test_loader, criterion)
        print('--Train loss: {}'.format(train_loss))
        print('--Test loss: {}'.format(test_loss))
    torch.save(model, Config.MODEL_NAME)
