import torch
import torch.nn as nn
import torch.optim as optim

import utils.datautils as datautils
import utils.initutils as initutils
import utils.visualizeutils as visualizeutils
from config import Config
from dataset.energy_dataset import EnergyDataset
from network.encoder import Encoder
from network.decoder import Decoder
from network.seq2seq import Seq2Seq


def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_day_loss = 0
    epoch_hour_loss = 0
    epoch_weekday_loss = 0
    epoch_gen_loss = 0
    epoch_con_loss = 0
    for i, (x, y) in enumerate(loader):
        x = x.to(torch.float32).to(Config.DEVICE).permute(1, 0, 2)
        y = y.to(torch.float32).to(Config.DEVICE).permute(1, 0, 2)

        optimizer.zero_grad()
        output = model(x, y)
        # loss = criterion(output, y)
        day_loss = criterion(output[:, :, 0], y[:, :, 0])
        hour_loss = criterion(output[:, :, 1], y[:, :, 1])
        weekday_loss = criterion(output[:, :, 2], y[:, :, 2])
        gen_loss = criterion(output[:, :, 3], y[:, :, 3])
        con_loss = criterion(output[:, :, 4], y[:, :, 4])

        epoch_day_loss += day_loss.item()
        epoch_hour_loss += hour_loss.item()
        epoch_weekday_loss += weekday_loss.item()
        epoch_gen_loss += gen_loss.item()
        epoch_con_loss += con_loss.item()
        loss = day_loss + hour_loss + weekday_loss + gen_loss + con_loss

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if i % int(len(loader) / 100) == 0:
            print('\r--({:.2f}%) Loss: {}'.format(i / (int(len(loader))) * 100, loss.item()), end='', flush=True)

    print('')
    print('--Loss day: {}'.format(epoch_day_loss / len(loader)))
    print('--Loss hour: {}'.format(epoch_hour_loss / len(loader)))
    print('--Loss weekday: {}'.format(epoch_weekday_loss / len(loader)))
    print('--Loss generation: {}'.format(epoch_gen_loss / len(loader)))
    print('--Loss consumption: {}'.format(epoch_con_loss / len(loader)))
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

    optimizer = optim.Adam(model.parameters(), lr=Config.LR, betas=(0.9, 0.999), eps=1e-08)
    criterion = nn.MSELoss()

    train_loss_list = []
    test_loss_list = []

    for e in range(Config.EPOCH):
        print('Epoch: {}'.format(e))
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = test(model, test_loader, criterion)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print('--Train loss: {}'.format(train_loss))
        print('--Test loss: {}'.format(test_loss))
        if e % 10 == 0:
            torch.save(model, Config.CHECKPOINT_DIR + 'checkpoint_' + str(e) + '.hdf5')

    torch.save(model, Config.MODEL_NAME)
    visualizeutils.plot_loss(range(Config.EPOCH), train_loss_list, 'Train_loss', True)
    visualizeutils.plot_loss(range(Config.EPOCH), test_loss_list, 'Test_loss', True)
