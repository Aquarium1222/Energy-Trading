import random
import torch
import torch.nn as nn

from config import Config


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        batch_size = y.shape[1]
        trg_len = y.shape[0]
        trg_output_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_output_size).to(Config.DEVICE)
        hidden, cell = self.encoder(x)
        dec_x = x[-1]
        for t in range(trg_len):
            output, hidden, cell = self.decoder(dec_x, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            dec_x = y[-1] if teacher_force else output

        return outputs
