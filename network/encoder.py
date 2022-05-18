import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.raise_fc = nn.Linear(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

    def forward(self, x):
        embedding = self.dropout(self.raise_fc(x))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell
