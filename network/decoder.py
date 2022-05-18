import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.raise_fc = nn.Linear(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.down_fc = nn.Linear(hid_dim, output_dim)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.raise_fc(x))
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        prediction = self.down_fc(output.squeeze(0))
        return prediction, hidden, cell
