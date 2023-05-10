import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_rnn_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_rnn_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_rnn_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_rnn_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)
        # output[0] gets rid of the extra dimension
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class ModelBuilder:
    def __init__(self, english_words, french_words, hidden_size=256):
        self.input_size = len(english_words)
        self.output_size = len(french_words)
        self.hidden_size = hidden_size

    def build(self):
        encoder = Encoder(self.input_size, self.hidden_size)
        decoder = Decoder(self.hidden_size, self.output_size)
        return encoder, decoder

