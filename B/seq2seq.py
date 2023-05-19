import random

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, rnn_num_layers, dropout_rate, pretrained_embeddings):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_num_layers = rnn_num_layers

        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(input_size, embedding_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        # self.gru = nn.GRU(hidden_size, hidden_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, rnn_num_layers, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        # TODO: add dropout
        word_embedding = self.embedding(input)
        # output, hidden = self.gru(embedded, hidden)
        outputs, (hidden, cell) = self.rnn(word_embedding)
        # return output, hidden
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_size, rnn_num_layers, dropout_rate, pretrained_embeddings):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.rnn_num_layers = rnn_num_layers
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(output_size, embedding_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = nn.LSTM(embedding_size, hidden_size, rnn_num_layers, dropout=dropout_rate)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input, hidden, cell):
        # TODO: add dropout
        input = input.unsqueeze(0)
        word_embedding = self.embedding(input)
        output, (hidden, cell) = self.rnn(word_embedding, (hidden, cell))
        # output[0] gets rid of the extra dimension
        output = self.out(output[0])
        return output, hidden, cell


class ModelBuilder:
    def __init__(self, english_words, french_words, hidden_size=512, enc_embedding_size=256, dec_embedding_size=256,
                 rnn_num_layers=2, enc_dropout_rate=0.5, dec_dropout_rate=0.5, pretrained_embeddings=None):
        self.input_size = len(english_words)
        self.output_size = len(french_words)
        self.hidden_size = hidden_size
        self.enc_embedding_size = enc_embedding_size

        self.dec_embedding_size = dec_embedding_size

        self.rnn_num_layers = rnn_num_layers

        self.enc_dropout_rate = enc_dropout_rate

        self.dec_dropout_rate = dec_dropout_rate

        self.pretrained_en_embeddings = pretrained_embeddings[
            "english_embeddings"] if pretrained_embeddings is not None else None
        self.pretrained_fr_embeddings = pretrained_embeddings[
            "french_embeddings"] if pretrained_embeddings is not None else None

    def build(self):
        encoder = Encoder(self.input_size, self.hidden_size, self.enc_embedding_size, self.rnn_num_layers,
                          self.enc_dropout_rate, self.pretrained_en_embeddings)
        decoder = Decoder(self.hidden_size, self.output_size, self.dec_embedding_size, self.rnn_num_layers,
                          self.dec_dropout_rate, self.pretrained_fr_embeddings)
        return encoder, decoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.rnn_num_layers == decoder.rnn_num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        print("src shape: ", src.shape)
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        outputs[0, :, 1] = 1
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
