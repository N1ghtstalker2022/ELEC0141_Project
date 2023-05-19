import random
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, rnn_num_layers, dropout_rate, pretrained_embeddings):
        """
        Encoder module of the sequence-to-sequence model.

        Args:
            input_size (int): Size of the input vocabulary.
            hidden_size (int): Size of the hidden state in the LSTM.
            embedding_size (int): Size of the word embedding.
            rnn_num_layers (int): Number of layers in the LSTM.
            dropout_rate (float): Dropout rate.
            pretrained_embeddings (torch.Tensor): Pretrained word embeddings (optional).

        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_num_layers = rnn_num_layers

        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(input_size, embedding_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)

        self.rnn = nn.LSTM(embedding_size, hidden_size, rnn_num_layers, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        """
        Forward pass of the encoder.

        Args:
            input (torch.Tensor): Input tensor representing a sequence of word indices.

        Returns:
            hidden (torch.Tensor): Hidden state of the encoder LSTM.
            cell (torch.Tensor): Cell state of the encoder LSTM.

        """
        word_embedding = self.embedding(input)
        outputs, (hidden, cell) = self.rnn(word_embedding)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_size, rnn_num_layers, dropout_rate, pretrained_embeddings):
        """
        Decoder module of the sequence-to-sequence model.

        Args:
            hidden_size (int): Size of the hidden state in the LSTM.
            output_size (int): Size of the output vocabulary.
            embedding_size (int): Size of the word embedding.
            rnn_num_layers (int): Number of layers in the LSTM.
            dropout_rate (float): Dropout rate.
            pretrained_embeddings (torch.Tensor): Pretrained word embeddings (optional).

        """
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
        """
        Forward pass of the decoder.

        Args:
            input (torch.Tensor): Input tensor representing a sequence of word indices.
            hidden (torch.Tensor): Hidden state of the decoder LSTM.
            cell (torch.Tensor): Cell state of the decoder LSTM.

        Returns:
            output (torch.Tensor): Output tensor representing predicted word scores.
            hidden (torch.Tensor): Hidden state of the decoder LSTM.
            cell (torch.Tensor): Cell state of the decoder LSTM.

        """
        input = input.unsqueeze(0)
        word_embedding = self.embedding(input)
        output, (hidden, cell) = self.rnn(word_embedding, (hidden, cell))
        output = self.out(output[0])
        return output, hidden, cell


class ModelBuilder:
    def __init__(self, english_words, french_words, hidden_size=512, enc_embedding_size=256, dec_embedding_size=256,
                 rnn_num_layers=2, enc_dropout_rate=0.5, dec_dropout_rate=0.5, pretrained_embeddings=None):
        """
        Builder class for constructing the Seq2Seq model.

        Args:
            english_words (list): List of English vocabulary words.
            french_words (list): List of French vocabulary words.
            hidden_size (int): Size of the hidden state in the LSTM (default: 512).
            enc_embedding_size (int): Size of the input word embedding (default: 256).
            dec_embedding_size (int): Size of the output word embedding (default: 256).
            rnn_num_layers (int): Number of layers in the LSTM (default: 2).
            enc_dropout_rate (float): Dropout rate for the encoder (default: 0.5).
            dec_dropout_rate (float): Dropout rate for the decoder (default: 0.5).
            pretrained_embeddings (dict): Dictionary of pretrained word embeddings (optional).

        """
        self.input_size = len(english_words)
        self.output_size = len(french_words)
        self.hidden_size = hidden_size
        self.enc_embedding_size = enc_embedding_size
        self.dec_embedding_size = dec_embedding_size
        self.rnn_num_layers = rnn_num_layers
        self.enc_dropout_rate = enc_dropout_rate
        self.dec_dropout_rate = dec_dropout_rate
        self.pretrained_en_embeddings = pretrained_embeddings["english_embeddings"] if pretrained_embeddings else None
        self.pretrained_fr_embeddings = pretrained_embeddings["french_embeddings"] if pretrained_embeddings else None

    def build(self):
        """
        Build the Seq2Seq model.

        Returns:
            encoder (Encoder): The encoder module.
            decoder (Decoder): The decoder module.

        """
        encoder = Encoder(self.input_size, self.hidden_size, self.enc_embedding_size, self.rnn_num_layers,
                          self.enc_dropout_rate, self.pretrained_en_embeddings)
        decoder = Decoder(self.hidden_size, self.output_size, self.dec_embedding_size, self.rnn_num_layers,
                          self.dec_dropout_rate, self.pretrained_fr_embeddings)
        return encoder, decoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
        Sequence-to-sequence model for machine translation.

        Args:
            encoder (Encoder): The encoder module.
            decoder (Decoder): The decoder module.
            device (torch.device): The device to run the model on.

        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_size == decoder.hidden_size, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.rnn_num_layers == decoder.rnn_num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass of the Seq2Seq model.

        Args:
            src (torch.Tensor): Source sequence tensor.
            trg (torch.Tensor): Target sequence tensor.
            teacher_forcing_ratio (float): Probability of using teacher forcing (default: 0.5).

        Returns:
            outputs (torch.Tensor): Predicted output tensor.

        """
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        outputs[0, :, 1] = 1

        hidden, cell = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = trg[t] if teacher_force else top1

        return outputs
