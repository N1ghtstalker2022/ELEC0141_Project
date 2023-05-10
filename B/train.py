import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


def print_gpu_info():
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current GPU index:", torch.cuda.current_device())
        print("Current GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("No GPU available.")


class Trainer:
    def __init__(self, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, encoder_scheduler=None, decoder_scheduler=None):
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.encoder_scheduler = encoder_scheduler
        self.decoder_scheduler = decoder_scheduler
        self.train_dataloader = None
        self.val_dataloader = None
        self.english_words = None
        self.french_words = None
        self.french_word_to_idx = None

    def train(self, train_dataloader, val_dataloader, english_words, french_words, french_word_to_idx=None,
              num_epochs=50):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.english_words = english_words
        self.french_words = french_words
        self.french_word_to_idx = french_word_to_idx
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_gpu_info()

        # enable distributed learning if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        train_losses, val_losses = [], []

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            train_loss = self.run_epoch(device, True)
            val_loss = self.run_epoch(device, False)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, time: {time.time() - epoch_start_time:.2f} seconds")

            if self.encoder_scheduler:
                self.encoder_scheduler.step()
            if self.decoder_scheduler:
                self.decoder_scheduler.step()

        print(f"Training time: {time.time() - start_time:.2f} seconds")
        return self.encoder, self.decoder,train_losses, val_losses

    def run_epoch(self, device, training=True):
        dataloader = self.train_dataloader if training else self.val_dataloader
        self.encoder.train(training)
        self.decoder.train(training)
        total_loss = 0

        for english_sentences, french_sentences in dataloader:
            english_sentences = english_sentences.to(device)
            french_sentences = french_sentences.to(device)

            if training:
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

            encoder_hidden = self.encoder.module.init_hidden(english_sentences.size(1)).to(device)
            _, encoder_hidden = self.encoder(english_sentences, encoder_hidden)

            decoder_input = torch.tensor(
                [[self.french_word_to_idx['<sos>']]] * french_sentences.size(1)).transpose(0, 1).to(device)
            decoder_hidden = encoder_hidden
            # print(f"decoder input shape {decoder_input.shape}")
            # print(f" decoder hidden shape {decoder_hidden.shape}")

            loss = 0
            for di in range(french_sentences.size(0)):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                loss += self.criterion(decoder_output, french_sentences[di])
                decoder_input = french_sentences[di].unsqueeze(0)

            if training:
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

            total_loss += loss.item() / len(dataloader)
        return total_loss




if __name__ == '__main__':
    pass
