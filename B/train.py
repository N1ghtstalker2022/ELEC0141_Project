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
    def __init__(self, model, criterion, optimizer, encoder_scheduler=None, decoder_scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.encoder_scheduler = encoder_scheduler
        self.decoder_scheduler = decoder_scheduler
        self.train_dataloader = None
        self.val_dataloader = None
        self.english_words = None
        self.french_words = None
        self.french_word_to_idx = None

    def train(self, train_dataloader, val_dataloader, english_words, french_words, french_word_to_idx=None,
              num_epochs=10):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.english_words = english_words
        self.french_words = french_words
        self.french_word_to_idx = french_word_to_idx
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_gpu_info()

        # enable distributed learning if multiple GPUs are available
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs")
        #     self.model = nn.DataParallel(self.model)
        #
        print(f"Using one GPU")

        self.model = self.model.to(device)

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
        return self.model, train_losses, val_losses

    def run_epoch(self, device, training=True):
        dataloader = self.train_dataloader if training else self.val_dataloader
        self.model.train(training)
        total_loss = 0

        for english_sentences, french_sentences in dataloader:
            src_sentences = english_sentences.to(device)
            print(src_sentences.shape)
            trg_sentences = french_sentences.to(device)

            if training:
                self.optimizer.zero_grad()

            output = self.model(src_sentences, trg_sentences)
            # output = nn.LogSoftmax(dim=2)(output)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg_sentences[1:].view(-1)
            loss = self.criterion(output, trg)

            if training:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() / len(dataloader)
        return total_loss




if __name__ == '__main__':
    pass
