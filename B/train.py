"""
This script provides utility functions and a Trainer class for training a Seq2Seq model for machine translation tasks.
It includes functions for printing GPU information, as well as a Trainer class that handles the training process.

List of functions and classes:
- print_gpu_info(): Prints information about the available GPUs.
- Trainer: A class for training a Seq2Seq model, including methods for training and running epochs.

The main section of the code is empty and can be used for customization or calling the utility functions and Trainer class.

Note: The code assumes the availability of the necessary libraries and modules such as torch, matplotlib, and spacy.
"""

import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR


def print_gpu_info():
    """
    Print information about the available GPUs.
    """
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current GPU index:", torch.cuda.current_device())
        print("Current GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("No GPU available.")


class Trainer:
    def __init__(self, model, criterion, optimizer, encoder_scheduler=None, decoder_scheduler=None):
        """
        Trainer class for training a Seq2Seq model.

        Args:
            model (nn.Module): The Seq2Seq model.
            criterion: Loss criterion.
            optimizer: Optimizer for model parameters.
            encoder_scheduler: Learning rate scheduler for the encoder (optional).
            decoder_scheduler: Learning rate scheduler for the decoder (optional).
        """
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
        """
        Train the Seq2Seq model.

        Args:
            train_dataloader: Dataloader for the training set.
            val_dataloader: Dataloader for the validation set.
            english_words (list): List of English vocabulary words.
            french_words (list): List of French vocabulary words.
            french_word_to_idx (dict): French word-to-index dictionary (optional).
            num_epochs (int): Number of training epochs (default: 10).

        Returns:
            model: Trained Seq2Seq model.
            train_losses (list): List of training losses for each epoch.
            val_losses (list): List of validation losses for each epoch.
        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.english_words = english_words
        self.french_words = french_words
        self.french_word_to_idx = french_word_to_idx
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_gpu_info()

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

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, time: {time.time() - epoch_start_time:.2f} seconds")

            if self.encoder_scheduler:
                self.encoder_scheduler.step()
            if self.decoder_scheduler:
                self.decoder_scheduler.step()

        print(f"Training time: {time.time() - start_time:.2f} seconds")
        return self.model, train_losses, val_losses

    def run_epoch(self, device, training=True):
        """
        Run a single epoch of training or evaluation.

        Args:
            device: Device to run the model on.
            training (bool): Whether to run the model in training mode (default: True).

        Returns:
            total_loss: Total loss for the epoch.
        """
        dataloader = self.train_dataloader if training else self.val_dataloader
        self.model.train(training)
        total_loss = 0

        for english_sentences, french_sentences in dataloader:
            src_sentences = english_sentences.to(device)
            trg_sentences = french_sentences.to(device)

            if training:
                self.optimizer.zero_grad()

            output = self.model(src_sentences, trg_sentences)
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
