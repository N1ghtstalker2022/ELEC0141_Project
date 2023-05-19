"""
This script trains and tests a machine translation model using a sequence-to-sequence architecture.

1. Data preprocessing:
   - Loads and preprocesses the training, validation, and test data.
   - Builds vocabularies for English and French words.

2. Train and validate Machine Translation Model
   - Sets up the machine translation model using an encoder and decoder.
   - Loads pretrained word embeddings (optional).
   - Trains the model using the training set and validates using the validation set.
   - Saves the trained model.

3. Testing:
   - Loads the saved model.
   - Tests the model using the test set.
"""
import os
import random

import numpy as np
import torch
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import StepLR
import B
import gensim.downloader as api


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def get_embedding_matrix(keyed_vectors, word_to_idx):
    embedding_dim = keyed_vectors.vector_size
    embeddings = np.zeros((len(word_to_idx), embedding_dim))
    for word, idx in word_to_idx.items():
        try:
            embeddings[idx] = keyed_vectors[word]
        except KeyError:  # word not in pretrained vocab
            pass  # keep as zero vector
    return torch.tensor(embeddings).float()


def load_pretrained_embeddings(english_word_to_idx):
    english_model = KeyedVectors.load_word2vec_format('/scratch/zczqyc4/english_embeddings.bin', binary=True)
    english_embeddings = get_embedding_matrix(english_model, english_word_to_idx)
    french_embeddings = None
    pretrained_embeddings = {
        "english_embeddings": english_embeddings,
        "french_embeddings": french_embeddings,
    }

    return pretrained_embeddings


# TODO: add this function to utils.pyh, make the display more beautiful: prevent overlap of labels of x-axis
def plot_losses(train_losses, val_losses, lr, weight_decay):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.xticks(range(1, len(train_losses) + 1))
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'./loss_{lr}_{weight_decay}.png')


# set the random seeds for deterministic results.
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# set device to GPU if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================================================================
# Data preprocessing
print("Data preprocessing...")
train_dataloader, val_dataloader, test_dataloader, english_words, french_words, english_word_to_idx, french_word_to_idx = B.get_dataloader()
# ======================================================================================================================
# Task A
# setup machine translation model
print("Setup machine translation model...")

# Load pretrained word embeddings
pretrained_embeddings = None
# pretrained_embeddings = load_pretrained_embeddings(english_word_to_idx)
enc_dec_builder = B.ModelBuilder(english_words, french_words, pretrained_embeddings=pretrained_embeddings)
encoder, decoder = enc_dec_builder.build()
seq2seq_model = B.Seq2Seq(encoder, decoder, device)

seq2seq_model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(seq2seq_model):,} trainable parameters')

print("Training...")
criterion = nn.CrossEntropyLoss()  # Negative Log-Likelihood Loss
weight_decay_rate = 0

optimizer = torch.optim.SGD(seq2seq_model.parameters(), lr=0.01,
                                    weight_decay=weight_decay_rate)  # SGD optimizer

# encoder_scheduler = StepLR(encoder_optimizer, step_size=10, gamma=0.5)  # Learning rate scheduler for encoder
# decoder_scheduler = StepLR(decoder_optimizer, step_size=10, gamma=0.5)  # Learning rate scheduler for decoder
encoder_scheduler = None
decoder_scheduler = None

trainer = B.Trainer(seq2seq_model, criterion, optimizer, encoder_scheduler,
                    decoder_scheduler)
trained_seq2seq_model, train_losses, val_losses = trainer.train(train_dataloader, val_dataloader,
                                                                           english_words, french_words,
                                                                           french_word_to_idx)  # Train model based on the training set (you should fine-tune your model based on validation set.)
torch.save(trained_seq2seq_model.state_dict(), 'seq2seq_20epoch.pth')
# torch.save(trained_decoder.module.state_dict(), 'decoder_50epoch.pth')

plot_losses(train_losses, val_losses, 0.01, 0)

# ======================================================================================================================
# Test
print("Testing...")
device = torch.device("cpu")

# load the saved model
seq2seq_state_dict = torch.load('seq2seq_20epoch.pth', map_location=torch.device('cpu'))
# decoder_state_dict = torch.load('decoder_50epoch.pth', map_location=torch.device('cpu'))

# create a new instance of your model
enc_dec_builder = B.ModelBuilder(english_words, french_words, pretrained_embeddings=pretrained_embeddings)
encoder, decoder = enc_dec_builder.build()
seq2seq_model = B.Seq2Seq(encoder, decoder, device)


seq2seq_model.load_state_dict(seq2seq_state_dict)


B.test_model(seq2seq_model, test_dataloader, french_words,
             french_word_to_idx, device)  # Test model based on the test set.
