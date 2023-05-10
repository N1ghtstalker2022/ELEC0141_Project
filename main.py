import os

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import StepLR

import B


# TODO: add this function to utils.pyh
def plot_losses(train_losses, val_losses, lr, weight_decay):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.xticks(range(1, len(train_losses) + 1))
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'./loss_{lr}_{weight_decay}.png')


# ======================================================================================================================
# Data preprocessing
print("Data preprocessing...")
train_dataloader, val_dataloader, test_dataloader, english_words, french_words, english_word_to_idx, french_word_to_idx = B.get_dataloader()
# ======================================================================================================================
# Task A
# setup machine translation model
print("Setup machine translation model...")
seq2seq_model = B.ModelBuilder(english_words, french_words)
encoder, decoder = seq2seq_model.build()

print("Training...")
criterion = torch.nn.NLLLoss()  # Negative Log-Likelihood Loss
encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=0.005, weight_decay=0.001)  # SGD optimizer for encoder
decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=0.005, weight_decay=0.001)  # SGD optimizer for decoder

encoder_scheduler = StepLR(encoder_optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler for encoder
decoder_scheduler = StepLR(decoder_optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler for decoder

trainer = B.Trainer(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, encoder_scheduler,
                    decoder_scheduler)
trained_encoder, trained_decoder, train_losses, val_losses = trainer.train(train_dataloader, val_dataloader,
                                                                           english_words, french_words,
                                                                           french_word_to_idx)  # Train model based on the training set (you should fine-tune your model based on validation set.)
torch.save(trained_encoder.module.state_dict(), 'encoder_50epoch.pth')
torch.save(trained_decoder.module.state_dict(), 'decoder_50epoch.pth')

plot_losses(train_losses, val_losses, 0.001, 0.01)

# ======================================================================================================================
# Test
print("Testing...")

# torch.save(trained_encoder.state_dict(), 'my_model_encoder.pt')
# torch.save(trained_decoder.state_dict(), 'my_model_decoder.pt')

# load the saved model
encoder_state_dict = torch.load('encoder_50epoch.pth', map_location=torch.device('cpu'))
decoder_state_dict = torch.load('decoder_50epoch.pth', map_location=torch.device('cpu'))


# create a new instance of your model
seq2seq_model = B.ModelBuilder(english_words, french_words)
encoder, decoder = seq2seq_model.build()
#
# # load the state dictionary into the model
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     encoder = nn.DataParallel(encoder)
#     decoder = nn.DataParallel(decoder)
#
#
encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)

device = torch.device("cpu")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs")
#     encoder = nn.DataParallel(encoder)
#     decoder = nn.DataParallel(decoder)

B.test_model(encoder, decoder, test_dataloader, french_words,
             french_word_to_idx, device)  # Test model based on the test set.
##
# ======================================================================================================================
# acc_A_test = model_A.test(args...)  # Test model based on the test set.
# Clean up memory / GPU
# etc...  # Some code to free memory if necessary.

# ======================================================================================================================
# Task B
# model_B = B(args...)
# acc_B_train = model_B.train(args...)
# acc_B_test = model_B.test(args...)
# Clean
# up
# memory / GPU
# etc...

# ======================================================================================================================
# Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                   acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'
