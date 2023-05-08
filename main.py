import torch
from matplotlib import pyplot as plt

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
train_dataloader, val_dataloader, english_words, french_words, english_word_to_idx, french_word_to_idx = B.get_dataloader()
# ======================================================================================================================
# Task A
# setup machine translation model
print("Setup machine translation model...")
seq2seq_model = B.seq2seq.ModelBuilder(english_words, french_words)
encoder, decoder = seq2seq_model.build()

print("Training...")
criterion = torch.nn.NLLLoss()  # Negative Log-Likelihood Loss
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001, weight_decay=0.01) # Adam optimizer for encoder
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001, weight_decay=0.01) # Adam optimizer for decoder
trainer = B.train.Trainer(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer)
train_losses, val_losses = trainer.train(train_dataloader, val_dataloader, english_words, french_words, french_word_to_idx,)  # Train model based on the training set (you should fine-tune your model based on validation set.)



plot_losses(train_losses, val_losses, 0.001, 0.01)

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

