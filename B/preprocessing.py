import re
from collections import Counter
from itertools import chain
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


# from B.seq2seq import Encoder, Decoder
# TODO: Add 进度条。 upload to github

def read_sentences(file_path):
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]


def tokenize(sentence):
    return [word.lower() for word in re.findall(r'\b\w+\b', sentence)]


def build_vocab(tokenized_sentences, min_freq=2):
    counter = Counter(chain.from_iterable(tokenized_sentences))
    words = ['<pad>', '<sos>', '<eos>', '<unk>'] + [word for word, count in counter.items() if count >= min_freq]
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    return words, word_to_idx


class TranslationDataset(Dataset):
    def __init__(self, english_sentences, french_sentences, english_vocab, french_vocab):
        self.english_sentences = english_sentences
        self.french_sentences = french_sentences
        self.english_vocab = english_vocab
        self.french_vocab = french_vocab

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        eng_sentence = [self.english_vocab['<sos>']] + \
                       [self.english_vocab.get(word, self.english_vocab['<unk>']) for
                        word in tokenize(self.english_sentences[idx])] + \
                       [self.english_vocab['<eos>']]
        fr_sentence = [self.french_vocab['<sos>']] + [self.french_vocab.get(word, self.french_vocab['<unk>']) for word
                                                      in tokenize(self.french_sentences[idx])] + [
                          self.french_vocab['<eos>']]
        return torch.tensor(eng_sentence), torch.tensor(fr_sentence)


def collate_fn(batch):
    english_sentences, french_sentences = zip(*batch)
    english_sentences_padded = pad_sequence(english_sentences, padding_value=0)
    french_sentences_padded = pad_sequence(french_sentences, padding_value=0)
    return english_sentences_padded, french_sentences_padded


def get_dataloader():
    # Read sentences from files
    english_sentences = read_sentences("/scratch/zczqyc4/ELEC0141-DataSet/europarl-v7.fr-en.en")
    french_sentences = read_sentences("/scratch/zczqyc4/ELEC0141-DataSet/europarl-v7.fr-en.fr")
    # Calculate the 10% index
    num_sentences = len(english_sentences)
    end_index = int(num_sentences * 0.01)
    print("Number of sentences: {}".format(num_sentences))

    # Keep only 10% of the sentences
    english_sentences = english_sentences[:end_index]
    french_sentences = french_sentences[:end_index]

    # Tokenize and build vocabularies
    tokenized_english = [tokenize(sentence) for sentence in english_sentences]
    tokenized_french = [tokenize(sentence) for sentence in french_sentences]
    english_words, english_word_to_idx = build_vocab(tokenized_english)
    french_words, french_word_to_idx = build_vocab(tokenized_french)

    # Create dataset and dataloader
    dataset = TranslationDataset(english_sentences, french_sentences, english_word_to_idx, french_word_to_idx)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders for training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # TODO: return words ... to feed into training process
    return train_dataloader, val_dataloader, english_words, french_words, english_word_to_idx, french_word_to_idx


if __name__ == '__main__':
    dataloader = get_dataloader()
