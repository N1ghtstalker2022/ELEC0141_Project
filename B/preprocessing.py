"""
This script provides utility functions and classes for data preprocessing and creating data loaders for machine translation tasks.
It includes functions for reading sentences from a file, tokenizing sentences, building vocabulary, and creating PyTorch datasets and dataloaders.
The script also contains a main section that demonstrates the usage of these functions.

List of functions and classes:
- read_sentences(file_path): Reads sentences from a file and returns a list of sentences.
- tokenize(sentence, tokenizer): Tokenizes a sentence using a tokenizer and returns a list of tokens.
- build_vocab(tokenized_sentences, min_freq=2): Builds a vocabulary from a list of tokenized sentences and returns the vocabulary words and word-to-index mapping.
- TranslationDataset: A PyTorch dataset class for translation tasks, with support for English and French sentences.
- collate_fn(batch): A function to collate a batch of data samples, padding sequences to the same length.
- split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=True): Splits a dataset into training, validation, and test sets.

The main section of the code demonstrates the usage of these functions by reading English and French sentences from files,
tokenizing them, building vocabularies, creating datasets, splitting them into train, validation, and test sets,
and creating dataloaders for training, validation, and test sets.

Note: The code assumes that the input files and tokenizers for English and French sentences are available.
"""


import random
import re
from collections import Counter
from itertools import chain
import torch
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
import spacy


def read_sentences(file_path):
    """
    Read sentences from a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        List of sentences.
    """
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]


def tokenize(sentence, tokenizer):
    """
    Tokenize a sentence using a tokenizer.

    Args:
        sentence (str): Input sentence.
        tokenizer: Tokenizer object.

    Returns:
        List of tokens.
    """
    return [tok.text.lower() for tok in tokenizer.tokenizer(sentence)]


def build_vocab(tokenized_sentences, min_freq=2):
    """
    Build vocabulary from a list of tokenized sentences.

    Args:
        tokenized_sentences (list): List of tokenized sentences.
        min_freq (int): Minimum frequency for a word to be included in the vocabulary (default: 2).

    Returns:
        words (list): List of vocabulary words.
        word_to_idx (dict): Mapping of word to index.
    """
    counter = Counter(chain.from_iterable(tokenized_sentences))
    words = ['<pad>', '<sos>', '<eos>', '<unk>'] + [word for word, count in counter.items() if count >= min_freq]
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    return words, word_to_idx


class TranslationDataset(Dataset):
    """PyTorch dataset for translation tasks.

    Args:
        english_sentences (List[str]): List of English sentences.
        french_sentences (List[str]): List of French sentences.
        english_vocab (Dict[str, int]): English vocabulary mapping word to index.
        french_vocab (Dict[str, int]): French vocabulary mapping word to index.
        en_tokenizer (spacy.lang): English tokenizer.
        fr_tokenizer (spacy.lang): French tokenizer.

    Attributes:
        english_sentences (List[str]): List of English sentences.
        french_sentences (List[str]): List of French sentences.
        english_vocab (Dict[str, int]): English vocabulary mapping word to index.
        french_vocab (Dict[str, int]): French vocabulary mapping word to index.
        en_tokenizer (spacy.lang): English tokenizer.
        fr_tokenizer (spacy.lang): French tokenizer.

    """
    def __init__(self, english_sentences, french_sentences, english_vocab, french_vocab, en_tokenizer, fr_tokenizer):
        self.english_sentences = english_sentences
        self.french_sentences = french_sentences
        self.english_vocab = english_vocab
        self.french_vocab = french_vocab
        self.en_tokenizer = en_tokenizer
        self.fr_tokenizer = fr_tokenizer

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The number of English sentences in the dataset.

        """
        return len(self.english_sentences)

    def __getitem__(self, idx):
        """Returns a specific item from the dataset at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the English sentence and the corresponding French sentence.

        """
        eng_sentence = [self.english_vocab['<sos>']] + \
                       [self.english_vocab.get(word, self.english_vocab['<unk>']) for
                        word in tokenize(self.english_sentences[idx], self.en_tokenizer)] + \
                       [self.english_vocab['<eos>']]
        fr_sentence = [self.french_vocab['<sos>']] + [self.french_vocab.get(word, self.french_vocab['<unk>']) for word
                                                      in tokenize(self.french_sentences[idx], self.fr_tokenizer)] + [
                          self.french_vocab['<eos>']]

        return torch.tensor(eng_sentence), torch.tensor(fr_sentence)


def collate_fn(batch):
    english_sentences, french_sentences = zip(*batch)
    english_sentences_padded = pad_sequence(english_sentences, padding_value=0)
    french_sentences_padded = pad_sequence(french_sentences, padding_value=0)
    return english_sentences_padded, french_sentences_padded


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=True):
    """
    Splits a dataset into training, validation, and test sets.

    Args:
        dataset: Dataset object.
        train_ratio (float): Ratio of training set (default: 0.8).
        val_ratio (float): Ratio of validation set (default: 0.1).
        test_ratio (float): Ratio of test set (default: 0.1).
        shuffle (bool): Whether to shuffle the dataset before splitting (default: True).

    Returns:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle:
        random.shuffle(indices)

    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def get_dataloader():
    """
    Get data loaders for training, validation, and test sets.

    Returns:
        train_dataloader: Data loader for the training set.
        val_dataloader: Data loader for the validation set.
        test_dataloader: Data loader for the test set.
        english_words: List of English vocabulary words.
        french_words: List of French vocabulary words.
        english_word_to_idx: Mapping of English word to index.
        french_word_to_idx: Mapping of French word to index.
    """
    # Read sentences from files
    english_sentences = read_sentences("/scratch/zczqyc4/ELEC0141-DataSet/english_subset.txt")
    french_sentences = read_sentences("/scratch/zczqyc4/ELEC0141-DataSet/french_subset.txt")
    print(english_sentences[503])
    num_sentences = len(english_sentences)
    print("Number of sentences: {}".format(num_sentences))

    # Tokenize and build vocabularies
    en_tokenizer = spacy.load("en_core_web_sm")
    fr_tokenizer = spacy.load("fr_core_news_sm")

    tokenized_english = [tokenize(sentence, en_tokenizer) for sentence in english_sentences]
    tokenized_french = [tokenize(sentence, fr_tokenizer) for sentence in french_sentences]

    longest_length = len(max(tokenized_english, key=len))
    print("Longest sentence length: {}".format(longest_length))
    english_words, english_word_to_idx = build_vocab(tokenized_english)
    french_words, french_word_to_idx = build_vocab(tokenized_french)

    # Create dataset and dataloader
    dataset = TranslationDataset(english_sentences, french_sentences, english_word_to_idx, french_word_to_idx,
                                 en_tokenizer, fr_tokenizer)
    # Split the dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1,
                                                             test_ratio=0.1, shuffle=True)
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    print(f"Number of English Vocabularies: {len(english_words)}")
    print(f"Number of French Vocabularies: {len(french_words)}")

    # Create dataloaders for training and validation sets, applying batch-wise padding using collate_fn
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader, english_words, french_words, \
        english_word_to_idx, french_word_to_idx


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader, english_words, french_words, \
        english_word_to_idx, french_word_to_idx = get_dataloader()

    for english_sentences, french_sentences in test_dataloader:
        print(english_sentences.shape)
        print(french_sentences.shape)
        break
