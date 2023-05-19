import os

import numpy as np
import torch
import sacrebleu
from torch import nn

from B import get_dataloader, ModelBuilder, Seq2Seq


def clean_text(text):
    # text = text.replace('<sos>', '')
    # text = text.replace('<eos>', '')
    # text = text.replace('<pad>', '')
    return text.strip()


def test_model(model, test_dataloader, french_words, french_word_to_idx, device):
    model = model.to(device)

    model.eval()
    references = []
    hypotheses = []
    example_count = 0

    with torch.no_grad():
        for english_sentences, french_sentences in test_dataloader:
            example_count = 0
            src_sentences = english_sentences.to(device)
            trg_sentences = french_sentences.to(device)
            output = model(src_sentences, trg_sentences, 0)
            # output = nn.LogSoftmax(dim=2)(output)
            numpy_ = output.numpy()[3, 2, :]
            print(numpy_)
            max_value = np.max(numpy_)
            print(max_value)
            # output_dim = output.shape[-1]
            output_tokens = torch.argmax(output, dim=2)

            print(output_tokens.shape)

            # hypotheses.extend([' '.join([french_words[i] for i in tokens.flatten()]) for tokens in output_tokens])

            # turn tokens into words
            for i in range(output_tokens.shape[1]):
                flattened_tokens = output_tokens[:, i].flatten()
                french_sentence = []
                for i in flattened_tokens:
                    french_word = french_words[i]
                    french_sentence.append(french_word)
                french_sentence_string = ' '.join(french_sentence)

                cleaned_french_sentence = clean_text(french_sentence_string)
                hypotheses.append(cleaned_french_sentence)

            references.extend([' '.join([french_words[i] for i in sentence]) for sentence in
                               trg_sentences.transpose(0, 1).tolist()])
            # print predicted and true translations

            while example_count < 10:
                example_count = example_count + 1
                print(f"Predicted: {hypotheses[example_count]}")
                print(f"True: {references[example_count]}")
                print("=" * 50)

    # compute BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"BLEU score: {bleu.score:.2f}")


if __name__ == '__main__':

    train_dataloader, val_dataloader, test_dataloader, english_words, french_words, english_word_to_idx, french_word_to_idx = get_dataloader()

    device = torch.device("cpu")

    # Test model based on the test set.
    print("Testing...")
    # load the saved model
    seq2seq_state_dict = torch.load('../seq2seq_20epoch.pth', map_location=torch.device('cpu'))

    # create a new instance of your model
    enc_dec_builder = ModelBuilder(english_words, french_words, pretrained_embeddings=None)
    encoder, decoder = enc_dec_builder.build()
    seq2seq_model = Seq2Seq(encoder, decoder, device)

    seq2seq_model.load_state_dict(seq2seq_state_dict)


    test_model(seq2seq_model, test_dataloader, french_words,
               french_word_to_idx, device)
