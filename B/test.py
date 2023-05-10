import os

import torch
import sacrebleu
from torch import nn

from B import get_dataloader, ModelBuilder

def clean_text(text):
    text = text.replace('<sos>', '')
    text = text.replace('<eos>', '')
    text = text.replace('<pad>', '')
    return text.strip()

def test_model(encoder, decoder, test_dataloader, french_words, french_word_to_idx, device):
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()
    references = []
    hypotheses = []
    example_count = 0

    with torch.no_grad():
        for english_sentences, french_sentences in test_dataloader:
            english_sentences = english_sentences.to(device)
            french_sentences = french_sentences.to(device)


            encoder_hidden = encoder.init_hidden(english_sentences.size(1)).to(device)
            _, encoder_hidden = encoder(english_sentences, encoder_hidden)

            decoder_input = torch.tensor(
                [[french_word_to_idx['<sos>']]] * french_sentences.size(1)).transpose(0, 1).to(device)
            decoder_hidden = encoder_hidden
            output_tokens = []
            # print(decoder_input.shape)
            # print(decoder_hidden.shape)

            for di in range(french_sentences.size(0)):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                output_tokens.append(topi)
                decoder_input = topi.squeeze().detach().unsqueeze(0)


            # convert output tokens to sentences
            # print(output_tokens)
            output_tokens = torch.stack(output_tokens, dim=1)

            # hypotheses.extend([' '.join([french_words[i] for i in tokens.flatten()]) for tokens in output_tokens])

            # turn tokens into words
            for tokens in output_tokens:
                flattened_tokens = tokens.flatten()
                french_sentence = []
                for i in flattened_tokens:
                    french_word = french_words[i]
                    french_sentence.append(french_word)
                french_sentence_string = ' '.join(french_sentence)

                cleaned_french_sentence = clean_text(french_sentence_string)
                hypotheses.append(cleaned_french_sentence)


            references.extend([' '.join([french_words[i] for i in sentence if
                                         i not in [french_word_to_idx['<sos>'], french_word_to_idx['<eos>'],
                                                   french_word_to_idx['<pad>']]]) for sentence in
                               french_sentences.transpose(0, 1).tolist()])
            # print predicted and true translations

            if example_count < 5:
                example_count = example_count + 1
                print(f"Predicted: {hypotheses[-1]}")
                print(f"True: {references[-1]}")
                print("=" * 50)

    # compute BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"BLEU score: {bleu.score:.2f}")


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader, english_words, french_words, english_word_to_idx, french_word_to_idx = get_dataloader()
    encoder_state_dict = torch.load('../encoder.pth', map_location=torch.device('cpu'))
    decoder_state_dict = torch.load('../decoder.pth', map_location=torch.device('cpu'))

    # create a new instance of your model
    seq2seq_model = ModelBuilder(english_words, french_words)
    encoder, decoder = seq2seq_model.build()

    encoder.load_state_dict(encoder_state_dict)
    decoder.load_state_dict(decoder_state_dict)

    device = torch.device("cpu")

    test_model(encoder, decoder, test_dataloader, french_words,
               french_word_to_idx, device)  # Test model based on the test set.
