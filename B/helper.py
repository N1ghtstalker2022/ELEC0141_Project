import gensim.downloader as api
from gensim.models import KeyedVectors

# Download and load the word2vec embeddings
english_model = api.load('word2vec-google-news-300')

# Save the English word embeddings
english_output_path = "/scratch/zczqyc4/english_embeddings.bin"
english_model.save_word2vec_format(english_output_path, binary=True)
#
# english_model = KeyedVectors.load_word2vec_format('/scratch/zczqyc4/english_embeddings.bin', binary=True)
#
# # Access the word embeddings
# # english_embedding = english_model["word"]
# print(english_model)
