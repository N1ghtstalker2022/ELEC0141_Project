"""Sample from the original dataset to create a smaller dataset.

Since there is limited computational power, we will only use 10% of the original dataset.

"""
def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    return sentences

def write_sentences(file_path, sentences):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(sentences)

# Read sentences from files
english_sentences = read_sentences("/scratch/zczqyc4/ELEC0141-DataSet/europarl-v7.fr-en.en")
french_sentences = read_sentences("/scratch/zczqyc4/ELEC0141-DataSet/europarl-v7.fr-en.fr")

# Calculate the 10% index
num_sentences = len(english_sentences)
end_index = int(num_sentences * 0.1)
print("Number of sentences: {}".format(num_sentences))

# Keep only 10% of the sentences
english_sentences = english_sentences[:end_index]
french_sentences = french_sentences[:end_index]

MIN_LENGTH = 3
MAX_LENGTH = 20
extracted_english_sentences = list()
extracted_french_sentences = list()
# Filter sentences with length shorter than 20
for i in range(len(english_sentences)):
    if MIN_LENGTH <= len(english_sentences[i].split()) <= MAX_LENGTH and  MIN_LENGTH <= len(french_sentences[i].split()) <= MAX_LENGTH:
        extracted_english_sentences.append(english_sentences[i])
        extracted_french_sentences.append(french_sentences[i])

# Write the sentences to new files
write_sentences("/scratch/zczqyc4/ELEC0141-DataSet/english_subset.txt", extracted_english_sentences)
write_sentences("/scratch/zczqyc4/ELEC0141-DataSet/french_subset.txt", extracted_french_sentences)
