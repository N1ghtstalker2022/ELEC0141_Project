The README file should contain
● a brief description of the organization of your project;
● the role of each file;
● the packages required to run your code (e.g. numpy, scipy, transformers,
etc.).


# ELEC0141: Deep Learning with Natural Language Processing Assignment 22_23

## Organization of the project
- folder A contains the code for seq2seq model with attention mechanism
- folder B contains the code for plain seq2seq model, along with data preprocessing, model training and testing function.

Take folder B as an example, the role of each file is as follows:
- build_dataset.py: load original dataset, sample from it and store it to the disk.
- explore.py: explore the dataset, including the length of the sentences, the number of words, etc.
- preprocess.py: preprocess the dataset, including tokenization, padding, PyTorch dataset build.
- seq2seq.py: define the seq2seq model.
- train.py: define the training function and generate learning curve.
- test.py: test the model with BLEU score.

## packages required to run your code
- torch=2.0.0
- spacy=3.5.3
- numpy=1.24.3
- matplotlib=3.7.1
- sacrebleu=2.3.1


The entrance of the project is main.py, which can be run directly in the terminal.
```commandline
python main.py
```

