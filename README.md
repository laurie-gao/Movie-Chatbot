## Introduction

This project is an experiment in sequence to sequence natural language processing with an aim of building a chatbot model capable of conversing in a human-like manner. To make a state-of-the-art chatbot used by services today, a substantially larger dataset and superior hardware would be required. Therefore, the focus of this project is not to build a robust chatbot, but to test whether the model can pick up patterns in speech and give reasonable responses with limited data and training time. 

A full list of the results outputed by the model is in [test_results.csv](https://github.com/laurie-gao/Movie-Chatbot/blob/master/results/test_results.csv)

## Data preprocessing

The training data is obtained from the [cornell movie dialogs corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

1. Run parse_data.py to convert the raw data into train_data.csv
2. Turn sentences into sequences of word tokens
3. Map unique word tokens to unique indices
4. Add padding to ensure all sequences are of same length

## Model architecture

The code for the entire model can be found in Model.py. The main building blocks of the model are listed below:

#### Embedding Layer
Use glove 100-dimensional pre-trained word embeddings to turn word index into corresponding word vector.
The same embedding layer is used for input and target sequences, and no further training is done on the embeddings.

#### Encoder
Pre-attention bi-directional lstm that returns an output at every timestep to be used by attention.

#### Attention
At every timestamp, computes the weights (amount of attention that should be paid to every encoder output) and uses the weights to compute a context vector that is fed into the decoder.

#### Decoder
Post-attention lstm that returns probability of each word at every timestep based on the context vector from attention and the word outputed by the previous decoder unit.

## Results

In total, the model was trained for 50 epochs (~9 hours)

![accuracy](https://github.com/laurie-gao/Movie-Chatbot/blob/master/graphs/acc.png)
![loss](https://github.com/laurie-gao/Movie-Chatbot/blob/master/graphs/loss.png)

Not enough training examples resulted in the model overfitting to the test set. The low validation accuracy can be attributed to the fact that there is no "correct" response to a sentence. The focus is to output responses that make sense gramatically and logically.
