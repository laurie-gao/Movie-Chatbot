import numpy as np 
import pandas as pd
import re 

#Create a line ID to movie line dictionary
id2line = {}

# Parse movie_lines.txt
for line in open('../data/movie_lines.txt', errors = 'ignore'):
	line_ = line.split(" +++$+++ ")
	id2line[line_[0]] = line_[4]

#Create list of conversations
conversations = []

#Parse movie_conversations.txt
for convo in open('../data/movie_conversations.txt', errors = 'ignore'):
    convo_ = convo.split(' +++$+++ ')[-1].strip()[1:-1].replace("'", "").replace(" ", "")
    conversations.append(convo_.split(","))

#Create input and ouput pairs
inputs = []
outputs = []

for convo in conversations:
    inputs.append(id2line[convo[0]])
    outputs.append(id2line[convo[1]])

#Clean input and output line
def clean_line(line):
    line = line.lower() 
    line = re.sub(r"i'm", "i am", line)
    line = re.sub(r"he's", "he is", line)
    line = re.sub(r"she's", "she is", line)
    line = re.sub(r"that's", "that is", line)
    line = re.sub(r"what's", "what is", line)
    line = re.sub(r"where's", "where is", line)
    line = re.sub(r"\'ll", " will", line)
    line = re.sub(r"\'ve", " have", line)
    line = re.sub(r"\'re", " are", line)
    line = re.sub(r"\'d", " would", line)
    line = re.sub(r"won't", "will not", line)
    line = re.sub(r"can't", "cannot", line)
    line = re.sub(r"[-()\"#/@;:<>{}+=~|]", "", line)
    return line

clean_inputs = []
for line in inputs:
    clean_inputs.append(clean_line(line))

clean_outputs = []
for line in outputs:
    clean_outputs.append(clean_line(line))

#create pandas df and save as csv
df = pd.DataFrame({'inputs': clean_inputs, 'outputs': clean_outputs})
df.to_csv('train_data.csv', index=False)

