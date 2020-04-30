import numpy as np 
import pandas as pd
import re 

file = pd.read_csv('../results/model_outputs.csv')
inputs = file['inputs'].tolist()
model_outputs = file['outputs'].tolist()
outputs = []
for line in model_outputs:
	line = line.replace(" '", "'").replace(" n't", "n't").replace(" ...", "...").replace(" .", ".").replace(" !", "!").replace(" ,", ",").replace(" ?", "?")
	outputs.append(line)

test_res = pd.DataFrame({'inputs': inputs, 'outputs': outputs})
test_res.to_csv('test_results.csv')