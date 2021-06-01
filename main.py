from os import sep
from utils import Vocabulary
import pandas as pd
import string
from nltk import word_tokenize
import numpy as np
from utils import Vocabulary
from utils import MarkovChain
import jiwer
import pandas as pd

def calculate_wer(data, output):
    total_wer = 0
    counter = 0
    for a, b in zip(data, output):
        total_wer += jiwer.wer(a, b)
        counter += 1
    return total_wer/counter

def map_fun(el):
    if type(el) == list:
        return el[0]
    else:
        return el

def list_to_string(list):
    return ' '.join(map(map_fun, list))
     
def transform_input(data, model: MarkovChain, context_size: int) -> list:
    transformed = []
    for row in data:
        tokenized = word_tokenize(row)
        new_row = []
        for i in range(0, len(tokenized) - context_size - 1):
            context = tokenized[i: i + context_size]
            draw = model.predict(context, 5)
            new_row.append(context)
            for word in draw:
                if word in string.punctuation:
                    new_row.append(word)
        transformed.append(list_to_string(new_row))
    return transformed

pathData = "train/in.tsv"
pathExpected = "train/expected.tsv"

headers = ['FileI','ASROutput']
data = pd.read_csv(pathData, sep="\t")
data.columns = headers

expected = pd.read_csv(pathExpected, sep="\t")
expected.columns = ['FixedOutput']

vocabulary = Vocabulary()
vocabulary.create_vocab(expected['FixedOutput'])
context_size = 1

model = MarkovChain(context_size, expected['FixedOutput'], vocabulary)
model.fit()

transformed = transform_input(data['ASROutput'], model, context_size)

print("Initial wer")
print(calculate_wer(expected['FixedOutput'], data['ASROutput']))

print("Final wer")
print(calculate_wer(expected['FixedOutput'], transformed))

df = pd.DataFrame(transformed)
df.to_csv("train/out.tsv", sep='\t')

# X = np.asarray(list(itertools.chain.from_iterable([word_tokenize(x) for x in expected['FixedOutput']])))
# model = hmm.MultinomialHMM()
# model.fit(X.reshape(-1, 1))