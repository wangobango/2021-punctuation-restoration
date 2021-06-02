from os import sep
from typing import Tuple
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

def solve(s):
    seen = s[0]
    ans = s[0]
    for i in s[1:]:
        if i != seen:
            ans += i
            seen = i
    return ans
     
def transform_input(data, model: MarkovChain, context_size: int, threshold: int) -> list:
    transformed = []
    for row in data:
        tokenized = word_tokenize(row)
        new_row = []
        for i in range(0, len(tokenized) - context_size - 1):
            context = tokenized[i: i + context_size]
            draw = model.predict(context, 5, threshold)
            new_row.append(context)
            for word in draw:
                if word in string.punctuation:
                    new_row.append(word)
                    break
        transformed.append(solve(list_to_string(new_row)))
    return transformed

def transform_and_calculate_accuracy(data, expected, model: MarkovChain, context_size: int, threshold: int) -> int:
    # location -> (row_idx, char_idx, punctuation_mark)
    expected_mark_location = [] 

    for row_idx, row in enumerate(expected):
        for word_idx, word in enumerate(word_tokenize(row)):
            if word in string.punctuation:
                expected_mark_location.append((row_idx, word_idx, word))
         
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for location in expected_mark_location:
        row_idx, char_idx, punctuation_mark = location
        row = word_tokenize(data[row_idx])
        context = row[char_idx - context_size: char_idx]
        draw = model.predict(context, 1, threshold)
        if len(draw) > 0:
            if draw[0] == punctuation_mark:
                tp += 1
            # elif draw != punctuation_mark and draw in string.punctuation:
            #     fp += 1
            # elif draw != punctuation_mark and draw in string.punctuation:
            #     fn += 1
            elif draw[0] != punctuation_mark and draw[0] in string.punctuation:
                tn += 1

    return tp / (tp + tn)

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
threshold = 0.9

model = MarkovChain(context_size, expected['FixedOutput'], vocabulary)
model.fit()

transformed = transform_input(data['ASROutput'], model, context_size, threshold)

print("Initial wer")
print(calculate_wer(expected['FixedOutput'], data['ASROutput']))

print("Final wer")
print(calculate_wer(expected['FixedOutput'], transformed))

print("Accuracy:")
print(transform_and_calculate_accuracy(data['ASROutput'], expected['FixedOutput'], model, context_size, threshold))

df = pd.DataFrame(transformed)
df.to_csv("train/out.tsv", sep='\t')
