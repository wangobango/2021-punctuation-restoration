import string
from nltk import word_tokenize
import numpy as np
import spacy

nlp = spacy.load("pl_core_news_sm")

class Vocabulary():
    def __init__(self) -> None:
        self.__UKNOWN__ = -1
        self.vocab = {}

    def create_vocab(self, data):
        counter = 0 
        vocab = {}
        for row in data:
            for word in word_tokenize(row):
                ex = word.lower()
                if ex in vocab:
                    pass
                else:
                    vocab[ex] = counter
                    counter += 1

        self.vocab = vocab

    def word_to_number(self, word):
        if word.lower() in self.vocab:
            return self.vocab[word]
        else:
            return self.__UKNOWN__


class MarkovChain():
    def __init__(self, context_size: int, data: list, vocabulary: dict) -> None:
        self.context_size = context_size
        self.data = data
        self.vocabulary = vocabulary

    def __preprocess(self, word: str) -> str:
        # word = nlp(word)
        return word

    def __make_immutable(self, obj):
        return str(obj)
    
    def __assign_word_to_context(self, next_word: str, context: list, vector: dict) -> None:
        next_word = self.__preprocess(next_word)
        context = self.__make_immutable([self.__preprocess(x) for x in context])
        if context in vector:
            if next_word in vector[context]:
                vector[context][next_word] += 1
            else:
                vector[context][next_word] = 0
        else:
            vector[context] = {}

    def fit(self):
        vector = {}
        for row in self.data:
            tokenized = word_tokenize(row)
            for i in range(0, len(tokenized) - self.context_size - 1):
                context = tokenized[i: i + self.context_size]
                next_word = tokenized[i + self.context_size + 1]
                self.__assign_word_to_context(next_word, context, vector)
        self.vector = vector

    @property
    def get_vector(self) -> dict:
        return self.vector
                
    def predict(self, data: list, num_of_items_to_pick: int) -> str:
        data = self.__make_immutable([self.__preprocess(x) for x in data])
        if data in self.vector:
            values = np.asarray(list(self.vector[data].values()))
            if values.size == 0:
                return ''
            if np.count_nonzero(values) == 0:
                return ''
            keys = list(self.vector[data].keys())
            normalized_probs = values / np.sum(values)
            draw = np.random.choice(keys, size=num_of_items_to_pick, p=normalized_probs)
            return draw
        else:
            return ''

    
