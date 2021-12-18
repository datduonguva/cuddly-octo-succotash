---
layout: post
title:  Word Complexity Estimation
tags: "Machine Learning"
---

## Introduction
Predicting word complexity is a natural language processing task that given an input string and a specified substring, the model should make a prediction for the complexity of that substring. The data set were annotated by having human labellers grading the level of complexity fo the word phrase. The complexity of the a phrase is defined as the number of annotators who mark the phrase as complex over total number of annotators

## Load required packages
```python
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
```

## Read the training data
Let's import the data and take a quick look at them

```python
df = pd.read_csv("train_full.txt", header=None, delimiter = "\t")
df.columns = [
    "id", "sentence", "start_index", "end_index", "word", "num_native", "num_non_native",
    "num_native_yes", "num_non_native_yes", "complexity"
]

df.head()
```

## Preprocessing
* changes the letter to upper case, only keep ASCII characters. Replace all non-ASCII characters with "?
* create the new columns with the masked tokens is replaced by `<MASK>`

```python
df.sentence = df.sentence.str.upper().apply(lambda x: x.encode("ascii", errors='replace').decode())
df["masked_sentence"] = df.apply(
    lambda row: (
        row.sentence[:row.start_index] + 
        " ".join(["<MASK>"]*len(row.sentence[row.start_index: row.end_index].split(" "))) + 
        row.sentence[row.end_index:]
    ),
    axis=1
)

```
* Get all the tokens and build the tokenizer

```python
# tokens from both df["sentence"] and df["masked_sentence"]
tokens = set([])
for sentence in df.sentence.to_list():
    tokens.update(set(sentence.split(" ")))

for sentence in df.masked_sentence.to_list():
    tokens.update(set(sentence.split(" ")))


# for padding use '<PAD>' tokens
tokens.add("<PAD>")

# create a map from token to a unique number, which is the index of that tokens in the sorted `tokens` list
tokens = sorted(tokens)
token2idx = {token: i for i, token in enumerate(tokens)}
print("total number of tokens: ", len(tokens)

```

Tokenize all the input sentence and masked_sentence, pad all the sentence to the same length using the id of the `<PAD>` token. This must be done to df["sentence"] and df["masked_sentence"]

In addition, we also need to add one more feature to the model, which is the ratio between the length of the masked texts over the length of the sentence (without padding). The reason behind this is that a longer masked phrase tends to be more complex compare to a shorter one.

```python
# tokenize sentences and masked_sentences
tokenized_sentences = [[token2idx[token] for token in sentence.split(" ")] for sentence in df.sentence.tolist()]
tokenized_masked_sentences = [[token2idx[token] for token in sentence.split(" ")] for sentence in df.masked_sentence.tolist()]

# pad them to the same length
max_length = max([len(_) for _ in tokenized_sentences])
pad_token = token2idx["<PAD>"]
tokenized_sentences = np.array([(sentence + [pad_token]*max_length)[:max_length] for sentence in tokenized_sentences])
tokenized_masked_sentences = np.array([(sentence + [pad_token]*max_length)[:max_length] for sentence in tokenized_masked_sentences])

# calculate the length of the target words relative to the total length of the sentence
target_length = np.array((df.end_index - df.start_index)/(df.end_index - df.start_index).max())

# getting the label
y_true = np.array(df.complexity.tolist()
```

## Split the data for K-fold cross validations

## Build the data generator

## Build the model

## Train the model

## K-fold Validation

## Load and preprocess the test data

## Load the best model and make the predictions for test data
