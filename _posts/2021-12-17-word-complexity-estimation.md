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
* create the new columns with the masked tokens is replaced by <MASK>

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


## Split the data for K-fold cross validations

## Build the data generator

## Build the model

## Train the model

## K-fold Validation

## Load and preprocess the test data

## Load the best model and make the predictions for test data
