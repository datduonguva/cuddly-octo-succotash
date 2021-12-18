---
layout: post
title:  Word Complexity Estimation
tags: "Machine Learning"
---

## Introduction
Predicting word complexity is a natural language processing task that given an input string and a specified substring, the model should make a prediction for the complexity of that substring. The data set were annotated by having human labellers grading the level of complexity fo the word phrase. The complexity of the a phrase is defined as the number of annotators who mark the phrase as complex over total number of annotators

## Load required packages
```
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
```

## Read the training data
Let's import the data and take a quick look at them

```
df = pd.read_csv("train_full.txt", header=None, delimiter = "\t")
df.columns = [
    "id", "sentence", "start_index", "end_index", "word", "num_native", "num_non_native",
    "num_native_yes", "num_non_native_yes", "complexity"
]

df.head()
```

## Preprocessing

## Split the data for K-fold cross validations

## Build the data generator

## Build the model

## Train the model

## K-fold Validation

## Load and preprocess the test data

## Load the best model and make the predictions for test data
