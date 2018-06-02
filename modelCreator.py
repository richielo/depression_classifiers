from __future__ import print_function
from IPython.display import display, HTML
import os
import sys
import csv
import time
import statistics
import numpy as np
import string
import re
import pandas as pd
import text_features_extractor as tfExtractor
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV
#Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Input, LSTM, Bidirectional, Flatten, Dropout, Concatenate
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline
import tensorflow
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import pickle

def createCnnMcModel(length, vocab_size):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 50)(inputs1)
    conv1 = Conv1D(filters=100, kernel_size=2, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 50)(inputs2)
    conv2 = Conv1D(filters=100, kernel_size=3, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 50)(inputs3)
    conv3 = Conv1D(filters=100, kernel_size=4, activation='relu')(embedding3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # channel 4
    inputs4 = Input(shape=(length,))
    embedding4 = Embedding(vocab_size, 50)(inputs4)
    conv4 = Conv1D(filters=100, kernel_size=5, activation='relu')(embedding4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling1D(pool_size=2)(drop4)
    flat4 = Flatten()(pool4)
    # merge
    merged = concatenate([flat1, flat2, flat3, flat4])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    #plot_model(model, show_shapes=True, to_file='cnnmc_model.png')
    return model

def createLexCnnMcModel(maxDim, featureLength):
    # channel 1
    inputs1 = Input(shape=(maxDim, featureLength))
    conv1 = Conv1D(filters=100, kernel_size=2, activation='relu', input_shape=(maxDim,featureLength))(inputs1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(maxDim, featureLength))
    conv2 = Conv1D(filters=100, kernel_size=3, activation='relu', input_shape=(maxDim,featureLength))(inputs2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)
    # channel 3
    inputs3 = Input(shape=(maxDim, featureLength))
    conv3 = Conv1D(filters=100, kernel_size=4, activation='relu', input_shape=(maxDim,featureLength))(inputs3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)
    # channel 4
    inputs4 = Input(shape=(maxDim, featureLength))
    conv4 = Conv1D(filters=100, kernel_size=5, activation='relu', input_shape=(maxDim,featureLength))(inputs4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling1D(pool_size=2)(drop4)
    flat4 = Flatten()(pool4)
    # merge
    merged = concatenate([flat1, flat2, flat3, flat4])
    # interpretation
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    #plot_model(model, show_shapes=True, to_file='cnnmc_model.png')
    return model

def mergeCnnModel(cnnModel, lexCnnModel):
    merged = concatenate([cnnModel.layers[-2].output, lexCnnModel.layers[-2].output])
    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[cnnModel.input, lexCnnModel.input], outputs=outputs)
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='mergedLexCnnModel.png')
    return model