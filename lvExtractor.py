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
import jieba.posseg as pseg

#Read in HowNet
hnPosSentFile = open("sentiment_lexicons/sentiment/正面情感词语（中文）.txt", encoding="utf8")
hnPosSentWords = []
hnCount = 0
for line in hnPosSentFile:
    hnCount += 1
    if(hnCount <= 2):
        continue
    hnPosSentWords.append(line.replace('\n','').replace(' ', ''))
hnNegSentFile = open("sentiment_lexicons/sentiment/负面情感词语（中文）.txt", encoding="utf8")
hnNegSentWords = []
hnCount = 0
for line in hnNegSentFile:
    hnCount += 1
    if(hnCount <= 2):
        continue
    hnNegSentWords.append(line.replace('\n','').replace(' ', ''))
hnNegOpFile = open("sentiment_lexicons/sentiment/负面评价词语（中文）.txt", encoding="utf8")
hnNegOpWords = []
hnCount = 0
for line in hnNegOpFile:
    hnCount += 1
    if(hnCount <= 2):
        continue
    hnNegOpWords.append(line.replace('\n','').replace(' ', ''))
hnDegreeFile = open("sentiment_lexicons/sentiment/程度级别词语（中文）.txt", encoding="utf8")
hnDegreeWords = []
hnCount = 0
for line in hnDegreeFile:
    hnCount += 1
    if(hnCount <= 2):
        continue
    hnDegreeWords.append(line.replace('\n','').replace(' ', ''))
hnSubjectiveFile = open("sentiment_lexicons/sentiment/主张词语（中文）.txt", encoding="utf8")
hnSubjectiveWords = []
hnCount = 0
for line in hnSubjectiveFile:
    hnCount += 1
    if(hnCount <= 2):
        continue
    hnSubjectiveWords.append(line.replace('\n','').replace(' ', ''))
#Read in NTUSD
ntPosFile = open("sentiment_lexicons/ntusd-positive.txt", encoding="utf8")
ntPosWords = []
for line in ntPosFile:
    ntPosWords.append(line.replace('\n','').replace(' ', ''))
ntNegFile = open("sentiment_lexicons/ntusd-negative.txt", encoding="utf8")
ntNegWords = []
for line in ntNegFile:
    ntNegWords.append(line.replace('\n','').replace(' ', ''))
eoLexiconDf = pd.read_csv("sentiment_lexicons/EOLexicon.csv", encoding='UTF-8')

def searchEOLexicon(word, lexVec):
    searchDf = eoLexiconDf[eoLexiconDf.词语 == word]
    if(searchDf.shape[0] > 0):
        #Found word in lexicon
        sentiment = str(searchDf["情感分类"].iloc[0])[0].upper()
        polarity = searchDf["极性"].iloc[0]
        pos = searchDf["词性种类"].iloc[0]
        if(sentiment == 'N' or sentiment):
            #negative 
            lexVec[0][0] = 0
            lexVec[0][1] = 1
        else:
            lexVec[0][0] = 1
            lexVec[0][1] = 0
        if(polarity == 2 or polarity == 3):
            lexVec[0][4] = 1
        return True,lexVec
    return False,lexVec

def searchNtusd(word, lexVec):
    if(word in ntPosWords):
        lexVec[0][0] = 1
        lexVec[0][1] = 0
    elif(word in ntNegWords):
        lexVec[0][0] = 0
        lexVec[0][1] = 1
    else:
        return False, lexVec
    return True, lexVec

def searchHowNet(word, lexVec, foundSent):
    if(foundSent == False):
        #Need to search for sentiment
        if(word in hnPosSentWords):
            lexVec[0][0] = 1
            lexVec[0][1] = 0
        elif(word in hnNegSentWords):
            lexVec[0][0] = 0
            lexVec[0][1] = 1
    if(word not in hnSubjectiveWords):
        lexVec[0][3] = 1
    if(word in hnNegOpWords):
        lexVec[0][4] = 1
    if(word in hnDegreeWords):
        lexVec[0][5] = 1
    return lexVec
    
def getLexiconVector(text):
    words = pseg.cut(text)
    lexVecs = []
    for word in words:
        lexVec = np.zeros([1,10])
        foundInEo, lexVec = searchEOLexicon(word.word, lexVec)
        if(foundInEo):
            lexVec = searchHowNet(word.word, lexVec, foundInEo)
        else:
            foundInNtusd, lexVec = searchNtusd(word.word, lexVec)
            lexVec = searchHowNet(word.word, lexVec, foundInNtusd)
        if(lexVec[0][0] == 0 and lexVec[0][1] == 0):
            #Neutral
            lexVec[0][2] = 1
        if(word.flag[0] == 'n'):
            lexVec[0][6] = 1
        elif(word.flag[0] == 'v'):
            lexVec[0][7] = 1
        elif(word.flag[0] == 'a'):
            lexVec[0][8] = 1
        elif(word.flag[0] == 'd'):
            lexVec[0][9] = 1
        lexVecs.append(lexVec)
    return np.stack(lexVecs).reshape(len(lexVecs), 10, 1)