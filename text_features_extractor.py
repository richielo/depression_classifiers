#!/usr/bin/python
#coding:utf-8
"""
Author: Richie Lo
Email: richielo@connect.hku.hk
Description: The code provides utility functions to extract text sentiment and readability features, using LIWC, ANEW, GunningFog Index etc. his is primarily used for the course project of IIMT4601, University of Hong Kong.
"""
from __future__ import print_function
import os
import sys
import csv
import time
import statistics
import numpy as np
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import tokenize
from nltk import word_tokenize
import gensim
from gensim import corpora
from hanziconv import HanziConv
import jieba
import chardet
from zhon import hanzi
import re
#from stanfordcorenlp import StanfordCoreNLP

#For readability
#from textstat.textstat import textstat as ts
#from readcalc import readcalc
#import nltk
#nltk.download("punkt")

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
anewEngPath = "anew_dicts/EnglishShortened.csv"
anewChinPath = "anew_dicts/cvaw2.csv"

#-------------------------Utility functions---------------------------------
#Remove dictionary entries based on keys
def filterDict(rawDict, filters):
    cleanedDict = {}
    for f in filters:
        for key in rawDict.keys():
            if(f in key):
                cleanedDict[key] = rawDict[key]
    return cleanedDict

#Split Chinese and English
def splitChinEng(text):
    #Confirm whether the line below is needed or not
    #text = unicode(text, 'utf-8')
    foundChin = re.findall('[{}]'.format(hanzi.characters), text)
    foundEng =  re.sub("[^A-Za-z]", " ", text)
    #print("Found Chin:")
    #print(" ".join(chinPreprocessing(''.join(foundChin))))
    #print("Found Eng:")
    #print(" ".join(foundEng.split()))
    return ''.join(foundChin), " ".join(foundEng.split())
#---------------------------------------------------------------------------


#------- LIWC Chinese features extractor functions--------------------------

def removePuncFromContext(contextList):
    nopuncContextList = []
    for context in contextList:
        if(context not in stop):
            nopuncContextList.append(context)
    return nopuncContextList

def removeChinStopwords(context):
    # read stopwords list from local file
    stop_f = open('stopwords/stopwords-zh.txt','r',encoding='utf-8')
    stopwords = [l.strip() for l in stop_f.readlines()]
    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].encode("utf8", errors="ignore")
    #print(stopwords)
    stop_f.close()
    clean = [t for t in context if t not in stopwords]
    return clean

def chinPreprocessing(text):
    simChinText = HanziConv.toSimplified(text.replace(' ', ''))
    segList = jieba.cut(simChinText, cut_all=False)
    cleanedText = removePuncFromContext(removeChinStopwords(segList))
    return cleanedText

def parseChinCatDict(catLines, catDict):
    for cLine in catLines:
        catTokens = cLine.replace('\t', '').split(' ')
        catTokens = [t for t in catTokens if t != '']
        catDict[catTokens[0]] = catTokens[1]
    return catDict

def parseChinWordDict(wordLines, wordDict):
    for wLine in wordLines:
        wordTokens = wLine.replace('\n', '').split(' ')
        word = wordTokens.pop(0)
        word = unicode(word, "utf8", errors="ignore")
        wordDict[word] = []
        for t in wordTokens:
            wordDict[word].append(t)
    return wordDict

def chinParseLIWC(liwcDict,catDict, wordDict):
    catLines = []
    wordLines = []
    inCat = True
    # Get relevant lines for categories and words
    for line in liwcDict:
        if('%' in line):
            inCat = False
        else:
            if(inCat):
                catLines.append(line)
            else:
                wordLines.append(line)
    catDict = parseChinCatDict(catLines, catDict)
    wordDict = parseChinWordDict(wordLines, wordDict)
    return catDict,wordDict

def chinInitLiwcParse():
    #Import and parse LIWC dictionary
    path_to_liwc = 'liwc_dicts/sc_liwc.dic'
    liwcDict = open(path_to_liwc, 'r')
    #Dictionary for categories, words and stem words, ready for parsing
    catDict = {}
    wordDict = {}
    #stemWordDict = {}
    return chinParseLIWC(liwcDict,catDict, wordDict)

def chinLiwcAnalysis(text, liwcResults, liwcWordTriggersResults, catDict, wordDict):
    dictWords = wordDict.keys()
    for word in dictWords:
        frequency = 0.0
        for c in text:
            if(word in c):
                frequency += 1.0
        #Update categorical results
        if(frequency > 0):
            wordCatCodes = wordDict[word]
            for catCode in wordCatCodes:
                catName = catDict[catCode]
                liwcResults[catName] += frequency
                liwcWordTriggersResults[catName].append(HanziConv.toTraditional(word))
    # normalize results
    cleanedTextLength = len(text)
    for k in liwcResults.keys():
        liwcResults[k] =  (float(liwcResults[k]) / float(cleanedTextLength))
        
    return liwcResults, liwcWordTriggersResults

#----------------------------------------------------------------------


#------- LIWC English features extractor functions---------------------

def engPreprocessing(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    return punc_free

def engParseLIWC(liwcDict, catDict, wordDict, stemWordDict):
    separatorCount = 0
    catCount = 0
    wordCount = 0
    stemWordCount = 0
    for line in liwcDict:
        if('%' in line):
            separatorCount += 1
            continue
        tokens = line.split('\t')
        tokens[-1]. replace('\n', '')
        if(separatorCount == 1):
            #Parse category
            catCount += 1
            catDict[tokens[0]] = tokens[1]
        else:
            # Parse words
            #Check whether it's stem word
            if('*' in tokens[0]):
                stemWordCount += 1
                stemWordDict[tokens[0].replace('*', '')] = tokens[1:]
            else:
                wordCount += 1
                wordDict[tokens[0]] = tokens[1:]
    #print("-------Dictionary parsing summary-------")
    #print("Category count: " + str(catCount))
    #print("Non-stem word count: " + str(wordCount))
    #print("Stem word count: " + str(stemWordCount))

def createCountDict(keys):
    tempDict = {}
    for key in keys:
        tempDict[key] = 0
    return tempDict

def getWordCat(catDict, wordDict, word):
    if(wordDict.get(word) != None):
        categories = wordDict[word]
        catNames = []
        for cat in categories:
            catNames.append(catDict[cat.replace('\n', '')])
        return catNames
    return []

def getStemWordCat(catDict, stemWordDict, word):
    catNames = []
    for key in stemWordDict.keys():
        if word.startswith(key):
            categories = stemWordDict[key]
            for cat in categories:
                catNames.append(catDict[cat.replace('\n', '')])
    return catNames

def engInitLiwcParse():
    #Import and parse LIWC dictionary
    path_to_liwc = 'liwc_dicts/LIWC2007_English080730.dic'
    liwcDict = open(path_to_liwc, 'r')
    #Dictionary for categories, words and stem words, ready for parsing
    catDict = {}
    wordDict = {}
    stemWordDict = {}

    #Begin parsing
    engParseLIWC(liwcDict, catDict, wordDict, stemWordDict)
    return catDict, wordDict, stemWordDict

# expect 1 string of text
def engLiwcAnalysis(text, liwcResults, liwcWordTriggersResults, catDict, wordDict, stemWordDict):
    #Count Contents
    nopunc_content_tokens  = text.replace('.', '').replace(',', '').replace('?', '').replace('!', '').lower().split(' ')
    for c in nopunc_content_tokens:
        catWordList = getWordCat(catDict, wordDict, c)
        if(catWordList == []):
            catStemList = getStemWordCat(catDict, stemWordDict, c)
            for cat in catStemList:
                liwcResults[cat] += 1
                liwcWordTriggersResults[cat].append(c)
        else:
            for cat in catWordList:
                liwcResults[cat] += 1
                liwcWordTriggersResults[cat].append(c)
    #convert to percentage
    for key in liwcResults.keys():
        liwcResults[key] = float(liwcResults[key])/len(nopunc_content_tokens)
    
    return liwcResults, liwcWordTriggersResults

def getLiwcResultsAggregate(batchResultList):
    aggResults = {}
    aggResults['triggers'] = []
    statDicts = []
    for result in batchResultList:
        aggResults['triggers'].append(result['triggers'])
        statDicts.append(result['stats'])
    statDictsDf = pd.DataFrame(statDicts)
    aggStatDict = statDictsDf.mean().to_dict()
    #Weighted average?
    aggResults['agg_stats'] = aggStatDict
    return aggResults
#--------------------------------------------------------------------------

#-------------------------------Batch run LIWC-----------------------------
#batch run liwc
def batchLiwcRun(textList, resultType, filters):
    print("Run LIWC")
    #Init both english and chinese liwc resources
    engCatDict, engWordDict, engStemWordDict = engInitLiwcParse()
    chinCatDict, chinWordDict = chinInitLiwcParse()
    batchResultList = []
    for text in textList:
        #Split the text into english and chinese
        text = re.sub('\W+',' ', text)
        totalLength = len(text)
        chinText, engText = splitChinEng(text)
        print("text: " + str(text.encode('utf-8')))
        print("chinText: " + str(chinText))
        chinLength = len(chinText)
        engLength = len(engText)
        mergedLiwcResults = {}
        
        #Init results data structures
        engLiwcResults = {}
        engLiwcWordTriggersResults = {}
        for key in engCatDict.keys():
            engLiwcResults[engCatDict[key]] = 0.0
            engLiwcWordTriggersResults[engCatDict[key]] = []
        if(engLength > 0):
            #Preprocessing
            cleanedEngText = engPreprocessing(engText)
            engLength = len(cleanedEngText)
            #run liwc
            engLiwcResults, engLiwcWordTriggersResults = engLiwcAnalysis(cleanedEngText, engLiwcResults, engLiwcWordTriggersResults, engCatDict, engWordDict, engStemWordDict)
            #filter out unwanted columns
        if(filters):
            engLiwcResults = filterDict(engLiwcResults, filters)
            engLiwcWordTriggersResults = filterDict(engLiwcWordTriggersResults, filters)
        #Init results data structures
        chinLiwcResults = {}
        chinLiwcTriggerResults = {}
        for k in chinCatDict.keys():
            chinLiwcResults[chinCatDict[k]] = 0.0 
            chinLiwcTriggerResults[chinCatDict[k]] = []
        if(chinLength > 0):
            #text = text.decode('utf-8')
            #Preprocessing
            cleanedChinText = chinPreprocessing(chinText)
            chinLength = len(''.join(cleanedChinText))
            #run liwc
            chinLiwcResults, chinLiwcTriggerResults = chinLiwcAnalysis(cleanedChinText, chinLiwcResults, chinLiwcTriggerResults, chinCatDict, chinWordDict)
            #filter out unwanted columns
        if(filters):
            chinLiwcResults = filterDict(chinLiwcResults, filters)
            chinLiwcTriggerResults = filterDict(chinLiwcTriggerResults, filters)
        #Get weighted average and merge result of chin and eng (Be careful, Chin and Eng don't have same keys)
        #Get common keys and different keys
        totalLength = engLength + chinLength
        chinEngCommonKeys = list(set(engLiwcResults.keys()) & set(chinLiwcResults.keys()))
        engOnlyKeys = list(set(engLiwcResults.keys()) - set(chinLiwcResults.keys()))
        chinOnlyKeys = list(set(chinLiwcResults.keys()) - set(engLiwcResults.keys()))
        mergedStatsResults = {}
        mergedTriggersResults = {}
        for k in chinEngCommonKeys:
            mergedStatsResults[k.replace('\n','')] = float(engLength) / float(totalLength) * engLiwcResults[k] + float(chinLength) / float(totalLength) * chinLiwcResults[k]
            mergedTriggersResults[k.replace('\n','')] = []
            mergedTriggersResults[k.replace('\n','')] = engLiwcWordTriggersResults[k] + chinLiwcTriggerResults[k]
        for k in engOnlyKeys:
            mergedStatsResults[k.replace('\n','')] = float(engLength) / float(totalLength) * engLiwcResults[k]
            mergedTriggersResults[k.replace('\n','')] = []
            mergedTriggersResults[k.replace('\n','')] = engLiwcWordTriggersResults[k]
        for k in chinOnlyKeys:
            mergedStatsResults[k.replace('\n','')] = float(chinLength) / float(totalLength) * chinLiwcResults[k]
            mergedTriggersResults[k.replace('\n','')] = []
            mergedTriggersResults[k.replace('\n','')] = chinLiwcTriggerResults[k]
        #print('done merging') 
        #Combine stats and triggers
        mergedLiwcResults['stats'] = mergedStatsResults
        mergedLiwcResults['triggers'] = mergedTriggersResults
        #print(mergedLiwcResults)
        #Add to batch result
        batchResultList.append(mergedLiwcResults)
        

    if(resultType == "aggregate"):
        #Perform further processing
        batchResultList = getLiwcResultsAggregate(batchResultList)
        
    return batchResultList

'''
#Deprecated liwc batch run function -- DO NOT DELETE UNTIL NEW ONE IS PROVEN TO WORK

def batchLiwcRun(textList, resultType, filters):
    print("Run liwc")
    #Init both english and chinese liwc resources
    engCatDict, engWordDict, engStemWordDict = engInitLiwcParse()
    chinCatDict, chinWordDict = chinInitLiwcParse()
    batchResultList = []
    for text in textList:
        #print("encoding: " + str(chardet.detect(text)['encoding']))
        #ASSUME ONLY CHIN and ENG, lang detect is not accurate
        lang = detect(text) 
        print("lang detected: " + lang)
        if(lang == 'en'):
            #Init results data structures
            engMergedLiwcResults = {}
            emgLiwcResults = {}
            engLiwcWordTriggersResults = {}
            for key in engCatDict.keys():
                emgLiwcResults[engCatDict[key]] = 0.0
                engLiwcWordTriggersResults[engCatDict[key]] = []
            #Preprocessing
            cleanedText = engPreprocessing(text)
            #run liwc
            emgLiwcResults, engLiwcWordTriggersResults = engLiwcAnalysis(cleanedText, emgLiwcResults, engLiwcWordTriggersResults, engCatDict, engWordDict, engStemWordDict)
            #filter out unwanted columns
            if(filters):
                emgLiwcResults = filterDict(emgLiwcResults, filters)
                engLiwcWordTriggersResults = filterDict(engLiwcWordTriggersResults, filters)
            #Combine stats and triggers
            engMergedLiwcResults['stats'] = emgLiwcResults
            engMergedLiwcResults['triggers'] = engLiwcWordTriggersResults
            batchResultList.append(engMergedLiwcResults)
        else:
            #text = text.decode('utf-8')
            #Init results data structures
            chinMergedLiwcResults = {}
            chinLiwcResults = {}
            chinLiwcTriggerResults = {}
            for k in chinCatDict.keys():
                chinLiwcResults[chinCatDict[k]] = 0.0 
                chinLiwcTriggerResults[chinCatDict[k]] = []
            #Preprocessing
            cleanedText = chinPreprocessing(text)
            #run liwc
            chinLiwcResults, chinLiwcTriggerResults = chinLiwcAnalysis(cleanedText, chinLiwcResults, chinLiwcTriggerResults, chinCatDict, chinWordDict)
            #filter out unwanted columns
            if(filters):
                chinLiwcResults = filterDict(chinLiwcResults, filters)
                chinLiwcTriggerResults = filterDict(chinLiwcTriggerResults, filters)
            #Combine stats and triggers
            chinMergedLiwcResults['stats'] = chinLiwcResults
            chinMergedLiwcResults['triggers'] = chinLiwcTriggerResults
            batchResultList.append(chinMergedLiwcResults)
        """
        else:
            # text list contains unsupported language
            return 0;
        """

    if(resultType == "aggregate"):
        #Perform further processing
        batchResultList = getLiwcResultsAggregate(batchResultList)
        
    return batchResultList
'''
#--------------------------------------------------------------------------

#-------------GunningFog Index feature extractor functions------------------
"""
def getGunningFogIndex(text):
    calc = readcalc.ReadCalc(text)
    return calc.get_gunning_fog_index()
"""
#---------------------------------------------------------------------------


#------------ANEW features extractor (English)------------------------------
#----------Adapted from Doris Zhou (https://github.com/dwzhou/SentimentAnalysis)
#----------NLTK POS-Tagger is used here instead of StanfordCoreNlp

def engAnewAnalysis(fulltext, mode='mean'):

    # end method if file is empty
    if len(fulltext) < 1:
        return {'text': fulltext, "valence": 0.0, "arousal": 0.0, "triggers":[]}

    lmtzr = WordNetLemmatizer()

    # print("S" + str(i) +": " + s)
    all_words = []
    found_words = []
    total_words = 0
    v_list = []  # holds valence scores
    a_list = []  # holds arousal scores
    d_list = []  # holds dominance scores

    # search for each valid word's sentiment in ANEW
    tokens = word_tokenize(fulltext.lower())
    words = nltk.pos_tag(tokens)
    for index, p in enumerate(words):
        # don't process stop or words w/ punctuation
        w = p[0]
        pos = p[1]
        if w in stop or not w.isalpha():
            continue

        # check for negation in 3 words before current word
        j = index-1
        neg = False
        while j >= 0 and j >= index-3:
            if words[j][0] == 'not' or words[j][0] == 'no' or words[j][0] == 'n\'t':
                neg = True
                break
            j -= 1

        # lemmatize word based on pos
        if pos[0] == 'N' or pos[0] == 'V':
            lemma = lmtzr.lemmatize(w, pos=pos[0].lower())
        else:
            lemma = w

        all_words.append(lemma)
        # search for lemmatized word in ANEW
        with open(anewEngPath) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Word'].lower() == lemma.lower():
                    if neg:
                        found_words.append("neg-"+lemma)
                    else:
                        found_words.append(lemma)
                    v = float(row['valence'])
                    a = float(row['arousal'])
                    d = float(row['dominance'])

                    if neg:
                        # reverse polarity for this word
                        v = 5 - (v - 5)
                        a = 5 - (a - 5)
                        d = 5 - (d - 5)

                    v_list.append(v)
                    a_list.append(a)
                    d_list.append(d)
    if len(found_words) == 0:  # no words found in ANEW for this sentence
        '''print({'text': fulltext,
                            'Sentiment': 'N/A',
                            'Sentiment Label': 'N/A',
                            'Arousal': 'N/A',
                            'Dominance': 'N/A',
                            '# Words Found': 0,
                            'Found Words': 'N/A',
                            'All Words': all_words
                            })
        '''
        return {'text': fulltext, "valence": 0.0, "arousal": 0.0, "triggers":[]}
    else:  # output sentiment info for this sentence

        # get values
        sentiment = 0.0
        arousal = 0.0
        dominance = 0.0
        if mode == 'median':
            sentiment = statistics.median(v_list)
            arousal = statistics.median(a_list)
            dominance = statistics.median(d_list)
        else:
            sentiment = statistics.mean(v_list)
            arousal = statistics.mean(a_list)
            dominance = statistics.mean(d_list)

        # set sentiment label
        label = 'neutral'
        if sentiment > 6:
            label = 'positive'
        elif sentiment < 4:
            label = 'negative'
        '''
        print({'text': fulltext,
                            'Sentiment': sentiment,
                            'Sentiment Label': label,
                            'Arousal': arousal,
                            'Dominance': dominance,
                            '# Words Found': ("%d out of %d" % (len(found_words), len(all_words))),
                            'Found Words': found_words,
                            'All Words': all_words
                            })
        '''
        return {'text': fulltext, "valence": (sentiment-1.0)/8.0, "arousal": (arousal-1.0)/8.0, "triggers":found_words}

#--------------------------------------------------------------------------


#------------ANEW features extractor (Chinese)------------------------------
def chinAnewAnalysis(dictDf, dictList, textList):
    cvawResult = {'text':textList,'valence':0.0, 'arousal':0.0, 'triggers':[]}
    triggerWords = []
    vList = []
    aList = []
    for dictEntry in dictList:
        cDictEntry = HanziConv.toSimplified(dictEntry.replace(' ', ''))
        for word in textList:
            if(cDictEntry in word):
                row = dictDf[dictDf['Word'] == dictEntry]
                vList.append(row['Valence_Mean'].iloc[0])
                aList.append(row['Arousal_Mean'].iloc[0])
                triggerWords.append(HanziConv.toTraditional(cDictEntry))
    if(len(triggerWords) > 0):
        cvawResult['valence'] = (np.mean(vList)-1.0)/8.0
        cvawResult['arousal'] = (np.mean(aList)-1.0)/8.0
        cvawResult['triggers'] = triggerWords
    return cvawResult
#--------------------------------------------------------------------------

#------------Batch run ANEW------------------------------------------------
#TODO: return top words instead
def getAnewResultsAggregate(batchResultList):
    aggResults = {'valence':0.0,'arousal':0.0}
    triggerList = []
    for result in batchResultList:
        aggResults['valence'] += result['valence']
        aggResults['arousal'] += result['arousal']
        triggerList.append(result['triggers'])
    aggResults['valence'] = aggResults['valence'] / float(len(batchResultList))
    aggResults['arousal'] = aggResults['arousal'] / float(len(batchResultList))
    aggResults['triggers'] = triggerList
    return aggResults  

def batchAnewRun(textList, resultType):
    print("Run ANEW")
    #Read in Chin Dict
    chinDictDf = pd.read_csv(anewChinPath,encoding='utf-8')
    chinDictList = chinDictDf['Word'].tolist()

    batchResultList = []
    for text in textList:
        #Split the text into english and chinese
        text = re.sub('\W+',' ', text)
        print("text:" + str(text))
        totalLength = 0
        chinText, engText = splitChinEng(text)
        chinLength = len(chinText)
        engLength = len(engText)
        mergedAnewDict = {'text':text,'valence':0.0,'arousal':0.0, 'triggers':[]}
        engAnewDict = {'text':engText,'valence':0.0,'arousal':0.0, 'triggers':[]}
        chinAnewDict = {'text':chinText,'valence':0.0,'arousal':0.0, 'triggers':[]}
        if(engLength > 0):
            cleanedEngText = engPreprocessing(engText)
            engLength = len(cleanedEngText)
            if(engLength > 0):
                totalLength += engLength
                engAnewDict = engAnewAnalysis(cleanedEngText, mode='mean')
            else:
                engLength = 0
        if(chinLength > 0):
            cleanedChinText = chinPreprocessing(chinText)
            chinLength  = len(cleanedChinText)
            if(chinLength > 0):
                totalLength += chinLength
                chinAnewDict = chinAnewAnalysis(chinDictDf, chinDictList, cleanedChinText)
            else:
                chinLength = 0
        mergedAnewDict['valence'] = (float(engLength) / float(totalLength)) * float(engAnewDict['valence']) + (float(chinLength) / float(totalLength)) * float(chinAnewDict['valence'])
        mergedAnewDict['arousal'] = float(engLength) / float(totalLength) * float(engAnewDict['arousal']) + float(chinLength) / float(totalLength) * float(chinAnewDict['arousal'])
        mergedAnewDict['triggers'] = chinAnewDict['triggers'] + engAnewDict['triggers']
        batchResultList.append(mergedAnewDict)

    if(resultType == "aggregate"):
        batchResultList = getAnewResultsAggregate(batchResultList)
        
    
    return batchResultList

#------------Punctuation Statistics extractor------------------------------
def getPuncStats(text):
    textNoSpace = text.replace(' ', '')
    print("textNoSpace: " + str(textNoSpace))
    #Count question mark and exclamation mark
    questionMarkCount = textNoSpace.count('?')
    exclaimationMarkCount = textNoSpace.count('!')
    #count the use of ...
    tripleDotsCount = 0
    countIndex = 0
    while(countIndex < len(textNoSpace)):
        if(textNoSpace[countIndex] == '.'):
            tempIndex = countIndex
            dotCount = 0
            while(textNoSpace[tempIndex] == '.'):
                dotCount += 1
                tempIndex += 1
                if(tempIndex >= len(textNoSpace)):
                    break
            if(dotCount >= 3):
                tripleDotsCount += 1
            countIndex = tempIndex
        else:
            countIndex += 1
    puncStats = {'questionMarkCount':questionMarkCount,'exclaimationMarkCount':exclaimationMarkCount, 'tripleDotsCount':tripleDotsCount}
    return puncStats

#------------TODO: 1 call to extract all features--------------------------
