
import re
import numpy as np
import sys
import math
import collections
import pandas as pd

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import nlp_util as nu
import data_util as du

#initialize key variables..

sno = nltk.stem.SnowballStemmer('english')

exclude_stopwords = ['in','can','i','is','why','what','how','where','when','no','have','having','not','off']
include_stopwords = ['xxx','xxxx']
init_flag = 0

#function based initialization
def initVariables():
  stopWords = set(stopwords.words('english'))
  #print("init_flag=[{}]".format(init_flag)) 
  for x in exclude_stopwords:
    #print("excluding [{}]".format(x))
    if x in stopWords:
      stopWords.remove(x)
      #print("excluded [{}]".format(x))
  for x in include_stopwords:
    #print("including [{}]".format(x))
    stopWords.add(x)
  
  return stopWords
'''
    if x in stopWords:
      #print("included [{}]".format(x))
  init_flag = 1

#Call initialization of Variables.
initVariables()
'''

def processPunctuation(data): 
  tokenizer = RegexpTokenizer(r'\w+')
   
  return tokenizer.tokenize(data)

def removeStopWords(data): 
  '''
  global init_flag
  global stopWords
   
  if init_flag == 0: #check for initialization
  '''
  stopWords = initVariables()

  #print(stopWords) 
  #words = word_tokenize(data)
  wordsFiltered = []
    
  for w in data:
    if w not in stopWords:
      wordsFiltered.append(w)
   
  return(wordsFiltered)

def stemAndGetWord(w):
  return sno.stem(w.lower())

def stemAndGetSentence(data):
  #sno = nltk.stem.SnowballStemmer('english')
  str = ''

  for w in data:
    #print(w)
    str = str + sno.stem(w.lower()) + ' '
  #print(str)
  
  return str

def processWordsWithNLTK1(data):
  #print(data)
  data = processPunctuation(data)
  #print(data)
  data = removeStopWords(data)
  #print(data)
  data = stemAndGetSentence(data)
  #print(word_tokenize(data))
  
  return word_tokenize(data)

def processWordsWithNLTK(data):
  data = processPunctuation(data)
  #print(data)
  data = stemAndGetSentence(data)
  #print(data)
  
  #return removeStopWords(data)
  data = word_tokenize(data)

  return data

#Vocabulary class
class Vocab:

  def createVocab(self):
    self.cntr = collections.Counter()
    
    for i in range(len(self.data)):
      if type(self.data[i]) is list:
        for word in self.data[i]:
          self.cntr[word] += 1
      else:
         self.cntr[self.data[i]] += 1
    
    self.dict_size = len(self.cntr)
    print("Total words in data set: ", self.dict_size)

    #truncate vocab of data to carry fwd just 10% of top dictinary words
    if self.dict_size > 100:
      self.dict_size = int(round(self.dict_size*self.dict_ratio))
      print("Words dictionary truncated to: ", self.dict_size)
  
    self.vocab = sorted(self.cntr, key=self.cntr.get, reverse=True)[:self.dict_size]
    
    #Add not in vocab tag and update dict_size
    self.vocab.append(self.UNK)
    #self.vocab.append('<NUM>')
    
    #update dict_size again
    self.dict_size = len(self.vocab)
     
    print(self.vocab[:60])
    print('#no of time last word from vocab: ',self.vocab[-1], ': ', self.cntr[self.vocab[-1]])
    
    vocab1 = sorted(self.cntr, key=self.cntr.get, reverse=True)[:len(self.cntr)]
    if self.dict_ratio == 1: #Since label do not truncate for train
      du.printCollection(self.cntr,type="Test")     
    else: 
      #create vocab csv
      v_df = pd.DataFrame(columns=["word","occurence","truncated_flag"])
      for i,v in enumerate(self.vocab):
        v_df.loc[i] = [v,self.cntr[v],0]
      
      truncated_vocab = sorted(self.cntr, key=self.cntr.get, reverse=True)[self.dict_size+1:]
      for i,v in enumerate(truncated_vocab):
        v_df.loc[v_df.shape[0]] = [v,self.cntr[v],1]
      
      v_df.to_csv(self.config.datadir + self.config.model_name + "_vocab.csv")
      
      #print the Collection stats for Vocab
      du.printCollection(self.cntr,type="Train")     

    self.word2idx = {}
    for i,word in enumerate(self.vocab):
       self.word2idx[word] = i
    print('few keys of word2idx :',{k: self.word2idx[k] for k in self.word2idx.keys()[:5]})
  
    self.idx2word = dict((v, k) for k, v in self.word2idx.items()) 
    print('few keys of idx2word :',{k: self.idx2word[k] for k in self.idx2word.keys()[:5]})
   
  def __init__(self,idata,config,label=False):
     self.config = config
     if label == True:
       self.dict_ratio = 1 #If vocab is for label then do not truncate dictionary size as each vector is imp
     else:
       self.dict_ratio = self.config.vocab_dict_ratio
     self.data = idata
     self.vocab = []
     self.word2idx = {}
     self.idx2word = {}
     self.cntr = None
     self.UNK = 'IG'
     self.createVocab()

  def convData(self,data):
     rawdata = []
     for i in range(len(data)):
       if type(data[i]) is list:
         for idx in data[i]:
           rawdata.append(self.idx2word[idx]) 
       else:
         rawdata.append(self.idx2word[data[i]]) 

     return rawdata

  def getData(self):
     rawdata = []
     for i in range(len(self.data)):
       if type(self.data[i]) is list:
         for word in self.data[i]:
           rawdata.append(word) 
       else:
         rawdata.append(self.data[i]) 

     return rawdata

  def encode(self,data):
     codes = []
     for i in range(len(data)):
       if type(data[i]) is list:
         r_code = []
         for word in data[i]:
           #code  = self.word2idx.get(word)
           #if self.word2idx.has_key(word) == False:
           #  print("word missing in dict: ",word)
           #print(word,code)
           #if word.isdigit():
           #  word = '<NUM>'
           r_code.append(self.word2idx.get(word,self.word2idx.get(self.UNK))) 
         codes.append(r_code)
       else:
         codes.append(self.word2idx.get(data[i],self.word2idx.get(self.UNK))) 

     return codes

  def setUNK(self,UNK):
    self.UNK = UNK

  def getCodedData(self):
     codes = []
     for i in range(len(self.data)):
       if type(self.data[i]) is list:
         r_code = []
         for word in self.data[i]:
           #code  = self.word2idx.get(word)
           #print(word,code)
           r_code.append(self.word2idx.get(word,self.word2idx.get(self.UNK))) 
         codes.append(r_code)
       else:
         codes.append(self.word2idx.get(self.data[i],self.word2idx.get(self.UNK))) 

     return codes

  def updateVocab(self,upd_data):
    '''
  	input:
  		upd_data: raw data records
  	output:
  		cntr: Counter object of words
  		word2idx: word to id dictionary
  		idx2word: id to word ditionary
    '''
    for i in range(len(upd_data)):
      for word in upd_data[i]:
        self.cntr[word] += 1
        if self.word2idx.has_key(word) == False:
          self.vocab = self.vocab.append(word) 
          idx = len(self.word2idx) + 1
          self.word2idx[word] = idx
          self.idx2word[idx] = word


if __name__ == "__main__":
  data = "All work and no playing |tiii taking makes jack dull boy. All work and no play makes jack a dull boy. Eighty-seven miles to go, yet.  Onward! _tag_unk1"
  if sys.argv[1] != None:
     data = sys.argv[1]
  #print(stemAndGetWord("discounts"))
  #print(processWordsWithNLTK(data))
  print(processWordsWithNLTK1(data))
  #print(data)
  #data = processRawData("../../data/chat/res5000.txt")
  '''
  for conv in data:
    for sent in conv:
      print(sent[:10])
  '''

