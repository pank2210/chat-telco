
import pandas as pd
import numpy as np
import collections
import pickle
import tensorflow as tf
import data_util as du

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
    '''
    vocab1 = sorted(self.cntr, key=self.cntr.get, reverse=True)[:len(self.cntr)]
    print('#no of time 500 th word from vocab: ',vocab1[500], ': ', self.cntr[vocab1[500]])
    print('#no of time 1000 th word from vocab: ',vocab1[1000], ': ', self.cntr[vocab1[1000]])
    print('#no of time 2000 th word from vocab: ',vocab1[2000], ': ', self.cntr[vocab1[2000]])
    print('#no of time 3000 th word from vocab: ',vocab1[3000], ': ', self.cntr[vocab1[3000]])
    print('#no of time 4000 th word from vocab: ',vocab1[4000], ': ', self.cntr[vocab1[4000]])
    err
    '''

    self.word2idx = {}
    for i,word in enumerate(self.vocab):
       self.word2idx[word] = i
    print('few keys of word2idx :',{k: self.word2idx[k] for k in self.word2idx.keys()[:5]})
  
    self.idx2word = dict((v, k) for k, v in self.word2idx.items()) 
    print('few keys of idx2word :',{k: self.idx2word[k] for k in self.idx2word.keys()[:5]})
   
  def __init__(self,idata,dict_ratio=.3):
     self.data = idata
     self.vocab = []
     self.word2idx = {}
     self.idx2word = {}
     self.cntr = None
     self.UNK = 'IG'
     self.dict_ratio = dict_ratio
     self.createVocab()

  def convData(self,data):
     rawdata = []
     for i in range(len(data)):
       if type(data[i]) is list:
         for word in data[i]:
           rawdata.append(word) 
       else:
         rawdata.append(data[i]) 

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

