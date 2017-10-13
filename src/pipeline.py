
import pandas as pd
import numpy as np
import collections
import pickle
import tensorflow as tf
import tflearn
import tflearn.data_utils as du
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

import utils.data_util as du
import utils.chat_util as cu
import ner as ner
import sent as sent

class Pipeline:
  def __init__(self,id,ddir,ner_train_fl,sent_train_fl,test_fl):
    self.id = id
    self.ddir = ddir
    self.ner_train_fl = ner_train_fl
    self.sent_train_fl = sent_train_fl
    self.test_fl = test_fl
    
    self.nerModel = ner.NERModel(self.id + '_ner',self.ner_train_fl,self.ddir)
    #self.sentModel = sent.SentClassificationModel(self.id + '_sent',self.sent_train_fl,self.ddir)
    
    self.nerModel.loadModel('ner_model_' + self.id + '_ner' + '.tfl')
    #self.sentModel.loadModel('sent_model_' + self.id + '_sent' + '.tfl')
  
  def p(self):
    print("Processing test data for pipeline[{}]...".format(self.id))
    #read test data in form of (n-gram) embeding
    self.test_data_file = self.ddir + self.test_fl

    #initialize all keys required to browse data
    raw_data_key = 'raw'
    data_key = 'data'
    conv_key = 'conv_ind'
    avg_words = 0.0
    avg_sents = 0.0 
    conv = []
    ner_df = pd.DataFrame()
    sent_df = pd.DataFrame()

    #read training data 
    avg_words,avg_sents,conv = cu.processTaggedChat(self.test_data_file,tagged=False)
    
    for i,cdata in enumerate(conv):
      for j,sdata in enumerate(cdata):
        rdf = self.nerModel.testTheModelForRawSent(test_data=sdata[raw_data_key],data_only=True,conv_id=sdata[conv_key],sent_id=j)
        ner_df = ner_df.append(rdf,ignore_index = True)
        #rdf = self.sentModel.testTheModelForRawSent(test_data=sdata[raw_data_key],data_only=True,conv_id=sdata[conv_key],sent_id=j)
        #sent_df = sent_df.append(rdf,ignore_index = True)
        
        #print("***********rdf len[{}] col[{}]".format(len(rdf),rdf.columns))
        #print("***********ner_df len[{}] col[{}]".format(len(ner_df),ner_df.columns))
        #print("***********sent_df len[{}] col[{}]".format(len(sent_df),sent_df.columns))
    
    ner_df.to_csv( self.ddir + self.id + "_ner_pipe_df.csv",index=False)
    #sent_df.to_csv( self.ddir + self.id + "_sent_pipe_df.csv",index=False)


if __name__ == "__main__":
  pipe_id = 'chat15k'
  ddir = "../data/chat/"
   
  pipeline = Pipeline(pipe_id,ddir=ddir,
                       ner_train_fl="senti_20k_train.txt.tagged",
                       sent_train_fl="senti_20k_train.txt",
                       test_fl="cust01.txt")
   
  pipeline.p()
