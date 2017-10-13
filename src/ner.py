
import pandas as pd
import numpy as np
import collections
import pickle
import tensorflow as tf
import tflearn
import tflearn.data_utils as du
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

import utils.data_util as util

#config class to hold DNN hyper param
class Config:
  def __init__(self,model_name,datadir):
    #dictionary
    self.model_name = model_name
    self.datadir = datadir
    print("Initializing Config for NERModel[{}]...".format(self.model_name))
    self.vocab = None
    self.label = None
    self.window = 3
    self.dict_ratio = 0.2
    
    #Model Param
    self.epoch = 8
    self.batch_size = 16
    self.lrate = 0.001
    self.drop_prob = 0.5
    self.w_initializer = 'Xavier'
    self.optimizer = 'adam'
    self.activation_1 = 'LeakyReLU'
    self.validation_set_ratio = 0.2
    self.wv_size = 128

#Vocabulary class
class Vocab:

  def createVocab(self):
    '''
  	input:
  		idata: raw data records
  	output:
  		cntr: Counter object of words
  		word2idx: word to id dictionary
  		idx2word: id to word ditionary
    '''
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

    #print Vocab stats
    util.printCollection(self.cntr,'Vocab')
    
    #create vocab csv
    v_df = pd.DataFrame(columns=["word","occurence"])
    for i,v in enumerate(self.vocab):
       v_df.loc[i] = [v,self.cntr[v]]
    v_df.to_csv(self.config.datadir + self.config.model_name + "_" + self.id + "_vocab.csv")

    self.word2idx = {}
    for i,word in enumerate(self.vocab):
       self.word2idx[word] = i
    print('few keys of word2idx :',{k: self.word2idx[k] for k in self.word2idx.keys()[:5]})
  
    self.idx2word = dict((v, k) for k, v in self.word2idx.items()) 
    print('few keys of idx2word :',{k: self.idx2word[k] for k in self.idx2word.keys()[:5]})
   
  def __init__(self,id,idata,config):
     self.id = id
     self.data = idata
     self.config = config #get is from caller. for access to entire config
     self.vocab = []
     self.word2idx = {}
     self.idx2word = {}
     self.cntr = None
     self.UNK = 'IG'
     self.dict_ratio = self.config.dict_ratio
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

  def encode(self,data,msg=''):
     codes = []
     not_in_dict = collections.Counter()
     in_dict = collections.Counter()
     for i in range(len(data)):
       if type(data[i]) is list:
         r_code = []
         for word in data[i]:
           #code  = self.word2idx.get(word)
           if self.word2idx.has_key(word) == False:
             #print("word missing in dict: ",word)
             not_in_dict[word] += 1
           else:
             in_dict[word] += 1
           #print(word,code)
           #if word.isdigit():
           #  word = '<NUM>'
           r_code.append(self.word2idx.get(word,self.word2idx.get(self.UNK))) 
         codes.append(r_code)
       else:
         codes.append(self.word2idx.get(data[i],self.word2idx.get(self.UNK))) 
     #print missing and present content stats
     #util.printCollection(not_in_dict,msg + "__" + "not_in_dict")
     #util.printCollection(in_dict,msg + "__" + "in_dict")
     
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


#Main class holding model
class NERModel:

  def processResults(self):
    self.pred_code = np.argmax(self.pred,axis=1)
    self.pred_prob = np.amax(self.pred,axis=1)
    #correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    self.printDiff()
    self.printPrediction()
   
  def printPrediction(self,data_only=False):
    print("Printing predictions for NERModel[{}]...".format(self.model_name))
    #sep = '\t'
    sep = ','
    rec = 0
    rec_change_flag = False
    rec_diff = 0
    rec_data = []
    i = 0
    
    p_df = pd.DataFrame(columns=[
               "sent_id",
               "word",
               "label",
               "vect_id",
               "pr"])

    with open(self.datadir + self.model_name + '_test.pred','w') as f1:
      prbuf = "sent_id" + sep + "word"  + sep + "tag"  + sep + "vect_id" + sep + "pr" + '\n'
      f1.write(prbuf)
      for i in range(len(self.testX)):
        #check for sentence end based in n-gram used using window size
        new_rec_ind = ''
        for j in range(self.config.window):
          new_rec_ind = new_rec_ind + self.testX[i][j]
        if new_rec_ind == ('UNK'*self.config.window):
          rec_change_flag = False
          rec_data = []
          #f1.write('\n')
          rec += 1
          #print(rec,self.testX[i])
        #print(i,self.testX[i]) 
        #prbuf = self.testX[i][self.config.window] + sep + self.labels.idx2word[self.pred_code[i]] + '\n'
        prbuf = str(rec) + sep  
        prbuf = prbuf + self.testX[i][self.config.window] + sep 
        prbuf = prbuf + self.labels.idx2word[self.pred_code[i]] + sep
        prbuf = prbuf + self.vocab.idx2word[self.encodedXtestdata[i][self.config.window]] + sep 
        prbuf = prbuf + str(self.pred_prob[i]) + '\n'

        p_df.loc[p_df.shape[0]] = [ str(rec),
                                    self.testX[i][self.config.window],
                                    self.labels.idx2word[self.pred_code[i]],
                                    self.vocab.idx2word[self.encodedXtestdata[i][self.config.window]],
                                    str(self.pred_prob[i])]
         
        rec_data.append(self.testX[i][self.config.window])
        f1.write(prbuf)
        if data_only == False:
          if self.pred_code[i] != self.encodedYtestdata[i] and rec_change_flag == False:
            #print(i,self.testX[i],self.labels.idx2word[self.pred_code[i]],rec_data) 
            rec_change_flag = True
            rec_diff += 1
   
    #write pandas DF to CSV file. 
    p_df.to_csv(self.datadir + self.model_name + '_pred_df.csv')
    if data_only:
      print("Processes [{}] words / [{}] records".format(len(self.testX),rec))
    else:
      self.rec_accu = (rec-rec_diff)/float(rec) 
      print(" {} total conv & {} errors Accuracy: {}".format(rec,rec_diff,self.rec_accu)) 

  def printDiff(self,data_config_file=None):
    print("Printing difference of predictions for NERModel[{}]...".format(self.model_name))
    sep = '|'
    diff = {}
    conf_df = []
    rec = 0
    rec_change_ind = False
   
    if data_config_file != None:
      print("printDiff Loading config from file[{}]...".format(data_config_file))
      conf_df = pd.read_csv(data_config_file,header=None,delimiter='|') 

    with open(self.datadir + self.model_name + '_diff.txt','w') as f1:
      prbuf = 'rec' + sep;
      for pos in range(0,2*self.config.window+1,1):
        prbuf = prbuf + 'n-' + str(pos) + sep
      prbuf = prbuf + 'y' + sep 
      prbuf = prbuf + 'yhat' + '\n'
          
      for i,j in enumerate(self.pred_code):
        new_rec_ind = ''
        for ci in range(self.config.window):
          new_rec_ind = new_rec_ind + self.testX[i][ci]
        if new_rec_ind == ('UNK'*self.config.window):
          rec += 1
          rec_change_ind = True
        if self.labels.idx2word[self.encodedYtestdata[i]] != self.labels.idx2word[j]:
          if rec_change_ind and len(conf_df) > 0:
             #f1.write(conf_df.iloc[rec,1])
             #f1.write('\n')
             rec_change_ind = False
          diff[i] = j
          prbuf = str(i) + sep;
          for pos in range(0,2*self.config.window+1,1):
             prbuf = prbuf + self.vocab.idx2word[self.encodedXtestdata[i][pos]] + sep 
          prbuf = prbuf + self.labels.idx2word[self.encodedYtestdata[i]] + sep 
          prbuf = prbuf + self.labels.idx2word[j] + sep
          prbuf = prbuf + str(self.pred_prob[i]) + '\n'
          f1.write(prbuf)
       
    self.accu = (len(self.pred_code) - len(diff))/float(len(self.pred_code)) 
    print(" {} words & {} errors Accuracy: {}".format(len(self.pred_code),len(diff),self.accu)) 

  def printSentClassification(self,data_config_file=None):
    print("Printing sentence classification corpus using NERModel[{}]...".format(self.model_name))
    conf_df = []
    rec = 0
    sent = []
    pred_sent = ''
   
    if data_config_file != None:
      print("printSentClassification Loading config from file[{}]...".format(data_config_file))
      conf_df = pd.read_csv(data_config_file,header=None,delimiter='|') 
     
    with open(self.datadir + self.model_name + '_sent.txt','w') as f1:
      for i,j in enumerate(self.pred_code):
        new_rec_ind = ''
        for ci in range(self.config.window):
          new_rec_ind = new_rec_ind + self.testX[i][ci]
        if i>0 and new_rec_ind == ('UNK'*self.config.window):
          sent.append(pred_sent)
          #print(rec,pred_sent,conf_df.iloc[rec,2])
          pred_sent = ''
          rec += 1
          f1.write('\n')
        prbuf = ''
        if self.labels.idx2word[j]  == 'UNK1':
          #prbuf = prbuf + self.vocab.idx2word[self.encodedXtestdata[i][self.config.window]]
          prbuf = prbuf + self.testX[i][self.config.window]
          pred_sent = pred_sent + self.testX[i][self.config.window] + ' '
        else:
          prbuf = prbuf + self.labels.idx2word[j]
          pred_sent = pred_sent + self.labels.idx2word[j] + ' '
        prbuf = prbuf + '\n'
        f1.write(prbuf)
        #if rec>5:
        #  err
        #print(pred_sent)
      sent.append(pred_sent)
      #print("***********",conf_df.iloc[395,2],sent[395],len(conf_df),len(sent))
      conf_df[4] = sent
      conf_df.to_csv(data_config_file + '.txt',index=None,header=None,sep='|')
   
  def __init__(self,model_name,trainfl,datadir):
    self.model_name = model_name
    self.src_file = 'ner_model_' + self.model_name + '.tfl'
    print("Initializing NERModel[{}]...".format(self.model_name))
    self.datadir = datadir
    self.train_data_file = datadir + trainfl
    self.config = Config(self.model_name,self.datadir) #create config file
    self.createNERModel() #create NERModel
  
  def buildTrainingData(self):
    print("Building train data for NERModel[{}]...".format(self.model_name))
    #read training data in form of (n-gram) embeding
    #self.train_data_file = train_data_file
    trainX, trainY = util.getTokenizeData(self.train_data_file,self.config.window) 
    self.vocab = Vocab("trainX",trainX,self.config) #build X vocab dict & required data
    self.labels = Vocab("trainY",trainY,self.config) #build Y vocab dict & required data
     
    self.labels.setUNK('UNK1')  #Explicitly set label for unknown classification
     
    #Create encoding for training data
    self.encodedXdata = self.vocab.getCodedData()
    self.encodedYdata = self.labels.getCodedData()
     
    print("Coded X {} data: {}".format(len(self.encodedXdata),self.vocab.getData()[:10]))
    print("Coded X code: {}".format(self.encodedXdata[:10]))
    print("Coded Y size {} unique {} data: {}".format(len(self.encodedYdata),len(set(self.encodedYdata)),self.labels.getData()[:10]))
    print("Coded Y code: {}".format(self.encodedYdata[:10]))
    
    self.no_classes = len(set(self.encodedYdata)) #no of target classes
    self.Y = to_categorical(self.encodedYdata, nb_classes=self.no_classes) #Y as required by tflearn
    
    #release unwanted variables.
    trainX = None
    trainY = None

  def buildSentTestData(self,testfl):
    print("Building sentence test data for NERModel[{}]...".format(self.model_name))
    #read test data in form of (n-gram) embeding
    self.test_data_file = self.datadir + testfl
    self.testX = util.getTokenizeSentenceData(self.test_data_file,self.config.window)
    self.encodedXtestdata = self.vocab.encode(self.testX,'testX')
    
    print("Coded Test X {} data: {}".format(len(self.encodedXtestdata),self.vocab.convData(self.encodedXtestdata)[:10]))
    print("Coded Test X code: {}".format(self.encodedXtestdata[:10]))
 
  def buildTestDataFromRawSent(self,sent):
    res = []
    
    #print("Building test data array and encoding for NERModel[{}]...".format(self.model_name))
    words = util.getWordArrayFromRawChatLine(sent)
    res.append(words) #prepare single sent structure as required by getNGramFromOfData
    self.testX = util.getNGramFormOfData(res,self.config.window)
    self.encodedXtestdata = self.vocab.encode(self.testX,'testX')
   
    """ 
    print("Coded Test X {} data: {}".format(len(self.encodedXtestdata),self.vocab.convData(self.encodedXtestdata[:10])))
    print("Coded Test X code: {}".format(self.encodedXtestdata[:10]))
    """ 
    
  def buildTestData(self,testfl,data_only=False):
    print("Building test data for NERModel[{}]...".format(self.model_name))
    #read test data in form of (n-gram) embeding
    self.test_data_file = self.datadir + testfl
    if data_only:
      self.testX = util.getTokenizeDataOnly(self.test_data_file,self.config.window)
      self.encodedXtestdata = self.vocab.encode(self.testX,'testX')
    else:
      self.testX, testY = util.getTokenizeData(self.test_data_file,self.config.window)
      self.encodedXtestdata = self.vocab.encode(self.testX,'testX')
      self.encodedYtestdata = self.labels.encode(testY,'testY')
    
    print("Coded Test X {} data: {}".format(len(self.encodedXtestdata),self.vocab.convData(self.encodedXtestdata)[:10]))
    print("Coded Test X code: {}".format(self.encodedXtestdata[:10]))
    if data_only == False:
      print("Coded test Y size {} unique {} data: {}".format(len(self.encodedYtestdata),len(set(self.encodedYtestdata)),self.labels.convData(self.encodedYtestdata)[:10]))
      print("Coded Y code: {}".format(self.encodedYtestdata[:10]))
  
  def createNERModel(self):
    print("Createing the NERModel[{}]...".format(self.model_name))
    self.buildTrainingData()
    
    net = tflearn.input_data([None, self.config.window*2+1])
    net = tflearn.embedding(net, 
               input_dim = self.vocab.dict_size, 
               weights_init = self.config.w_initializer,
               output_dim = self.config.wv_size)
    net = tflearn.fully_connected(net, 500, activation='tanh')
    net = tflearn.dropout(net,self.config.drop_prob)
    net = tflearn.fully_connected(net, self.no_classes, activation='softmax')
    net = tflearn.regression(net, 
               optimizer = self.config.optimizer,
               learning_rate = self.config.lrate, 
               loss = 'categorical_crossentropy')

    # Define model
    self.model = tflearn.DNN(net)

  def loadModel(self,model_file):
    # Restore model from earlier saved file 
    print("Restoring NERModel[{}] from [{}] file...".format(self.model_name,model_file))
    self.model.load(self.datadir + model_file)

  #Incomplete....
  def reTrainTheModel(self,retrain_data_file):
    # Start training (apply gradient descent algorithm)
    print("Training NERModel[{}]...".format(self.model_name))
    self.buildTrainingData()
    self.model.fit( self.encodedXdata, self.Y, 
         n_epoch = self.config.epoch, 
         validation_set = self.config.validation_set_ratio, 
         batch_size = self.config.batch_size, 
         show_metric = True)
    print("Saving the trained model...")
    self.model.save(self.datadir + self.src_file)
    print("Model saved in [{}] file.".format(self.src_file))

  def trainTheModel(self):
    # Start training (apply gradient descent algorithm)
    print("Training NERModel[{}]...".format(self.model_name))
    self.buildTrainingData()
    self.model.fit( self.encodedXdata, self.Y, 
         n_epoch = self.config.epoch, 
         validation_set = self.config.validation_set_ratio, 
         batch_size = self.config.batch_size, 
         show_metric = True)
    print("Saving the trained model...")
    self.model.save(self.datadir + self.src_file)
    print("Model saved in [{}] file.".format(self.src_file))

  #process data file with raw record data with every line in file as one sentence or record.
  def processRawDataRec(self,raw_data_file):
    print("Processing raw test data file[{}] on NERModel[{}]...".format(raw_data_file,self.model_name))
    interim_data_file = util.processRawData(self.datadir + raw_data_file)
    print(interim_data_file)
    self.testTheModel(interim_data_file,data_only=True)
   
  def genSentTrainingCorpus(self,test_data_file):
    print("Generating sentence training corpus using NERModel[{}]...".format(self.model_name))
    #Get predictions
    self.buildSentTestData(test_data_file)
    self.pred = self.model.predict(self.encodedXtestdata)
    self.pred_code = np.argmax(self.pred,axis=1)
    self.pred_prob = np.amax(self.pred,axis=1)
    #self.printPrediction(data_only)
    self.printSentClassification(data_config_file=self.datadir + test_data_file)
  
  def testTheModel(self,test_data_file,data_only=False,data_config_file=None):
    print("Executing test on NERModel[{}]...".format(self.model_name))
    # Predict surviving chances (class 1 results)
    if data_only:
      self.buildTestData(test_data_file,data_only)
    else: 
      self.buildTestData(test_data_file)

    #Get predictions
    self.pred = self.model.predict(self.encodedXtestdata)
    self.pred_code = np.argmax(self.pred,axis=1)
    self.pred_prob = np.amax(self.pred,axis=1)
    self.printPrediction(data_only)
    #self.printSentClassification(data_config_file=self.datadir + data_config_file)
    if data_only == False:
      if data_config_file == None:
        self.printDiff()
      else:
        self.printDiff(data_config_file=self.datadir + data_config_file)

  def testTheModelForRawSent(self,test_data,data_only=False,conv_id=0,sent_id=0):
    #print("Executing test on NERModel[{}] for conv_id[{}] sent_id[{}]...".format(self.model_name,conv_id,sent_id))
    # Predict surviving chances (class 1 results)
    self.buildTestDataFromRawSent(test_data)

    #Get predictions
    self.pred = self.model.predict(self.encodedXtestdata)
    self.pred_code = np.argmax(self.pred,axis=1)
    self.pred_prob = np.amax(self.pred,axis=1)
    
    return self.getPredAsDFForSent(conv_id=conv_id,sent_id=sent_id,data_only=True)
   
  def getPredAsDFForSent(self,conv_id=0,sent_id=0,data_only=False):
    #print("Generating predictions for NERModel[{}]...".format(self.model_name))
    sep = ','
    rec = 0
    rec_change_flag = False
    rec_diff = 0
    rec_data = []
    i = 0
    
    p_df = pd.DataFrame(columns=[
               "conv_id",
               "sent_id",
               "word",
               "label",
               "vect_id",
               "pr"])

    for i in range(len(self.testX)):
      #check for sentence end based in n-gram used using window size
      new_rec_ind = ''
      for j in range(self.config.window):
        new_rec_ind = new_rec_ind + self.testX[i][j]
      if new_rec_ind == ('UNK'*self.config.window):
        rec_change_flag = False
        rec_data = []
        rec += 1
       
      p_df.loc[p_df.shape[0]] = [ conv_id,
                                  sent_id,
                                  self.testX[i][self.config.window],
                                  self.labels.idx2word[self.pred_code[i]],
                                  self.vocab.idx2word[self.encodedXtestdata[i][self.config.window]],
                                  str(self.pred_prob[i])]
       
      rec_data.append(self.testX[i][self.config.window])
     
    #write pandas DF to CSV file. 
    p_df.to_csv(self.datadir + self.model_name + '_pred_df.csv',index=False)
    #print("Processes [{}] words / [{}] records".format(len(self.testX),rec))
    
    return p_df

if __name__ == "__main__":
  model_name = 'chat15k_ner'
  nerModel = NERModel(model_name,"senti_20k_train.txt.tagged","../data/chat/")
  #nerModel.trainTheModel()
  nerModel.loadModel('ner_model_' + model_name + '.tfl')
  
  #nerModel.genSentTrainingCorpus(test_data_file="res5000.txt.conf")
  #nerModel.testTheModel("senti_20k_test.txt.tagged",data_config_file="senti_20k_test.txt.conf")
  #nerModel.testTheModel("test1.txt",data_only=False)
  #nerModel.testTheModel("test2.txt",data_only=True)
  #nerModel.testTheModel("verizon_1.txt",data_only=True)
  #nerModel.processRawDataRec("verizon_1.txt")
  #nerModel.testTheModel("fail28rec.exp",data_only=False)
  nerModel.testTheModelForRawSent(test_data="I am afraid my phone is not working. Can you check my account if it is active?",data_only=True,conv_id=1,sent_id=2)
   
