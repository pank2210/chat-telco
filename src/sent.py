
import sys
import pandas as pd
import numpy as np
import collections
import pickle

import tensorflow as tf
import tflearn
import tflearn.data_utils as du

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool, global_avg_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.optimizers import Adam

import utils.data_util as util
import utils.chat_util as cu
import utils.nlp_util as nu

#config class to hold DNN hyper param
class Config:
  def __init__(self,model_name,datadir):
    #dictionary
    self.model_name = model_name
    self.datadir = datadir
    print("Initializing Config for SentClassificationModel[{}]...".format(self.model_name))
    self.vocab = None
    self.label = None
    self.window = 3
    self.vocab_dict_ratio = 0.20
    
    #Model Param
    self.epoch = 15
    self.batch_size = 32
    self.lrate = 0.001
    self.drop_prob = 0.5
    self.w_initializer = 'Xavier'
    self.optimizer = 'adam'
    #self.activation_1 = 'PReLU'
    #self.activation_1 = 'relu'
    #self.activation_1 = 'tanh'
    self.activation_1 = 'LeakyReLU'
    self.validation_set_ratio = 0.1
    self.wv_size = 128
    self.sent_size = 50  #no words from sentence to be used 

#Main class holding model
class SentClassificationModel:
   
  def printDiff(self,data_config_file=None):
    print("Printing difference of predictions for SentClassificationModel[{}]...".format(self.model_name))
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
             f1.write(conf_df.iloc[rec-1,1])
             f1.write('\n')
             rec_change_ind = False
          diff[i] = j
          prbuf = str(i) + sep;
          for pos in range(0,2*self.config.window+1,1):
             prbuf = prbuf + self.vocab.idx2word[self.encodedXtestdata[i][pos]] + sep 
          prbuf = prbuf + self.labels.idx2word[self.encodedYtestdata[i]] + sep 
          prbuf = prbuf + self.labels.idx2word[j] + sep
          prbuf = prbuf + str(self.pred_prob[i]) + '\n'
          f1.write(prbuf)
          
       
    self.accu = float((len(self.pred_code) - len(diff))/len(self.pred_code)) 
    print(" {} words & {} errors Accuracy: {}".format(len(self.pred_code),len(diff),self.accu)) 

  def printSentClassification(self,data_config_file=None):
    print("Printing sentence classification corpus using SentClassificationModel[{}]...".format(self.model_name))
    conf_df = []
    rec = 0
    sent = []
    pred_sent = ''
   
    if data_config_file != None:
      print("printSentClassification Loading config from file[{}]...".format(data_config_file))
      conf_df = pd.read_csv(data_config_file,header=None,delimiter='|') 
     
    p_df = pd.DataFrame(columns=[
               "conv_id",
               "sent",
               "label",
               #"vect_id",
               "pr"])

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
       
        ''' 
        prbuf = str(rec) + sep  
        prbuf = prbuf + self.testX[i][self.config.window] + sep 
        prbuf = prbuf + self.labels.idx2word[self.pred_code[i]] + sep
        prbuf = prbuf + self.vocab.idx2word[self.encodedXtestdata[i][self.config.window]] + sep 
        prbuf = prbuf + str(self.pred_prob[i]) + '\n'
        ''' 
        p_df.loc[p_df.shape[0]] = [ str(rec),
                                    #self.testX[i][self.config.window],
                                    pred_sent.split()[:10],
                                    self.labels.idx2word[self.pred_code[i]],
                                    #self.vocab.idx2word[self.encodedXtestdata[i][self.config.window]],
                                    str(self.pred_prob[i])]
         
      sent.append(pred_sent)
      p_df.to_csv(self.datadir + self.model_name + "_pred_df.csv")
      #print("***********",conf_df.iloc[395,2],sent[395],len(conf_df),len(sent))
      conf_df[4] = sent
      conf_df.to_csv(data_config_file + '.txt',index=None,header=None,sep='|')
   
  def __init__(self,model_name,trainfl,datadir):
    self.model_name = model_name
    self.src_file = 'sentclassi_model_' + self.model_name + '.tfl'
    print("Initializing SentClassificationModel[{}]...".format(self.model_name))
    self.datadir = datadir
    self.train_data_file = datadir + trainfl
    self.config = Config(self.model_name,self.datadir) #create config file
    self.createSentClassificationModel() #create SentClassificationModel
    #self.createSimpleDNN() #create SentSimpleDNN
 
  """ 
    Function does following things
	- reads input file with format of raw sentences which are tagged. Tag is last word of sentence line
        - builds vocab as per config
        - creates X and Y out of data. X are sentence words with 0 padding and Y are calssification labels.
  """ 
  def buildTrainingData(self):
    print("Building train data for SentClassificationModel[{}]...".format(self.model_name))
    #initialize all keys required to browse data
    raw_data_key = 'raw'
    data_key = 'data'
    sent_class = 'class'
    conv_key = 'conv_ind'

    #read training data 
    avg_words,avg_sents,conv = cu.processTaggedChat(self.train_data_file)
    trainX = [] 
    trainY = [] 
  
    for i,cdata in enumerate(conv):
      #if i >= 5:
      #  break
      for sdata in cdata:
        trainX.append(sdata[data_key])
        trainY.append(sdata[sent_class])
        #print("trainX[{}]****labels[{}]".format(sdata[data_key],sdata[sent_class]))
 
    print("Training data of [{}] sentences and [{}] labels loaded for classification...".format(len(trainX),len(trainY))) 
    self.vocab = nu.Vocab(trainX,self.config)  #build X vocab dict & required data
    self.labels = nu.Vocab(trainY,self.config,label=True) #build Y vocab dict & required data
     
    self.labels.setUNK('UNK1')  #Explicitly set label for unknown classification
     
    #Create encoding for training data
    self.encodedXdata = self.vocab.getCodedData()
    self.encodedYdata = self.labels.getCodedData()
     
    print("Coded X {} data: {}".format(len(self.encodedXdata),self.vocab.getData()[:10]))
    print("Coded X code: {}".format(self.encodedXdata[:10]))
    print("Coded Y size {} unique {} data: {}".format(len(self.encodedYdata),len(set(self.encodedYdata)),self.labels.getData()[:10]))
    print("Coded Y code: {}".format(self.encodedYdata[:10]))
   
    #pad sequence with zero's 
    self.encodedXdata = pad_sequences(self.encodedXdata,maxlen=self.config.sent_size,value=0)
    self.no_classes = len(set(self.encodedYdata)) #no of target classes
    self.Y = to_categorical(self.encodedYdata, nb_classes=self.no_classes) #Y as required by tflearn
    
    #release unwanted variables.
    trainX = None
    trainY = None

  def buildSentTestData(self,testfl):
    print("Building sentence test data for SentClassificationModel[{}]...".format(self.model_name))
    #read test data in form of (n-gram) embeding
    self.test_data_file = self.datadir + testfl
    self.testX = util.getTokenizeSent(self.test_data_file,self.config.window,sent_index=4)
    self.encodedXtestdata = self.vocab.encode(self.testX)
    
    print("Coded Test X {} data: {}".format(len(self.encodedXtestdata),self.vocab.convData(self.encodedXtestdata)[:10]))
    print("Coded Test X code: {}".format(self.encodedXtestdata[:10]))
  
  def buildTestData(self,testfl,data_only=False):
    print("Building test data for SentClassificationModel[{}]...".format(self.model_name))
    #read test data in form of (n-gram) embeding
    self.test_data_file = self.datadir + testfl

    #initialize all keys required to browse data
    raw_data_key = 'raw'
    data_key = 'data'
    sent_class = 'class'
    conv_key = 'conv_ind'

    #read training data 
    avg_words,avg_sents,conv = cu.processTaggedChat(self.test_data_file)
    self.testX = [] 
    self.raw_testX = [] 
    self.conv_ind = [] 
    testY = [] 
  
    for i,cdata in enumerate(conv):
      #if i >= 3:
      #  break
      for sdata in cdata:
        self.conv_ind.append(sdata[conv_key])
        self.raw_testX.append(sdata[raw_data_key])
        self.testX.append(sdata[data_key])
        testY.append(sdata[sent_class])
        #print("i[{}]****conv_ind[{}]****trainX[{}]****labels[{}]".format(i,sdata[conv_key],sdata[data_key],sdata[sent_class]))

    print("Test data of[{}] sentences and [{}] labels loaded for classification...".format(len(self.testX),len(testY))) 
    if data_only:
      self.testX = util.getTokenizeDataOnly(self.test_data_file,self.config.window)
      self.encodedXtestdata = self.vocab.encode(self.testX)
    else:
      self.encodedXtestdata = self.vocab.encode(self.testX)
      self.encodedYtestdata = self.labels.encode(testY)
    
    print("Coded Test X {} data: {}".format(len(self.encodedXtestdata),self.vocab.convData(self.encodedXtestdata)[:10]))
    print("Coded Test X code: {}".format(self.encodedXtestdata[:10]))
    if data_only == False:
      print("Coded test Y size {} unique {} data: {}".format(len(self.encodedYtestdata),len(set(self.encodedYtestdata)),self.labels.convData(self.encodedYtestdata)[:10]))
      print("Coded Y code: {}".format(self.encodedYtestdata[:10]))
    
    #pad sequence with zero's 
    self.encodedXtestdata = pad_sequences(self.encodedXtestdata,maxlen=self.config.sent_size,value=0)
  
  def createSimpleDNN(self):
    print("Createing the NERModel[{}]...".format(self.model_name))
    self.buildTrainingData()
    
    # Building convolutional network
    net = input_data(shape=[None, self.config.sent_size], name='input')
    #net = tflearn.input_data([None, self.config.window*2+1])
    net = tflearn.embedding(net, 
               input_dim = self.vocab.dict_size, 
               weights_init = self.config.w_initializer,
               output_dim = self.config.wv_size)
    net = tflearn.fully_connected(net, 500, activation=self.config.activation_1)
    net = tflearn.dropout(net,self.config.drop_prob)
    net = tflearn.fully_connected(net, self.no_classes, activation='softmax')
     
    # With TFLearn estimators
    adam = Adam(learning_rate=self.config.lrate, beta1=0.99)

    net = tflearn.regression(net, 
               #optimizer = self.config.optimizer,
               optimizer = adam,
               learning_rate = self.config.lrate, 
               loss = 'categorical_crossentropy')

    # Define model
    self.model = tflearn.DNN(net)

  def createSentClassificationModel(self):
    print("Createing the SentClassificationModel[{}]...".format(self.model_name))
    self.buildTrainingData()
    #err #error

    # Building convolutional network
    network = input_data(shape=[None, self.config.sent_size], name='input')
    network = tflearn.embedding(network, 
               input_dim = self.vocab.dict_size, 
               weights_init = self.config.w_initializer,
               output_dim = self.config.wv_size)
    """
    branch1 = conv_1d(network, 128, 3, padding='valid', activation=self.config.activation_1, regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation=self.config.activation_1, regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation=self.config.activation_1, regularizer="L2")
    branch4 = conv_1d(network, 128, 7, padding='valid', activation=self.config.activation_1, regularizer="L2")
    #branch5 = conv_1d(network, 128, 9, padding='valid', activation=self.config.activation_1, regularizer="L2")
    """
    branch1 = conv_1d(network, self.config.wv_size, 3, padding='valid', activation=self.config.activation_1, bias=True,weights_init = self.config.w_initializer )
    branch2 = conv_1d(network, self.config.wv_size, 4, padding='valid', activation=self.config.activation_1, bias=True, weights_init = self.config.w_initializer)
    branch3 = conv_1d(network, self.config.wv_size, 5, padding='valid', activation=self.config.activation_1, bias=True, weights_init = self.config.w_initializer)
    branch4 = conv_1d(network, self.config.wv_size, 7, padding='valid', activation=self.config.activation_1, bias=True, weights_init = self.config.w_initializer)

    network = merge([branch1, branch2, branch3, branch4], mode='concat', axis=1)
    #network = merge([branch2, branch3, branch4, branch5], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    #network = global_avg_pool(network)
    network = tflearn.dropout(network,self.config.drop_prob)
    network = fully_connected(network, self.no_classes, activation='softmax')
     
    # With TFLearn estimators
    #adam = Adam(learning_rate=self.config.lrate, beta1=0.99)

    network = regression(network, 
                     optimizer = self.config.optimizer,
                     #optimizer = adam,
                     learning_rate = self.config.lrate, 
                     loss='categorical_crossentropy', name='target')
    # Define model
    self.model = tflearn.DNN(network)

  def loadModel(self,model_file):
    # Restore model from earlier saved file 
    print("Restoring SentClassificationModel[{}] from [{}] file...".format(self.model_name,model_file))
    self.model.load(self.datadir + model_file)

  #Incomplete....
  def reTrainTheModel(self,retrain_data_file):
    # Start training (apply gradient descent algorithm)
    print("Training SentClassificationModel[{}]...".format(self.model_name))
    self.buildTrainingData()
    self.drop_prob = 0.0
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
    print("Training SentClassificationModel[{}]...".format(self.model_name))
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
    print("Processing raw test data file[{}] on SentClassificationModel[{}]...".format(raw_data_file,self.model_name))
    interim_data_file = util.processRawData(self.datadir + raw_data_file)
    print(interim_data_file)
    self.testTheModel(interim_data_file,data_only=True)
   
  def genSentTrainingCorpus(self,test_data_file):
    print("Gesentclassiating sentence training corpus using SentClassificationModel[{}]...".format(self.model_name))
    #Get predictions
    self.buildSentTestData(test_data_file)
    self.pred = self.model.predict(self.encodedXtestdata)
    self.pred_code = np.argmax(self.pred,axis=1)
    self.pred_prob = np.amax(self.pred,axis=1)
    #self.printPrediction(data_only)
    self.printSentClassification(data_config_file=self.datadir + test_data_file)

  """
    Desc - Iterate over predictions and log details where ever prediction is diff from actual.
	Prediction are loged in form accuracy summary and classifier level accuracy on screen.
	A Diff file is created for test data with raw data, its prediction and actual label.
        A editable file is created with original data and predicted label for case where it is 
	wrong.
  """
  def printPrediction(self,data_only=False):
    print("Printing predictions for SentClassificationModel[{}]...".format(self.model_name))
    sep = ','
    rec = 0
    rec_change_flag = False
    rec_diff = 0
    rec_data = []
    err_cnt = dict() #collects stats at each of classifier for accuracy.
    i = 0
    prev_conv_ind = -99 #required to keep track of conv id change to mimic original raw file for docstart.

    to = 20
    digits = len(str(to - 1))
    delete = "\b" * (digits)

    with open(self.datadir + self.model_name + '_test.pred','w') as f1, open(self.datadir + self.model_name + '_edit.pred','w') as f2:
      for i in range(len(self.raw_testX)):
        rec += 1
        err_cnt_key = str(self.labels.idx2word[self.encodedYtestdata[i]])
        err_cnt[err_cnt_key + '0'] = err_cnt.get(err_cnt_key + '0',0) + 1
        #check for sentence end based in n-gram used using window size
        prbuf = '' #self.testX[i][self.config.window] + sep + self.labels.idx2word[self.pred_code[i]] + '\n'
        if data_only == False:
          #logic for checking conversations.
          if prev_conv_ind != self.conv_ind[i]:
            f2.write("\ndocstart\n\n")
          #logic to create data if prediction error
          if self.pred_code[i] != self.encodedYtestdata[i]:
            #print(i,self.labels.idx2word[self.pred_code[i]],self.raw_testX) 
            err_cnt[err_cnt_key + '1'] = err_cnt.get(err_cnt_key + '1',0) + 1
            rec_diff += 1
            #print("{0}{1:{2}}".format(delete, rec_diff, digits))
            #sys.stdout.write('.')
            pybuf =  '\b' * 7 + "[%5d]" % (rec_diff)
            sys.stdout.write(prbuf)
            sys.stdout.flush()      
            self.rec_accu = float((rec-rec_diff))/rec 
            f1.write(self.labels.idx2word[self.pred_code[i]])  
            f1.write(sep)
            f1.write(self.labels.idx2word[self.encodedYtestdata[i]])
            f1.write(sep)
            f1.write(self.raw_testX[i])
            #f1.write(sep)
            #f1.write(" ".join(self.testX[i]))
            f1.write('\n')
            '''
            #Create original data file for editing to corrrect data based on prediction.
            #original data is appended with prediction so that decision can be made by reviewer
            f2.write(self.raw_testX[i])
            f2.write(sep)
            f2.write(self.labels.idx2word[self.pred_code[i]])  
            f2.write('\n')
          else:
            #Just dump write the original data as is.
            f2.write(self.raw_testX[i])
            f2.write(sep)
            f2.write(self.labels.idx2word[self.encodedYtestdata[i]])
            f2.write(sep)
            f2.write(self.labels.idx2word[self.pred_code[i]])  
            f2.write('\n')
            '''
          #Just dump write the original data as is.
          f2.write(self.raw_testX[i])
          f2.write(sep)
          f2.write(self.labels.idx2word[self.encodedYtestdata[i]])
          f2.write(sep)
          f2.write(self.labels.idx2word[self.pred_code[i]])  
          f2.write('\n')

          prev_conv_ind = self.conv_ind[i]
     
     
    if data_only:
      print("Processes [{}] words / [{}] records".format(len(self.raw_testX),rec))
    else:
      print(" {} total size & {} errors Accuracy: {}".format(rec,rec_diff,float(rec-rec_diff)/rec)) 
      myCodedLabels = set(self.encodedYtestdata)
      myLabels = []
      for k in myCodedLabels:
        myLabel = self.labels.idx2word[k]
        err = err_cnt[myLabel + '1']
        tot = err_cnt[myLabel + '0']
        #tot = err_cnt[self.labels.idx2word[myLabel]+'0']
        accu = (tot-err)/float(tot)*100
        print(" Label[%20s] Total[%5d] error[%5d] Accuracy[%.2f%%]" % (myLabel,tot,err,accu))

  def testTheModel(self,test_data_file,data_only=False,data_config_file=None):
    print("Executing test on SentClassificationModel[{}]...".format(self.model_name))
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

if __name__ == "__main__":
  model_name = "intent1"
  #sentclassiModel = SentClassificationModel(model_name,"res15000.txt","../data/chat/")
  #sentclassiModel = SentClassificationModel(model_name,"res5000.txt","../data/chat/")
  #sentclassiModel = SentClassificationModel(model_name,"watson_15k_train.txt","../data/chat/")
  #sentclassiModel = SentClassificationModel(model_name,"res20k.txt","../data/chat/")
  #sentclassiModel = SentClassificationModel(model_name,"res_15k_train_0623.csv","../data/chat/")
  sentclassiModel = SentClassificationModel(model_name,"senti_20k_train.txt","../data/chat/")
   
  sentclassiModel.trainTheModel()
  #sentclassiModel.reTrainTheModel("res5000.txt")
  #sentclassiModel.reTrainTheModel("res15000.txt")
  #sentclassiModel.loadModel('sentclassi_model_' + model_name + '.tfl')
   
  #sentclassiModel.testTheModel("res5000.txt",data_only=False)
  #sentclassiModel.testTheModel("res3000.txt",data_only=False)
  #sentclassiModel.testTheModel("watson_15k_test.txt",data_only=False)
  #sentclassiModel.testTheModel("res_3k_test_0623.csv",data_only=False)
  sentclassiModel.testTheModel("senti_20k_test.txt",data_only=False)
   
