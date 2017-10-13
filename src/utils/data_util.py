  
import os
import re
import numpy as np
import nlp_util as nu

"""
  input  - data file in format <Value>\t<Tag> and window size for NGram
  output - returns sent data converted to NGram model and respective label for every middle word of NGram.
"""
def getTokenizeData(file,window):  
  #window = 1
  dataX = []
  dataY = []
  septag = 'UNK'  #['UNK' for x in range(window)]
  curX = []
  curY = []
  data = []
  
  #file = 'tmptmp.txt'
  
  print("Processing [{}] file for data & label tokens with window size [{}]".format(file,window))
  with open(file,'rb') as fd:
    for count,line in enumerate(fd):
      line = line.strip() 
      match = re.match( r'^$',line,re.M|re.I)
      if match:
        if len(curX) > 0:
          dataX.append(curX)
          dataY.append(curY)
        curX = []
        curY = []
      else:
        token = line.split()
        if len(token) < 2:
          print("Error reading token in file at line[{}]".format(count))
          break
        #Need to process each word using standard stem processing
        curX.append(nu.stemAndGetWord(token[0]))
        curY.append(token[1])
       
      #print("Found X[{}] Y[{}] of token's in file.".format(len(dataX),len(dataY)))
      
  print("loaded {} X {} Y ".format(len(dataX),len(dataY)))
     
  data = []
  label = []
  
  for i,sent in enumerate(dataX):
    #print(sent)
    cur = np.empty([0])
    ll = len(sent)
    start = 0
    end = 0
    for pos,word in enumerate(sent):
      cur = np.empty([0])
      #print(pos,word)
      bef = [septag for j in range(window-pos)] #prepare UNK buf
      cur = np.append(cur,bef) #append n-1 gram buf, if any
      if pos < window:
        start = 0 #range should start form zero
      else:
        start = pos - window #set start n-1 as per window
      cur = np.append(cur,[sent[off_set] for off_set in range(start,pos,1)])
      cur = np.append(cur,sent[pos]) #append  main center word
      #print('***',start,cur)
      if((pos+1+window) <= ll):
        end = pos + 1 + window
      else:
        end = ll
      aft = [sent[off_set] for off_set in range(pos+1,end,1)]
      cur = np.append(cur,aft) #pad n+1 grams
      #cur = np.append(cur,[sent[off_set] for off_set in range(pos+1,end,1)])
      #print('****',pos+1,end,cur)
      aft = [septag for j in range(0,window-len(aft))] #prepare UNK buf
      cur = np.append(cur,aft)
      #print('*****',list(cur),dataY[i][pos])
      label.append(dataY[i][pos])
      data.append(list(cur))
    
  #print(cur)
   
  return data,label

"""
  input  - raw data file and NGram window 
  output - data array of NGram words parsed from raw sentences. Each row is window + Word + window
"""
def getTokenizeDataOnly(file,window):  
  #window = 1
  dataX = []
  septag = 'UNK'  #['UNK' for x in range(window)]
  curX = []
  data = []
  
  #file = 'tmptmp.txt'
  
  print("Processing [{}] file for data tokens with window size [{}]".format(file,window))
  with open(file,'rb') as fd:
    for count,line in enumerate(fd):
      line = line.strip() 
      match = re.match( r'^$',line,re.M|re.I)
      if match:
        if len(curX) > 0:
          dataX.append(curX)
        curX = []
      else:
        token = line
        curX.append(token)
       
      #print("Found X[{}] Y[{}] of token's in file.".format(len(dataX),len(dataY)))
      
  print("loaded {} X ".format(len(dataX)))
   
  return getNGramFormOfData(dataX,window) 

"""
   getNGramFormOfData : Accepts array of sentences where each sentence is array of word.
          Then converts given array of word into N gram array based on size of window passed.

   For example "I have a iphone" will be converted to following form for window=3
       [
	  [UNK1,UNK1,UNK1,I,have,a,iphone]
	  [UNK1,UNK1,I,have,a,iphone,UNK1]
	  [UNK1,I,have,a,iphone,UNK1,UNK1]
	  [I,have,a,iphone,UNK1,UNK1,UNK1]
	]
"""
def getNGramFormOfData( dataX, window):
  septag = 'UNK'  #['UNK' for x in range(window)]
  data = []
  
  for i,sent in enumerate(dataX):
    #print(sent)
    cur = np.empty([0])
    ll = len(sent)
    start = 0
    end = 0
    for pos,word in enumerate(sent):
      cur = np.empty([0])
      #print(pos,word)
      bef = [septag for j in range(window-pos)] #prepare UNK buf
      cur = np.append(cur,bef) #append n-1 gram buf, if any
      if pos < window:
        start = 0 #range should start form zero
      else:
        start = pos - window #set start n-1 as per window
      cur = np.append(cur,[sent[off_set] for off_set in range(start,pos,1)])
      cur = np.append(cur,sent[pos]) #append  main center word
      #print('***',start,cur)
      if((pos+1+window) <= ll):
        end = pos + 1 + window
      else:
        end = ll
      aft = [sent[off_set] for off_set in range(pos+1,end,1)]
      cur = np.append(cur,aft) #pad n+1 grams
      #cur = np.append(cur,[sent[off_set] for off_set in range(pos+1,end,1)])
      aft = [septag for j in range(0,window-len(aft))] #prepare UNK buf
      cur = np.append(cur,aft)
      #print('****',cur)
      data.append(list(cur))
    
  #print(cur)
   
  return data

def getSentLabelData(file,sent_index=2):  
  print("Processing [{}] file for geting data array and labels...".format(file))
  dataX = []
  septag = 'UNK'  #['UNK' for x in range(window)]
  curX = []
  labels = []
  tag_index = 1
  
  #file = 'tmptmp.txt'
  
  with open(file,'rb') as fd:
    for count,line in enumerate(fd):
      line = line.strip() 
      sent = line.split('|')
      words = sent[sent_index].split()
      dataX.append(words)
      labels.append(sent[tag_index])
      
  print("loaded {} X {} Y".format(len(dataX),len(labels)))
  #print(dataX[:5])

  return dataX, labels     

def convSentToNGramData(dataX,window=1): 
  data = []

  for i,sent in enumerate(dataX):
    #print(sent)
    cur = np.empty([0])
    ll = len(sent)
    start = 0
    end = 0
    for pos,word in enumerate(sent):
      cur = np.empty([0])
      #print(pos,word)
      bef = [septag for j in range(window-pos)] #prepare UNK buf
      cur = np.append(cur,bef) #append n-1 gram buf, if any
      if pos < window:
        start = 0 #range should start form zero
      else:
        start = pos - window #set start n-1 as per window
      cur = np.append(cur,[sent[off_set] for off_set in range(start,pos,1)])
      cur = np.append(cur,sent[pos]) #append  main center word
      #print('***',start,cur)
      if((pos+1+window) <= ll):
        end = pos + 1 + window
      else:
        end = ll
      aft = [sent[off_set] for off_set in range(pos+1,end,1)]
      cur = np.append(cur,aft) #pad n+1 grams
      #cur = np.append(cur,[sent[off_set] for off_set in range(pos+1,end,1)])
      aft = [septag for j in range(0,window-len(aft))] #prepare UNK buf
      cur = np.append(cur,aft)
      #print('****',cur)
      data.append(list(cur))
    
  #print(cur)
  #print(data[:5])
   
  return data

def getTokenizeSentenceData(file,window):  
  #window = 1
  dataX = []
  dataY = []
  septag = 'UNK'  #['UNK' for x in range(window)]
  curX = []
  curY = []
  data = []
  
  #file = 'tmptmp.txt'
  
  print("Processing [{}] file for data & label tokens with window size [{}]".format(file,window))
  with open(file,'rb') as fd:
    for count,line in enumerate(fd):
      line = line.strip() 
      match = re.match( r'^$',line,re.M|re.I)
      if match:
        if len(curX) > 0:
          dataX.append(curX)
          dataY.append(curY)
        curX = []
        curY = []
      else:
        token = line.split()
        if len(token) < 2:
          print("Error reading token in file at line[{}]".format(count))
          break
        curX.append(token[0])
        curY.append(token[1])
       
      #print("Found X[{}] Y[{}] of token's in file.".format(len(dataX),len(dataY)))
      
  print("loaded {} X {} Y ".format(len(dataX),len(dataY)))
     
  data = []
  label = []
  
  for i,sent in enumerate(dataX):
    #print(sent)
    cur = np.empty([0])
    ll = len(sent)
    start = 0
    end = 0
    for pos,word in enumerate(sent):
      cur = np.empty([0])
      #print(pos,word)
      bef = [septag for j in range(window-pos)] #prepare UNK buf
      cur = np.append(cur,bef) #append n-1 gram buf, if any
      if pos < window:
        start = 0 #range should start form zero
      else:
        start = pos - window #set start n-1 as per window
      cur = np.append(cur,[sent[off_set] for off_set in range(start,pos,1)])
      cur = np.append(cur,sent[pos]) #append  main center word
      #print('***',start,cur)
      if((pos+1+window) <= ll):
        end = pos + 1 + window
      else:
        end = ll
      aft = [sent[off_set] for off_set in range(pos+1,end,1)]
      cur = np.append(cur,aft) #pad n+1 grams
      #cur = np.append(cur,[sent[off_set] for off_set in range(pos+1,end,1)])
      #print('****',pos+1,end,cur)
      aft = [septag for j in range(0,window-len(aft))] #prepare UNK buf
      cur = np.append(cur,aft)
      #print('*****',list(cur),dataY[i][pos])
      label.append(dataY[i][pos])
      data.append(list(cur))
    
  #print(cur)
   
  return data,label

def removeJunkChar(buf):
  #replace non-printable unicodes. this list will needs to be enhance with every failure
  buf = re.sub(r'[\xb5\xc5\xb3\xc3\xc6\xb6\xb9\xc9\xba\xc1\xc4\xe5\xef\xcf\xc5\xe2\xbc\xbe\xcb\xbd\xc3\xc2\xa1]'," ",buf)

  return buf

"""
   getSentFromRawChatPara : Accepts text para and returns array of sentence. 
"""
def getSentFromRawChatPara(para):
  
  return para.splitlines()

"""
   getSentArrayFromRawChatPara : Accepts text para and returns array of word array. Each word array is sentence.
       It acts warpper for NKTK processing of raw text.
"""
def getSentArrayFromRawChatPara(para):
  data = []
  
  sents = para.splitlines()
   
  for sent in sents:
    data.append(getWordArrayFromRawChatLine(sent))
   
  #print("raw para has [{}] lines...".format(len(data)))
  return data   

"""
   getWordArrayFromRawChatLine : Accpets raw text chat line, remove junk char, apply NLTK based custom steming 
       and returns array of stem words.
"""
def getWordArrayFromRawChatLine(line):
  data = []
   
  #res = line.strip()
  res = removeJunkChar(line)
  tokens = nu.processWordsWithNLTK1(res)
  #tokens = res.split()
   
  for token in tokens:
    data.append(token)
   
  #print("raw line has [{}] words...".format(len(data)))
  
  return data

def processRawData(raw_data_file):
  tmpfl = os.path.dirname(raw_data_file) + '/' + os.path.basename(raw_data_file) + '.out'
  print("Processing raw data from [{}] file with interim results written to [{}]...".format(raw_data_file,tmpfl))
 
  with open(tmpfl,'wb') as ofd: 
    ofd.write('\n')
    with open(raw_data_file,'rb') as ifd:
      for line in (ifd):
        res = line.strip()
        '''
        res = removeJunkChar(res)
        #tokens = nu.processWordsWithNLTK(res)
        tokens = nu.processPunctuation(res)
        #tokens = res.split()
        '''
        tokens = getWordArrayFromRawChatLine(res)
        for token in tokens:
          ofd.write(nu.stemAndGetWord(token) + '\n')
        ofd.write('\n')
  
  return os.path.basename(tmpfl)

'''
   printCollection - utility takes collectiona nd print key stats around it
'''
def printCollection(cntr,type=''):
    if len(cntr) < 5:
      return
    tmp = sorted(cntr, key=cntr.get, reverse=True)[:len(cntr)]
    print("---------Details of {} collection-------------".format(type))
    print("  size - {}".format(len(cntr)))
    print("---------Top 4 words--------------------------")
    for tmp_cnt in range(1,5):
      print('#no of time ',tmp_cnt,' content ',tmp[tmp_cnt], ' : ', cntr[tmp[tmp_cnt]])
    print("---------Intermediate word count--------------")
    for tmp_cnt in range(1,10,2):
      ind_word = len(cntr)/10*tmp_cnt
      print('#no of time ',ind_word,' content ',tmp[ind_word], ' : ', cntr[tmp[ind_word]])

if __name__ == "__main__":
  #data, labels = getTokenizeData(file="tmptmp.txt",window=3)
  #data, labels = getTokenizeData(file="../../data/addr/test1.txt",window=3)
  #print("Token size data [{}] and label [{}]".format(len(data),len(labels)))
  #data, labels = getTokenizeData(file="../../data/addr/fail28rec.exp",window=3)
  #print("Token size data [{}] and label [{}]".format(len(data),len(labels)))
  #data = getTokenizeDataOnly(file="../../data/addr/test2.txt",window=3)
  #print("Token size data [{}]".format(len(data)))
  #data = getTokenizeDataOnly("../../data/addr/" + processRawData("../../data/addr/fail28rec.txt"),window=3)
  #print("Token size data [{}]".format(len(data)))
  #data, labels = getSentLabelData("../../data/chat/" + "res5000.txt.conf",sent_index=2)
  #print("Token size data [{}] labels [{}]".format(len(data),len(labels)))
  sample = "I have iphone.\n But offlate it is not working as expected.\n May be something wrong with it."
  _ = getSentArrayFromRawChatPara(sample)

