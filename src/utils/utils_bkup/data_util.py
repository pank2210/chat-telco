  
import os
import re
import numpy as np

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

def processRawData(raw_data_file):
  tmpfl = os.path.dirname(raw_data_file) + '/' + os.path.basename(raw_data_file) + '.out'
  print("Processing raw data from [{}] file with interim results written to [{}]...".format(raw_data_file,tmpfl))
 
  with open(tmpfl,'wb') as ofd: 
    ofd.write('\n')
    with open(raw_data_file,'rb') as ifd:
      for line in (ifd):
        res = line.strip()
        tokens = line.split()
        for token in tokens:
          ofd.write(token + '\n')
        ofd.write('\n')
  
  return os.path.basename(tmpfl)

if __name__ == "__main__":
  #data, labels = getTokenizeData(file="tmptmp.txt",window=3)
  #data, labels = getTokenizeData(file="../../data/addr/test1.txt",window=3)
  #print("Token size data [{}] and label [{}]".format(len(data),len(labels)))
  #data, labels = getTokenizeData(file="../../data/addr/fail28rec.exp",window=3)
  #print("Token size data [{}] and label [{}]".format(len(data),len(labels)))
  #data = getTokenizeDataOnly(file="../../data/addr/test2.txt",window=3)
  print("Token size data [{}]".format(len(data)))
  data = getTokenizeDataOnly("../../data/addr/" + processRawData("../../data/addr/fail28rec.txt"),window=3)
  print("Token size data [{}]".format(len(data)))

