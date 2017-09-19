from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
import re
import numpy as np

def processPunctuation(data): 
  tokenizer = RegexpTokenizer(r'\w+')
  return tokenizer.tokenize(data)

def removeStopWords(data): 
  stopWords = set(stopwords.words('english'))
  words = word_tokenize(data)
  wordsFiltered = []
 
  for w in words:
    if w not in stopWords:
      wordsFiltered.append(w)
   
  return(wordsFiltered)

def stemAndGetSentence(data):
  sno = nltk.stem.SnowballStemmer('english')
  str = ''

  for w in data:
    #print(w)
    str = str + sno.stem(w.lower()) + ' '
  #print(str)
  
  return str

def processWordsWithNLTK(data):
  data = processPunctuation(data)
  #print(data)
  data = stemAndGetSentence(data)
  #print(data)
  
  return removeStopWords(data)

def processRawData(data_file):
  print("NLP util processing raw text data file[{}]...".format(data_file))
  data = []
  lcnt = 0
  scnt = []
  clen = []
  ccnt = 0
  sent = []
  conv = []
  sent_log = {}
  raw_data_key = 'raw'
  data_key = 'data'
  
  with open(data_file,'rb') as ifd:
    for i,line in enumerate(ifd):
      #if i >= 2000: #early termination for debug
      #  break
      buf = line.strip()
      if re.search('^$',buf): #skip empty line
        continue
      #replace non-printable unicodes. this list will needs to be enhance with every failure
      buf = re.sub(r'[\xb5\xc5\xb3\xc3\xc6\xb6\xb9\xc9\xba\xc1\xc4\xe5\xef\xcf\xc5\xe2\xbc\xbe\xcb\xbd\xc3\xc2\xa1]'," ",buf)
      words = processWordsWithNLTK(buf)
      if len(words) == 0: #skip lines that ended up with nothing after processing
        continue
      #print(words)
      if words[0] == 'docstart':
        #check for conversation or logical grouping of sentences
        if len(conv) > 0:
          data.append(conv)
          #clen[ccnt] = len(conv)
          clen.append(len(conv))
          ccnt += 1
        conv = []
        continue
      sent_log = {}
      sent_log[raw_data_key] = buf
      sent_log[data_key] = words
      conv.append(sent_log)
      #scnt[i] = len(words)
      scnt.append(len(words))
      lcnt += 1
 
  print("NLP util processed [{}] conversations,[{}] lines and [{}] words...".format(ccnt,lcnt,np.sum(scnt)))
  print("NLP util data stats : [{}] avg words/sent and [{}] sent/conversations...".format(np.mean(scnt),np.mean(clen)))
  
  return data

if __name__ == "__main__":
  data = "All work and no playing taking makes jack dull boy. All work and no play makes jack a dull boy. Eighty-seven miles to go, yet.  Onward!"
  #print(data)
  #print(processWordsWithNLTK(data))
  data = processRawData("../../data/chat/res.txt")
  '''
  for conv in data:
    for sent in conv:
      print(sent[:10])
  '''
