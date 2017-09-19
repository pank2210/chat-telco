import os
import pickle
import nlp_util as nu
import numpy as np
import re
import collections
import math

"""
  input - Sent data file and NER tag dictionary 
  process - conv struct return from nlp.processRawData is processed to create config file tagged data file.
		- config file has conv_count|sent class|NLTK process sent|raw sent
                - tagged data file. same as NER training file.
  output - name of created config file and tagged data file.
"""
def chatGenDataTags(data_file,tag_dict_file):
  print("Generating tag file for data using tag dict[{}] for data file [{}]...".format(tag_dict_file,data_file))
  tagged_data_file = data_file + '.tagged'
  data_conf_file = data_file + '.conf'

  with open(tag_dict_file,'rb') as tfd:
    tag_dict = pickle.load(tfd)
   
  print("Loaded [{}] NER tags..".format(len(tag_dict)))
 
  data = processTaggedChat(data_file)

  with open( data_conf_file,'wb') as cfd: 
    with open( tagged_data_file,'wb') as ofd: 
      conv_cnt = 0
      sep = '\t'
      unk_tag = 'UNK1'
      for conv in data:
        conv_cnt += 1
        for sent_log in conv:
          raw_sent = sent_log.get('raw')
          sent = sent_log.get('data')
          sent_tag = sent_log.get('class')
          #ofd.write( '#@#' + raw_sent + '|' + '\n')
          processed_sent = ''
          for word in sent:
            tag = tag_dict.get(word,unk_tag)
            ofd.write(word + sep + tag + '\n') #o/p for NER training/testing
            processed_sent = processed_sent + ' ' + word
          ofd.write('\n')
          #o/p for processing sentence classification
          #cfd.write(str(conv_cnt) + '|' + sent_tag + '|' + processed_sent + '|' + raw_sent + '\n') 
          cfd.write(str(conv_cnt) + '|' + sent_tag + '|' + processed_sent + '|' ) 
          cfd.write(raw_sent) 
          cfd.write('\n') 
        #ofd.write('\n')
   
  return tagged_data_file,data_conf_file


"""
   input   - array
   process - conv array to dict using collection counter and print stats.
   output  - none
"""
def printLabelsStats(myLabels):
   cntr = collections.Counter()
   tot_labels = len(myLabels)
   for i in myLabels:
     cntr[i] += 1
   label_dict = sorted( cntr, key=cntr.get, reverse=True)[:len(cntr)]
   print("Total labels data/Sentences [{}] unique labels[{}]".format(tot_labels,len(label_dict)))
   
   for i in label_dict:
     per = float(cntr[i])/tot_labels*100
     print("Label[%15s] count[%5d] per[%.2f%%]" % (i,cntr[i],per))
     #print("Label[{%15s}] count[{}] per[{}]".format(i,cntr[i],(cntr[i]/tot_labels)))
     #print("label[{}] count[{}] ".format(i,cntr[i],))


"""
   input - data file with raw sentences which has last word as sentence class. 
   output - Array of conversation where each conv is a hash key data as follow
		- {key=raw} => as is original raw sentence
		- {key=data} => NLTK processed sentence
		- {class} => sentence classification
"""
def processTaggedChat(data_file):
  print("Chat util processing raw text data file[{}]...".format(data_file))
  data = []
  lcnt = 0
  scnt = []
  clen = []
  ccnt = 0
  sent = []
  conv = []
  tags = []
  sent_log = {}
  raw_data_key = 'raw'
  data_key = 'data'
  sent_class = 'class'
  conv_key = 'conv_ind'
  
  with open(data_file,'rb') as ifd:
    for i,line in enumerate(ifd):
      #if i >= 100: #early termination for debug
      #  break
      buf = line.strip()
      if re.search('^$',buf): #skip empty line
        continue
      #replace non-printable unicodes. this list will needs to be enhance with every failure
      buf = re.sub(r'[\xb5\xc5\xb3\xc3\xc6\xb6\xb9\xc9\xba\xc1\xc4\xe5\xef\xcf\xc5\xe2\xbc\xbe\xcb\xbd\xc3\xc2\xa1]'," ",buf)
      words = nu.processWordsWithNLTK(buf)
      #words = nu.processWordsWithNLTK1(buf)
      if len(words[:-1]) == 0: #skip lines that ended up with nothing after processing
        continue
      #print(words[0:2])
      if words[0] == 'docstart' or len(words)>3 and ((words[0] == 'custom' and words[1] == 'connect' and words[2] == 'lost') or (words[0] == 'chat' and words[1] == 'session' and words[2].startswith('end')) or (words[0] == 'custom' and words[1] == 'close' and words[2] == 'window') or (words[0] == 'custom' and words[1] == 'close' and words[2] == 'browser')):
        #check for conversation or logical grouping of sentences
        if len(conv) > 0:
          data.append(conv)
          #clen[ccnt] = len(conv)
          clen.append(len(conv))
          ccnt += 1
        conv = []
        continue
      #if len(words[:-1]) <= 4: #skip lines with words count <= 4
      #  continue
      #mm = re.search('(.*?)\s+(_tag_\w+)$',buf) #skip empty line
      tag = None
      #extract tag or class for sentence 
      if words[-1].startswith('_tag_'):
        tag = words[-1]
        #tag = buf.split()[-1]
      else:
        tag = '_tag_unk1'
      #if tag == '_tag_a':
      #  print("tag[{}] buf[{}] lst_word[{}] words[{}]".format(tag,buf,buf.split()[-1],words))
      sent_log = {}
      sent_log[raw_data_key] = buf
      sent_log[data_key] = words[:-1] #commit tag 
      sent_log[sent_class] = tag
      sent_log[conv_key] = ccnt
      tags.append(tag)
      conv.append(sent_log)
      #scnt[i] = len(words)
      scnt.append(len(words))
      lcnt += 1
 
  #Handling last conversation 
  data.append(conv) 
  clen.append(len(conv))
  ccnt += 1
  conv = []

  print("Chat util processed [{}] conversations,[{}] lines and [{}] words...".format(ccnt,lcnt,np.sum(scnt)))
  print("Chat util data info : [{}] avg words/sent and [{}] sent/conversations...".format(np.mean(scnt),np.mean(clen)))
  print("Chat util data info : [{}] min words/sent and [{}] max words/sent...".format(np.min(scnt),np.max(scnt)))
  printLabelsStats(tags)
  
  return math.ceil(np.mean(scnt)),math.ceil(np.mean(clen)),data

"""
   process file to make tag's binary set for each tag
"""
def createBinTagFiles(sfile):
  print("creating binary tagged files...")
  tag_unk1 = '_tag_unk1'
  tags = []
  o_sents = []

  with open(sfile,'rb') as ifd:
    for i,line in enumerate(ifd):
      #if i >= 100: #early termination for debug
      #  break
      buf = line.strip()
      if re.search('^$',buf): #skip empty line
        continue
      #replace non-printable unicodes. this list will needs to be enhance with every failure
      buf = re.sub(r'[\xb5\xc5\xb3\xc3\xc6\xb6\xb9\xc9\xba\xc1\xc4\xe5\xef\xcf\xc5\xe2\xbc\xbe\xcb\xbd\xc3\xc2\xa1]'," ",buf)
      words = nu.processWordsWithNLTK(buf)
      if words[-1].startswith('_tag_'):
        tag = words[-1]
        #tag = buf.split()[-1]
      else:
        tag = tag_unk1
      tags.append(tag)
      o_sents.append(words[:-1]
 
      tfile = sfile + '.' + tag
      fdict[tag] = fdict.get(tag, open(tfile,'a+'))
      
       


if __name__ == "__main__":
  #a,b = chatGenDataTags(data_file="../../data/chat/res5000.txt",tag_dict_file="../../data/chat/tags2.dict")
  #a,b = chatGenDataTags(data_file="../../data/chat/train1",tag_dict_file="../../data/chat/tags2.dict")
  #a,b = chatGenDataTags(data_file="../../data/chat/test1",tag_dict_file="../../data/chat/tags2.dict")
  #print(a,b)
  #data = processTaggedChat("../../data/chat/res15000.txt")
  data = processTaggedChat("../../data/chat/res5000.txt")
