
import os
import pickle
import nlp_util as nu
import numpy as np


def chatGenDataTags(data_file,tag_dict_file):
  print("Generating tag file for data using tag dict[{}] for data file [{}]...".format(tag_dict_file,data_file))
  tagged_data_file = data_file + '.tagged'
  data_conf_file = data_file + '.conf'

  with open(tag_dict_file,'rb') as tfd:
    tag_dict = pickle.load(tfd)
   
  print("Loaded [{}] tags..".format(len(tag_dict)))
 
  data = nu.processRawData(data_file)

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
          cfd.write(str(conv_cnt) + '|' + raw_sent + '\n')
          #ofd.write( '#@#' + raw_sent + '|' + '\n')
          for word in sent:
            tag = tag_dict.get(word,unk_tag)
            ofd.write(word + sep + tag + '\n')
          ofd.write('\n')
        #ofd.write('\n')
   
  return tagged_data_file,data_conf_file

if __name__ == "__main__":
  #a,b = chatGenDataTags(data_file="../../data/chat/res5000.txt",tag_dict_file="../../data/chat/tags.dict")
  a,b = chatGenDataTags(data_file="../../data/chat/train1",tag_dict_file="../../data/chat/tags.dict")
  print(a,b)
  a,b = chatGenDataTags(data_file="../../data/chat/test1",tag_dict_file="../../data/chat/tags.dict")
  print(a,b)
