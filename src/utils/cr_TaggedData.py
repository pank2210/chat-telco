
import os
import sys
import pickle
import re

idir = 'data/'
odir = 'ner/data/ner/'
cdir = 'data/'
#files = ['train','dev','test']
files = ['test']

with open(cdir + 'tagdict.pkl','rb') as f:
   tags_dict = pickle.load(f)

with open(cdir + 'classifiers.pkl','rb') as f:
   tagnames = pickle.load(f)
   num_to_tag = dict(enumerate(tagnames))
   tag_to_num = {v:k for k,v in num_to_tag.iteritems()}

for file in files:
   print('processing [',idir + file,']')
   with open(file + '.csv','w') as tfd:
     tfd.write('word' + ',' + 'y' + '\n')
     with open(odir + file,'w') as ofd:
       ofd.write('-DOCSTART-' + '\t' + 'UNKNOWN' + '\n')
       ofd.write('\n');
       i = 0
       with open(idir + file) as fd:
         for line in fd:
            match = re.search('^-DOCSTART-$',line.strip())
            if match:
               ofd.write(line.strip() + '\t' + 'UNKNOWN' + '\n')
               i = i + 1
               continue
            words = line.strip().split()
            #print(words)
            wl = len(words)
            idx = 0
            while idx < wl:
               skey = words[idx]
               if(idx < wl-1):
                  if tags_dict.has_key(skey + ' ' + words[idx+1]):
                     skey = skey + ' ' + words[idx+1]
                     idx = idx + 1
               #set tag=QUANTITY for all numeric data
               #print(' ',skey)
               #match = re.search('^\d+\.?\d+$',skey)
               if re.search('^\d+\.?\d+$',skey) or re.search('^\d+$',skey): 
                 tmp = tags_dict.get(words[idx-1],'NA')
                 #print(tmp,' ',skey)
                 #choice = raw_input()
		 #check if previous tag was BRAND, if yes then change tag to MODEL
                 if tmp == 'BRAND':
		   #print(skey,' set to MODEL')
                   tag = 'MODEL'
		 #check if previous tag any valid tag except unknown, then change tag to QUANTITY
                 #elif tmp in ['VALUE','REF','ATTR','ACTCHG','ACTBUY','ACTVAL','CUST1']:
                 elif tmp  == 'VALUE':
                   tag = 'QUANTITY'
                 else:
                   tag = 'UNKNOWN'
               elif re.search('^\d+gb$',skey): 
                   tag = 'MEM'
               else:
                 tag = tags_dict.get(skey,'UNKNOWN')
               ofd.write(skey.lower() + '\t' + tag + '\n')
               tfd.write(skey.lower() + ',' + tag + '\n')
               idx = idx + 1
            ofd.write('\n');
         #ofd.write('UUNKKK' + '\t' + 'UNKNOWN' + '\n')
         #ofd.write('UUUNKKK' + '\t' + 'UNKNOWN' + '\n')
