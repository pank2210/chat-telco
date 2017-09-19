
import sys
import os
import pickle

#idir = '../../other/'
idir = 'ner/'
#files = ['data.txt']
files = ['res.csv']

for file in files:
   with open(file,'w') as ofd:
      with open(idir + file,'r') as ifd:
         for line in ifd:
            ofd.write(line.lower())	
