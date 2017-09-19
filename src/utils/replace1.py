

import re
import os
import sys

file = '../data/cust01.txt'
#file = 'data/cust.txt'
ofile = '../data/res.txt'

with open(ofile,'wb') as ofd:
  with open( file, 'rb') as fd:
    for line in fd:
      res = re.sub(r'\.\.+','',line.strip())
      res = re.sub(r'\.\s+',' ',res)
      res = re.sub(r'\D\.',' ',res)
      res = re.sub(r',',' ',res)
      res = re.sub(r'#',' ',res)
      res = re.sub(r':',' ',res)
      res = re.sub(r'"',' ',res)
      res = re.sub(r'\*',' ',res)
      res = re.sub(r'\(',' ',res)
      res = re.sub(r'\)',' ',res)
      res = re.sub(r'\/',' ',res)
      res = re.sub(r'\?',' ',res)
      res = re.sub(r'\$',' $ ',res)
      #print('res - ',res)
      match = re.match( r'^\s*?$',res,re.M|re.I)
      if match:
         continue
      #if( line.strip() == 'xxxxxx'):
      match = re.search( r'^.*?customer closed chat .*?$',res,re.M|re.I)
      if match:
        #print(line)
        ofd.write('\n')
        ofd.write('-DOCSTART-' + '\n')
        ofd.write('\n')
      else:
        ofd.write(res.lower() + '\n')
