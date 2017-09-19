

import re
import os
import sys

file = '../data/addr/tmp1.txt'
ofile = '../data/addr/tmp3.txt'

prv1_match = False
i = 0
ln_cnt = 0
res = ""
pline = ""
ref_line = []

with open(ofile,'wb') as ofd:
  with open( file, 'rb') as fd:
    for cnt,line in enumerate(fd):
      #res = re.sub(r'\$',' $ ',res)
      res = line.strip()
      ref_line.append(res)
      #print('res - ',res)
      prv1_match = re.match( r'^$',res,re.M|re.I)
      if prv1_match:
        prv1_match = True
        i = 0
        #print(res)
      match = re.match( r'^\d+\s+ZA$|^\d+\s+UV$',res,re.M|re.I)
      ptoken = res.split()
      if match and i == 1:
        prv1_match = False
        print(cnt,res)
        res = re.sub(r'ZA$','HN',res)
        res = re.sub(r'UV$','HN',res)
        ofd.write(res + '\n')
      else:
        match = re.match( r'^$',res,re.M|re.I)
        if match:
          ofd.write( 'UNK' + '\t' + 'UNK' + '\n')
        else:
          ofd.write(res + '\n')
      pline = res 
      i += 1
