#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict


char_ids = defaultdict(lambda: len(char_ids))
char_ids["<unk>"] = 0
char_ids["<s>"] = 1
char_ids['</s>'] = 2

################### TRAIN #######################################

f = open('../data/train/text')
g = open('../text_characters.train','w')

for line in f:
    line = line.split('\n')[0]
    fname = line.split()[0]
    g.write(fname + ' ')
    content = ' '.join(k for k in line.split()[1:])
    content_char = [char_ids[k] for k in content]
    print content_char[0:5], content[0:5]
    g.write(' '.join(str(k) for k in content_char[2:]) + '\n')

g.close()
f.close()

h = open('train_cids.txt','w')
for v in char_ids:
    h.write(str(v) + ' ' + str(char_ids[v]) + '\n')
h.close()   

i2c = {i:w for w,i in char_ids.iteritems()}

h = open('train_i2cs.txt','w')
for v in i2c:
    h.write(str(v) + ' ' + str(char_ids[v]) + '\n')
h.close()



########################### TEST ###############################

f = open('../data/test/text')
g = open('../text_characters.test','w')

for line in f:
    line = line.split('\n')[0]
    fname = line.split()[0]
    g.write(fname + ' ')
    content = ' '.join(k for k in line.split()[1:])
    content_char = [char_ids[k] for k in content]
    print content_char[0:5], content[0:5]
    g.write(' '.join(str(k) for k in content_char[2:]) + '\n')

g.close()
f.close()

h = open('test_cids.txt','w')
for v in char_ids:
    h.write(str(v) + ' ' + str(char_ids[v]) + '\n')
h.close()

i2c = {i:w for w,i in char_ids.iteritems()}

h = open('test_i2cs.txt','w')
for v in i2c:
    h.write(str(v) + ' ' + str(char_ids[v]) + '\n')
h.close()


