import torch
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data_utils
from collections import defaultdict
import sys
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from keras.utils import to_categorical
from model import *
from util import *
import json

# Piazza posts: https://piazza.com/class/j9xdaodf6p1443?cid=2774
batch_size = 32
teacher_forcing  = 1
print_flag = 0

#### Load dev inputs and outputs
dev_input = np.load('../data/train.npy')
dev_output = np.load('../data/train_transcripts.npy')

#### Sort and numerify
dev_input, dev_output, words_dict = sort_arrays(dev_input, dev_output)
filename = 'words_dict.json'
json.dump(words_dict, open(filename, 'w'))

#### Data loader stuff
dataset = WSJ(dev_input[0:20000],dev_output[0:20000])
train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn = custom_collate_fn)

#### Parameters
i2w = {i:w for w,i in words_dict.items()}
embedding_dim = len(words_dict)
hidden_dim = 128
vocab_size = len(words_dict)
target_size = vocab_size
input_dim = 40
print("The target size is ", target_size)
baseline_listener = listener(input_dim, hidden_dim)
baseline_speller = speller(embedding_dim, hidden_dim, vocab_size)
criterion = nn.CrossEntropyLoss(reduce=False)
optimizer = torch.optim.Adam(list(baseline_listener.parameters()) + list( baseline_speller.parameters()), lr=1e-3, weight_decay=1e-5)
baseline_listener.cuda()
baseline_speller.cuda()
#objective = nn.CrossEntropyLoss(ignore_index=0)
objective = nn.CrossEntropyLoss()
objective.cuda()


logfilename = 'log_file_256hidden'

g = open(logfilename, 'w')
g.close()

def train(tf_rate):
 total_loss = 0
 for ctr, t in enumerate(train_loader):
    optimizer.zero_grad()
    a,b,m = t
    a,b,m = Variable(torch.from_numpy(a)), Variable(torch.from_numpy(b), requires_grad=False), Variable(torch.from_numpy(m), requires_grad=False)
    a,b, m = a.cuda(), b.cuda(), m.cuda()
    #print("Shape of b is ", b.shape)
    pred = baseline_listener(a)
    #print("Listener seems ok")
    words = baseline_speller(pred, b.float(), tf_rate)
    #print("Speller seems ok")
    #print("Shape of prediction from speller is ", words.shape)
    pred_y = torch.cat([torch.unsqueeze(each_y,1) for each_y in words],1).view(-1,embedding_dim)
    true_y = torch.cat([torch.unsqueeze(each_y[1:],1) for each_y in b],1).view(-1)
    mask_y = torch.cat([torch.unsqueeze(each_y[1:],1) for each_y in m],1).view(-1)
    #print("Shape of predicted Y is: ", pred_y.shape)
    #true_y = torch.max(b,dim=-1)[1].view(-1)
    #true_y = true_y.detach()
    #print("Shape of groundtruth is ", true_y.shape)
    loss = objective(pred_y, true_y)
    loss = loss * mask_y
    #loss = torch.sum(loss, dim = 1)
    loss = torch.mean(loss, dim = 0)

    total_loss += loss.cpu().data.numpy()
    loss.backward()
    del loss, a,b,m
    optimizer.step()

    if ctr % 15 == 1:
        print("Total loss after ", ctr, " batches is ", total_loss/float(ctr+1) )
        g = open(logfilename,'a')
        g.write(" Total loss after " + str(ctr) + ' batches is ' + str(total_loss/float(ctr+1)) +  '\n')
        g.close()


 return pred_y, true_y, total_loss/float(ctr)


for epoch in range(10):
    print("Running epoch ", epoch)
    tf_rate = 0.9 - (0.9 - 0)*(epoch/10)

    pred_y, true_y, total_loss = train(tf_rate)

    '''
    pred_y = pred_y.transpose(0,1)
    true_y = true_y.transpose(0,1)
    print(" Loss after ", epoch+1, "epochs: ", total_loss)
    sent = ' '
    for k in true_y:
       sent += ' ' + str(i2w[int(k.data.cpu().numpy())])
    print("   Original Sentence is ", sent.split()[0:20])
    print("\n")
    g = open(logfilename,'a')
    g.write("Loss after epoch " + str(epoch) + ': ' + str(total_loss) + '\n')
    g.write(" Original sentence: " + ' '.join(t for t in sent.split()[0:20]) + '\n')
    g.close()

    sent = ' '
    for k in pred_y:
     if k.shape[0] > 1:
          k = k.data.cpu().numpy()
          sent += ' ' + i2w[int(np.argmax(k))]
    print("   Predicted Sentence is ", sent.split()[0:20])
    print('\n')
    g = open(logfilename, 'a')
    g.write(" Predicted sentence: " + ' '.join(t for t in sent.split()[0:20]) + '\n')
    g.close()
    '''

    #torch.save(baseline_listener.state_dict(), './BaseLine_listener_' + str(epoch).zfill(3) + '.pt')
    #torch.save(baseline_speller.state_dict(), './BaseLine_speller_' + str(epoch).zfill(3) + '.pt')

    torch.save(baseline_listener.state_dict(), './BaseLine_listener.pt')
    torch.save(baseline_speller.state_dict(), './BaseLine_speller.pt')




