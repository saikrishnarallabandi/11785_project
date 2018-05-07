import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
#from decoder import GreedyDecoder
import numpy as np
#import Levenshtein as Lev
import random
#torch.backends.cudnn.enabled=False

A = np.load('../data/dev.npy')
B = np.load('../data/dev_phonemes.npy')

utterances = []
phonemes = []
labels = []
for i in range(len(A)):
   utterances.append(A[i])
   phonemes.append(B[i])
   for a in B[i]:
      labels.append(a)

utterances = np.array(utterances)
phonemes = np.array(phonemes)


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

import torch.nn.functional as F

# BiRNN Model (Many-to-One)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.inp = nn.Linear(input_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.1)
        self.fc = SequenceWise(nn.Linear(hidden_size*2, num_classes))  # 2 for bidirection 
        self.init_params()

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
              weight.data.uniform_(-stdv, stdv)

    def forward(self, x):

        # Send through  a linear layer
        x = self.inp(x)

        # Forward propagate RNN
        out, _ = self.lstm(x, None)

        # Decode hidden state of last time step
        out = self.fc(out)
        return out

        #return nn.softmax(out)

labels = list(set(labels))
print(labels)
net = BiRNN(40,256,3, 46)
print(net)
net.cuda()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] =  param_group['lr']  / ( 1 + epoch * np.sqrt(2))

def set_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] =  0.01



g = open('logfile','w')
g.close()

optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)
criterion = CTCLoss()

def train_model():
  print_flag = 1
  train_data = list(zip(utterances, phonemes))
  random.shuffle(train_data)
  for epoch in range(10):
    if epoch == 1:
       set_learning_rate(optimizer)
    if epoch > 2:
       adjust_learning_rate(optimizer, epoch)
    print("Starting epoch ", epoch) 
    net.train()
    total_loss = 0
    random.shuffle(train_data)
    for ctr, (utterance, phoneme) in enumerate(train_data):
          phoneme += 1
          #g = open('logfile','a')
          #g.write(" Phonemes: " + ' '.join(str(k) for k in phoneme)  + '\n')
          #g.close()

          input = Variable(torch.from_numpy(utterance).unsqueeze_(0))
          input = input.cuda()
          output = net(input)
          output = output.transpose(0, 1)
          output = output.contiguous() #.cpu()
          targets = Variable(torch.from_numpy(phoneme), requires_grad=False)
          output_size = Variable(torch.from_numpy(np.array([len(output)])).int(), requires_grad=False)
          target_size = Variable(torch.from_numpy(np.array([len(targets)])).int(), requires_grad=False)
          loss = criterion(output, targets, output_size, target_size)
          loss_sum = loss.data.sum()
          inf = float("inf")
          if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
          else:
                loss_value = loss.data[0]

          if ctr  == 100:
              model_name = 'model_test.pkl' 
              torch.save(net, model_name)

          total_loss += loss_value 

          if ctr % 2000 == 1:
               g = open('logfile','a')
               g.write("  Training Loss after " + str(ctr) +  " sequences is : " + str(float(total_loss/(ctr+0.0001))) + '\n') 
               g.close()
               model_name = 'model_latest_topline.pkl'
               torch.save(net, model_name)

               #adjust_learning_rate(optimizer, epoch)   

          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm(net.parameters(), 0.25)
          #if ctr % 32 == 1:
          optimizer.step()
          del loss, input, output, utterance, phoneme, targets

    print("Loss after epoch ", epoch, " is : ", total_loss)
    g = open('logfile','a')
    g.write("Train loss after " + str(epoch) + " is " + str(total_loss/float(ctr+0.001)) + '\n' )
    g.close()
    #evaluate()

    model_name = 'model_' + str(epoch).zfill(3) + '.pkl' 
    torch.save(net, model_name)
    #evaluate()


train_model()
