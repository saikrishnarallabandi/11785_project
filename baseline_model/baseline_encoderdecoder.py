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

# Piazza posts: https://piazza.com/class/j9xdaodf6p1443?cid=2774
batch_size = 24
teacher_forcing  = 0
print_flag = 0

#### Load dev inputs and outputs
dev_input = np.load('../data/train.npy')
dev_output = np.load('../data/train_transcripts.npy')

#### Get an index
frame_lengths = np.zeros( (len(dev_input)), dtype = 'float32')
for i, d in enumerate(dev_input):
   frame_lengths[i] = d.shape[0]
indices = np.argsort(frame_lengths)
dev_input = dev_input[indices]
dev_output = dev_output[indices]

def pad_seq(sequence):

  ordered = sorted(sequence, key=len, reverse=True)
  lengths = [len(x) for x in ordered]
  max_length = lengths[0]
  seq_len = [len(seq) for seq in sequence]
  padded = []
  for i in range(0,len(sequence)):
   npad = ((0, max_length-len(sequence[i])), (0,0))
   padded.append(np.pad(sequence[i], pad_width=npad, mode='constant', constant_values = 0))
  return padded, lengths

dev_input_padded = pad_seq(dev_input)
words_dict = defaultdict(lambda: len(words_dict))
words_dict['<s>']
words_dict['</s>']

for b in dev_output:
   for word in b.split():
       words_dict[word]

def custom_collate_fn(batch):
   
   lengths = np.zeros( (batch_size), dtype ='float32' )
   inputs = []
   outputs = []
   # Get Max length
   for i in range(len(batch)):
       #print("Loading ", i, "th element in ", len(batch), "elements")
       outs = []
       a,b = batch[i]
       b = '<s>' + ' ' + b + ' ' + '</s>' 
       lengths[i] =  len(b.split())
       if len(b) < 1.0:
          print("This is wierd. Inspect b")
          print(b)
          sys.exit()
       inputs.append(a)
       for word in b.split():
            wid = words_dict[word]
            outs.append(wid)
       outputs.append(outs)

   max_len = np.max(lengths) 
   min_len = np.min(lengths)
   #print("Maximum length in the current batch is ", max_len, " and the minimum is ", min_len, "all lengths: ", lengths)
   #print(outputs)
   
   # Pad the ones that dont have max length 
   inputs, outputs = np.array(inputs), np.array(outputs)
   inputs_padded, _ = pad_seq(inputs)

   outputs_padded = []
   for out in outputs:
      #print("Shape of out is ", out.shape)
      l = len(out)
      if l < max_len:
         difference = int(max_len - l)
         #print("The difference is ", difference)
         for k in range(difference):
             out.append(1)
      outputs_padded.append(out)
   #outputs_padded = pad_seq(outputs)

   outputs_categorical = []
   for out in outputs_padded:
         out = to_categorical(out, 17053)
         outputs_categorical.append(out)
   return np.array(inputs_padded), np.array(outputs_categorical)




class ASR(Dataset):

    def __init__(self, A,B):
      
        self.input = A
        self.output = B

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]

# Use Pytorch dataset and dataloading + custom collate function to pad. Shuffle should be false here
dataset = ASR(dev_input,dev_output)
train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn = custom_collate_fn)

def CreateOnehotVariable( input_x, encoding_dim=63):
    if type(input_x) is Variable:
        input_x = input_x.data 
    input_type = type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor)
    onehot_x = Variable(torch.LongTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1,input_x,1)).type(input_type)
    
    return onehot_x

class speller(nn.Module):
   
    def __init__(self, embedding_dim, hidden_dim, target_size, max_label_length=100):
       super(speller, self).__init__()
       self.hidden_dim = hidden_dim
       self.embedding_dim = embedding_dim
       self.target_size = target_size
       self.float_type = torch.torch.cuda.FloatTensor
       self.max_label_length = max_label_length
       self.rnn_layer = nn.LSTM(embedding_dim+hidden_dim*2,hidden_dim*2,num_layers=1)
       self.attention = Attention( mlp_preprocess_input=True, preprocess_mlp_dim=128,
                                    activate='relu', input_feature_dim=hidden_dim)
       self.softmax = nn.LogSoftmax(dim=-1)
       self.character_distribution = nn.Linear(hidden_dim*4,embedding_dim)

    def fwd_step(self,input_word, last_hidden_state,listener_feature):
        rnn_output, hidden_state = self.rnn_layer(input_word,last_hidden_state)
        attention_score, context = self.attention(rnn_output,listener_feature)
        #print("Returned from Attention")
        concat_feature = torch.cat([rnn_output.squeeze(dim=1),context],dim=-1)
        #print("Shape of concat feature", concat_feature.shape)
        raw_pred = self.softmax(self.character_distribution(concat_feature))
        return raw_pred, hidden_state, context, attention_score
    
    def forward(self, context_tensor, ground_truth):
      
       output_word = CreateOnehotVariable(self.float_type(np.zeros((context_tensor.shape[0],1))),embedding_dim)
       rnn_input = torch.cat([output_word.cuda(),context_tensor[:,0:1,:]],dim=-1)

       if print_flag == 1:
         print("  Shape of context matrix is ", context_tensor.shape)
         print("  Shape of word is ", output_word.shape)
         print("  Shape of RNN Input is ", rnn_input.shape)

       max_len = ground_truth.shape[1]
       hidden_state = None
       attention_tensor = []
       prediction_tensor = []
       for i in range(max_len):
           probs,hidden_state,context,attention = self.fwd_step(rnn_input, hidden_state, context_tensor)
           if print_flag == 1:
             print("Currently in timestep", i+1)
             print("  Shape of attention tensor: ", attention.shape)
             print("  Shape of probs is ", probs.shape)
             print("  Shape of context: ", context.shape)
           attention_tensor.append(attention)
           prediction = probs.unsqueeze(1)
           prediction_tensor.append(probs)
           if print_flag  == 1:
             print("  Shape of prediction: ", prediction.shape)
             print("  Shape of context tensor: ", context_tensor.shape)
 
           if teacher_forcing == 1:
               rnn_input = torch.cat([ground_truth[:,i:i+1,:].type(self.float_type), context_tensor[:,0:1,:]],dim=-1)
           else:
               rnn_input = torch.cat([prediction, context_tensor[:,0:1,:]],dim=-1)
           if  print_flag  == 1:
             print("Time step done", i+1)
             print('\n')

       return prediction_tensor


class Attention(nn.Module):
    def __init__(self, mlp_preprocess_input, preprocess_mlp_dim, activate, input_feature_dim=512):
        super(Attention,self).__init__()
        self.mlp_preprocess_input = False # mlp_preprocess_input
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        if mlp_preprocess_input:
            self.preprocess_mlp_dim  = preprocess_mlp_dim
            self.phi = nn.Linear(input_feature_dim,preprocess_mlp_dim)
            self.psi = nn.Linear(input_feature_dim,preprocess_mlp_dim)
            self.activate = getattr(F,activate)

    def forward(self, decoder_state, listener_feature):
        comp_decoder_state = decoder_state
        comp_listener_feature = listener_feature.transpose(1,2)
        energy = torch.bmm(comp_decoder_state,comp_listener_feature).squeeze(dim=1)
        attention_score = self.softmax(energy)
        context = torch.sum(listener_feature*attention_score.unsqueeze(2).repeat(1,1,listener_feature.size(2)),dim=1)

        return attention_score,context



class listener(nn.Module):
     
    def __init__(self, input_dim, hidden_dim):
       super(listener, self).__init__()
       self.hidden_dim = hidden_dim
       self.lstm = nn.LSTM(input_dim, hidden_dim, 1,  batch_first = True, bidirectional=True)

    def forward(self, utterance_batch):
        batch_size = utterance_batch.shape[0]
        time_steps = utterance_batch.shape[1]
        feature_dim = utterance_batch.shape[2]
        x_input = utterance_batch.contiguous()
        lstm_out, self.hidden = self.lstm(x_input, None)
        return lstm_out

i2w = {i:w for w,i in words_dict.items()}
embedding_dim = len(words_dict)
hidden_dim = 128
vocab_size = len(words_dict)
target_size = vocab_size
input_dim = 40
print("The target size is ", target_size)
baseline_listener = listener(input_dim, hidden_dim)
baseline_speller = speller(embedding_dim, hidden_dim, vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(list(baseline_listener.parameters()) + list( baseline_speller.parameters()), lr=0.01)
baseline_listener.cuda()
baseline_speller.cuda()
objective = nn.CrossEntropyLoss(ignore_index=0)
objective.cuda()

def train():
 total_loss = 0
 for ctr, t in enumerate(train_loader):
    optimizer.zero_grad()
    a,b = t
    #print(a.shape, b.shape)
    a,b = Variable(torch.from_numpy(a)), Variable(torch.from_numpy(b), requires_grad=False)
    a,b = a.cuda(), b.cuda()
    pred = baseline_listener(a)
    #print("Shape of listener output:", pred.shape)
    #print("Batch Done")
    words = baseline_speller(pred, b)
    pred_y = torch.cat([torch.unsqueeze(each_y,1) for each_y in words],1).view(-1,embedding_dim)
    true_y = torch.max(b,dim=2)[1].view(-1)
    true_y = true_y.detach()
    #print("Shape of words:", pred_y.shape)
    #print("Shape of Truth: ", true_y.shape)
    loss = objective(pred_y, true_y)
    total_loss += loss.cpu().data.numpy()
    #print('\n')
    #sys.exit()

    '''
    sent = ' '
    for k in true_y:
       sent += ' ' + i2w[int(k.data.cpu().numpy())]
    print("Sentnence is ", sent)
    total_loss += loss.cpu().data.numpy()
    if ctr % 4 == 1 :
       print(" Loss after ", ctr+1, "batches: ", total_loss/ (ctr+1) * 1.0 * batch_size)
    for k in true_y:
       sent += ' ' + i2w[int(k.data.cpu().numpy())]
    print("Original Sentence is ", sent)
    print("\n")

    sent = ' '
    for k in pred_y:
       k = torch.argmax(k)
       sent += ' ' + i2w[int(k.data.cpu().numpy())]
    print("Predicted Sentence is ", sent)
    print('\n')
    '''

    loss.backward()
    optimizer.step()
 return pred_y, true_y, total_loss/((ctr+1) * 1.0 * batch_size)  


for epoch in range(10):
    print("Running epoch ", epoch)
    pred_y, true_y, total_loss = train()

    print(" Loss after ", epoch+1, "epochs: ", total_loss)
    sent = ' '
    for k in true_y:
       sent += ' ' + i2w[int(k.data.cpu().numpy())]
    print("   Original Sentence is ", sent.split()[0:20])
    print("\n")

    sent = ' '
    for k in pred_y:
       k = torch.argmax(k)
       sent += ' ' + i2w[int(k.data.cpu().numpy())]
    print("   Predicted Sentence is ", sent.split()[0:20])
    print('\n')


