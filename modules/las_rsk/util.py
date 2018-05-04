import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from keras.utils import to_categorical


def sort_arrays(dev_input, dev_output):
   frame_lengths = np.zeros( (len(dev_input)), dtype = 'float32')
   for i, d in enumerate(dev_input):
      frame_lengths[i] = d.shape[0]
   indices = np.argsort(frame_lengths)
   dev_input = dev_input[indices]
   dev_output = dev_output[indices]
   
   words_dict = defaultdict(lambda: len(words_dict))
   words_dict['<s>'] 
   words_dict['</s>']
   output_numeric = []
   for b in dev_output:
      out = []
      for word in b.split():
          wid = words_dict[word]
          out.append(wid)
      output_numeric.append(out)
   output_numeric = np.array(output_numeric)
   return dev_input, output_numeric, words_dict


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


def custom_collate_fn(batch):
   
   lengths = np.zeros( len(batch), dtype ='float32' )
   inputs = []
   outputs = []

   # Get Max length
   for i in range(len(batch)):
       outs = []
       a,b = batch[i]
       b = [0]  + b + [1] 
       lengths[i] =  len(b)
       if len(b) < 1.0:
          print("This is wierd. Inspect b")
          print(b)
          sys.exit()
       inputs.append(a)
       for w in b:
           outs.append(w)
       outputs.append(outs)

   max_len = np.max(lengths) 
   min_len = np.min(lengths)
   
   # Pad the ones that dont have max length 
   inputs, outputs = np.array(inputs), np.array(outputs)
   inputs_padded, _ = pad_seq(inputs)

   outputs_padded = [] 
   masks = []
   for out in outputs:
      mask = np.zeros(int(max_len), dtype='float32')
      for i,k in enumerate(out):
          mask[i] = 1
          #mask.append(1.0)
      #print("DType of mask is ", mask.dtype)
      l = len(out)
      if l < max_len:
         difference = int(max_len - l)
         #print("The difference is ", difference)
         for k in range(difference):
             out.append(1)
      outputs_padded.append(out)
      masks.append(mask)
      #print("DType of masks is ", np.array(masks).dtype)

   #outputs_padded = pad_seq(outputs)

   outputs_categorical = []
   for out in outputs_padded:
         out = to_categorical(out, 17053)
         outputs_categorical.append(out)
   #print("Returning masks with dtype, ", np.array(masks).dtype)
   return np.array(inputs_padded), np.array(outputs_padded), np.array(masks)

class WSJ(Dataset):
    """WSJ Landmarks dataset."""

    def __init__(self, A,B):
      
        self.input = A
        self.output = B

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
       inp = self.input[idx]
       len_inp = len(inp)
       if len_inp % 8 == 0:
          pass
       else:
          k = len_inp % 8
          inp = inp[:-k] 
       #print("Number of timesteps", len(inp))
       return inp, self.output[idx]



