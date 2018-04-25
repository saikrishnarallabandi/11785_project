#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from pickle import load
from char_rnn import CharRNN
from utils import *


def sample_gumbel(shape, eps=1e-10, out=None):
        """
        Sample from Gumbel(0, 1) based on (MIT license)
        """
        U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
        return - torch.log(eps - torch.log(U + eps))
    

# generate a sequence of characters with a language model
def generate_seq(model, mapping, reverse_mapping, seed_text, n_chars):
    in_text = seed_text
    # encode the characters as integers
    primer = get_encoded_sequence(in_text, mapping)
    hidden = model.init_hidden(1)
    # Use priming string to "build up" hidden state
    for p in range(primer.size()[1] - 1):
        _, hidden = model(primer[:, p].unsqueeze(0), hidden)
    input = primer[:, -1]
    predicted = in_text
    # generate a fixed number of characters, to generate indefinite string replace this with while loop
    for _ in range(n_chars):
        # predict character
        y, hidden = model(input.unsqueeze(0), hidden)
        gumbel = torch.autograd.Variable(sample_gumbel(shape=y.size(), out=y.data.new()))
        y += gumbel
        _, yhat = y.max(dim=1)
        yhat = yhat.data.cpu()[0]
        char = reverse_mapping[yhat]
        predicted += char
        input = get_encoded_sequence(char, mapping)[0]
    return predicted


vocab_size, hidden_size, embedding_dim, n_layers = load(open('state_vars.pkl', 'rb'))
# load the model
model = CharRNN(hidden_size=hidden_size, embedding_dim=embedding_dim,
                output_size=vocab_size, n_layers=n_layers)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('bestModel_3.55.t7'))
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))
reverse_mapping = load(open('reverse_mapping.pkl', 'rb'))

# Generate few sentences
print(generate_seq(model, mapping, reverse_mapping, '0ప', 300))
#print(generate_seq(model, mapping, reverse_mapping, '0', 300))
#print(generate_seq(model, mapping, reverse_mapping, '0', 300))
