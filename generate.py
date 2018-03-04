import numpy as np
from pickle import load
from char_rnn import CharRNN
from utils import *


def get_encoded_sequence(text, mapping):
    seq = [mapping[char] for char in text]
    return to_variable(to_tensor(np.array(seq)).long()).view(1, -1)


# generate a sequence of characters with a language model
def generate_seq(model, mapping, seed_text, n_chars):
    in_text = seed_text
    # encode the characters as integers
    primer = get_encoded_sequence(in_text, mapping)
    hidden = model.init_hidden(1)
    # Use priming string to "build up" hidden state
    for p in range(primer.size()[1] - 1):
        _, hidden = model(primer[:, p], hidden)
    input = primer[:, -1]
    predicted = in_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        # predict character
        y, hidden = model(input, hidden)
        _, yhat = y.max(dim=1)
        yhat = yhat.data[0]
        # reverse map integer to character, not an efficient way
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # append to input
        predicted += char
        input = get_encoded_sequence(char, mapping)
    return predicted


# load the model
model = CharRNN(hidden_size=75, embedding_dim=75, output_size=53)   # 53 is the vocab_size
model.load_state_dict(torch.load('model.t7'))
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))

# test start of rhyme
print(generate_seq(model, mapping, 'టాలు కూడా వ', 3000))
# test mid-line
print(generate_seq(model, mapping, 'కదా అంత మంచ', 3000))
# test not in original
print(generate_seq(model, mapping, 'ును అందుకని', 3000))
