from pickle import load
from char_rnn import CharRNN
from utils import *


# generate a sequence of characters with a language model
def generate_seq(model, mapping, reverse_mapping, seed_text, n_chars):
    in_text = seed_text
    # encode the characters as integers
    primer = get_encoded_sequence(in_text, mapping)
    hidden = model.init_hidden(1)
    # Use priming string to "build up" hidden state
    for p in range(primer.size()[1] - 1):
        _, hidden = model(primer[:, p], hidden)
    input = primer[:, -1]
    predicted = in_text
    # generate a fixed number of characters, to generate indefinite string replace this with while loop
    for _ in range(n_chars):
        # predict character
        y, hidden = model(input, hidden)
        _, yhat = y.max(dim=1)
        yhat = yhat.data.cpu()[0]
        char = reverse_mapping[yhat]
        predicted += char
        input = get_encoded_sequence(char, mapping)
    return predicted


vocab_size, hidden_size, embedding_dim, n_layers = load(open('state_vars.pkl', 'rb'))
# load the model
model = CharRNN(hidden_size=hidden_size, embedding_dim=embedding_dim,
                output_size=vocab_size, n_layers=n_layers)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('model.t7'))
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))
reverse_mapping = load(open('reverse_mapping.pkl', 'rb'))

# Generate few sentences
print(generate_seq(model, mapping, reverse_mapping, 'టాలు కూడా వ', 3000))
print(generate_seq(model, mapping, reverse_mapping, 'కదా అంత మంచ', 3000))
print(generate_seq(model, mapping, reverse_mapping, 'ును అందుకని', 3000))
