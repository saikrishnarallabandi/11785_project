import torch
import numpy as np
from pickle import dump


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def get_char_tensor(text, mapping):
    seq = [mapping[char] for char in text]
    return to_tensor(np.array(seq)).long()


def get_encoded_sequence(text, mapping):
    return to_variable(get_char_tensor(text, mapping)).view(1, -1)


def read_file(filename):
    file = open(filename, 'r', encoding='utf-8').read()
    return file


def load_data(in_filename):
    # load
    raw_text = read_file(in_filename)

    # integer encode sequences of characters
    chars = sorted(list(set(raw_text)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    reverse_mapping = dict((i, c) for i, c in enumerate(chars))

    # save the mappings
    dump(mapping, open('mapping.pkl', 'wb'))
    dump(reverse_mapping, open('reverse_mapping.pkl', 'wb'))

    # vocabulary size
    vocab_size = len(mapping)
    print('Vocabulary Size: %d' % vocab_size)

    return ' '.join(raw_text.split()), vocab_size, mapping
