import numpy as np
import torch.utils.data
from char_rnn import CharRNN
from pickle import dump
from utils import *
import sys
from timeit import default_timer as timer


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def load_data():
    # load
    in_filename = 'char_sequences.txt'
    raw_text = load_doc(in_filename)
    lines = raw_text.split('\n')

    # integer encode sequences of characters
    chars = sorted(list(set(raw_text)))
    mapping = dict((c, i) for i, c in enumerate(chars))

    # save the mapping, TODO: dump a reverse map as well, useful at decoding time
    dump(mapping, open('mapping.pkl', 'wb'))

    sequences = list()
    for line in lines:
        # integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)

    # vocabulary size
    vocab_size = len(mapping)
    print('Vocabulary Size: %d' % vocab_size)

    # separate into input and output
    sequences = np.array(sequences)
    x, y = sequences[:, :-1], sequences[:, -1]
    return x, y, vocab_size


def train_batch(my_net, optim, loss_fn, input_val, label):
    hidden = my_net.init_hidden(input_val.size()[0])
    optim.zero_grad()  # Reset the gradients
    # No attention, vanilla stuff
    for seq in range(10):
        prediction, hidden = my_net(to_variable(input_val[:, seq]), hidden)  # Feed forward

    loss = loss_fn(prediction, to_variable(label))  # Compute losses
    loss.backward()  # Backpropagate the gradients
    optim.step()  # Update the network

    return np.asscalar(loss.data.cpu().numpy())


def train(learning_rate, minibatch_size, num_epochs):
    # load data
    x, y, vocab_size = load_data()
    my_net = CharRNN(hidden_size=75, embedding_dim=75, output_size=vocab_size)  # Create the network,
    loss_fn = torch.nn.CrossEntropyLoss()  # loss function / optimizer
    optim = torch.optim.Adam(my_net.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        my_net = my_net.cuda()
        loss_fn = loss_fn.cuda()

    dataset = torch.utils.data.TensorDataset(to_tensor(x).long(), to_tensor(y).long())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=minibatch_size, shuffle=True)
    loss_avg = 0
    for epoch in range(num_epochs):
        start_time = timer()
        minibatch = 1
        for (input_val, label) in data_loader:
            loss = train_batch(my_net, optim, loss_fn, input_val, label)
            loss_avg += loss
            sys.stdout.write("[%d/%d] :: Training Loss: %f   \r" % (
            minibatch, len(y) // minibatch_size, loss))
            sys.stdout.flush()
            minibatch += 1

        print(
            "Epoch {} : Training Loss: {:.5f}, Time elapsed {:.2f} mins"
            .format(epoch, loss, (timer() - start_time) / 60))
    return my_net


def main():
    trained_model = train(learning_rate=0.01, minibatch_size=128, num_epochs=100)
    torch.save(trained_model.state_dict(), 'model.t7')


if __name__ == '__main__':
    main()
