import torch.utils.data
from char_rnn import CharRNN
from utils import *
import argparse
import math
from timeit import default_timer as timer


def train_set(args, file, file_len, mapping):
    input = to_tensor(np.zeros((args.batch_size, args.seq_length))).long()
    target = to_tensor(np.zeros((args.batch_size, args.seq_length))).long()
    for i in range(args.batch_size):
        start_idx = np.random.randint(0, file_len - args.seq_length)
        end_idx = start_idx + args.seq_length + 1
        seq = file[start_idx:end_idx]
        input[i] = get_char_tensor(seq[:-1], mapping)
        target[i] = get_char_tensor(seq[1:], mapping)
    return input, target


def train_batch(my_net, optim, loss_fn, args, input_val, label):
    hidden = my_net.init_hidden(input_val.size()[0])
    optim.zero_grad()  # Reset the gradients
    prediction, _ = my_net(to_variable(input_val), hidden)  # Feed forward
    loss = loss_fn(prediction, to_variable(label).view(-1))  # Compute losses
    loss.backward()  # Backpropagate the gradients
    optim.step()  # Update the network
    return np.asscalar(loss.data.cpu().numpy())


def evaluate(my_net, loss_fn, mapping, test):
    total_loss = 0
    for each in test:
        input = get_char_tensor(each[:-1], mapping)
        target = get_char_tensor(each[1:], mapping)
        hidden = my_net.init_hidden(1)
        prediction, _ = my_net(to_variable(input).unsqueeze(0), hidden)
        total_loss += loss_fn(prediction, to_variable(target).view(-1)).data
    return total_loss / len(test)


def train(args):
    # load data
    text, vocab_size, mapping = load_data('transcription_train.txt')
    test = load_test_file('transcription_test.txt')
    # Dump few states to use in generation
    dump([vocab_size, args.hidden_size, args.embedding_dim, args.n_layers], open('state_vars.pkl', 'wb'))

    my_net = CharRNN(hidden_size=args.hidden_size, embedding_dim=args.embedding_dim,
                     output_size=vocab_size, n_layers=args.n_layers)  # Create the network,
    loss_fn = torch.nn.CrossEntropyLoss()  # loss function / optimizer
    optim = torch.optim.Adam(my_net.parameters(), lr=args.learning_rate)

    if torch.cuda.is_available():
        # Move the network and the optimizer to the GPU
        my_net = my_net.cuda()
        loss_fn = loss_fn.cuda()

    loss_avg = 0
    prev_ppl = 1000000
    for epoch in range(1, args.n_epochs + 1):
        start_time = timer()
        loss = train_batch(my_net, optim, loss_fn, args, *train_set(args, text, len(text), mapping))
        loss_avg += loss
        if epoch % args.print_every == 0:
            val_loss = evaluate(my_net, loss_fn, mapping, test)
            ppl = math.exp(val_loss)
            print(
                "Epoch {} : Training Loss: {:.5f}, Test ppl: {:.5f}, Time elapsed {:.2f} mins"
                    .format(epoch, loss, ppl, (timer() - start_time) / 60))
            if ppl < prev_ppl:
                prev_ppl = ppl
                torch.save(my_net.state_dict(), 'bestModel.t7')
                print("Perplexity reduced, saving model !!")
    return my_net


def main(args):
    train(args)


if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--n_epochs', type=int, default=100)
    argparser.add_argument('--hidden_size', type=int, default=256)
    argparser.add_argument('--embedding_dim', type=int, default=128)
    argparser.add_argument('--n_layers', type=int, default=3)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    argparser.add_argument('--seq_length', type=int, default=50)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--print_every', type=int, default=1)
    args = argparser.parse_args()
    main(args)
