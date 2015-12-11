import sys
import re
import matplotlib.pyplot as plt
import numpy as np

filename = 'hyperparam_output.txt'

is_autograd = True
just_saw_loading_text = False
created_file = False
curr_batch = 0
loading_text = "Loading training data...\n"
last_line = "RUNNING THE L-BFGS-B CODE\n"

class DataPoint:
    def __init__(self, batch_num, iter_num, train_err, test_err, alpha, loss, L, is_autograd):
        self.batch_num = batch_num
        self.iter_num = iter_num
        self.train_err = train_err
        self.test_err = test_err
        self.alpha = alpha
        self.loss = loss
        self.L = L
        self.is_autograd = is_autograd

class LLossPair:
    def __init__(self, L, loss, is_autograd):
        self.L = L
        self.loss = loss
        self.is_autograd = is_autograd

datapoints = []
n_loss_pairs = []
tmp_datapoints = []


with open(filename, 'r') as f:
    for line in f:
        if line == last_line:
            break
        elif line == loading_text:
            is_autograd = not is_autograd
            just_saw_loading_text = True
            curr_batch += 1
            continue
        elif just_saw_loading_text:
            just_saw_loading_text = False
            continue

        data_string = line[:-1].split('|')
        if len(data_string) == 5:
            if is_autograd:
            # deal with the autograd nodes
                valid_floats = data_string[:3]
                iter_num, train_err, test_err = map(float, valid_floats)
                alpha = re.findall("\d+\.\d+", data_string[3])
                loss = re.findall("\d+\.\d+", data_string[4])
            else:
                iter_num, train_err, test_err, alpha, loss = map(float, data_string)

                # won't specify L until we read it in
            curr_datapoint = DataPoint(curr_batch, iter_num, train_err, test_err, alpha, loss, None, is_autograd)
            tmp_datapoints.append(curr_datapoint)

        else: # it's L and the loss
            if is_autograd:
                split_string = data_string[0].split('(s)')
                L = float(re.findall("\d+\.\d+", split_string[0])[0])
                loss = float(re.findall("\d+\.\d+", split_string[1])[0])
            else:
                L, loss = map(float, data_string[0].split())

            for i in xrange(len(tmp_datapoints)):
                tmp_datapoints[i].L = L
            datapoints.extend(tmp_datapoints)
            tmp_datapoints = []

            n_loss_pair = LLossPair(L, loss, is_autograd)
            n_loss_pairs.append(n_loss_pair)

# plot L for the autograd (L-BFGS) runs
Ls = [pair.L for pair in n_loss_pairs if pair.is_autograd]
iter_Ls = np.arange(0, len(Ls))
plt.scatter(iter_Ls, Ls)
plt.title('L vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('L')
plt.savefig('iterationvsL')

# plot loss for the autograd (L-BFGS) runs
plt.clf()
losses = [pair.loss for pair in n_loss_pairs if pair.is_autograd]
iter_losses = np.arange(0, len(losses))
plt.scatter(iter_losses, losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('lossvsL')

# training_err, test_err, loss vs. epoch for some SGD runs
target_batch = 6

def plot_sgd_stats_for_batch(target_batch):
    train_errs = [data.train_err for data in datapoints if data.batch_num == target_batch and data.is_autograd]
    test_errs = [data.test_err for data in datapoints if data.batch_num == target_batch and data.is_autograd]
    losses = [data.loss for data in datapoints if data.batch_num == target_batch and data.is_autograd]
    iteration = np.arange(0, len(train_errs))
    plt.clf()
    plt.subplot(121)
    plt.plot(iteration, train_errs, color='blue', label='Training Error')
    plt.plot(iteration, test_errs, color='red', label='Test Error')
    plt.title('Train and Test Error vs Epoch for L-BFGS-B Run')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend(loc='upper right')
    plt.subplot(122)
    plt.title('Loss vs Epoch for L-BFGS-B Run')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(iteration, losses, label='Loss')
    plt.legend(loc='upper right')
    plt.savefig('lbfgs-stats-batch' + str(target_batch))

plot_sgd_stats_for_batch(6)
plot_sgd_stats_for_batch(20)
plot_sgd_stats_for_batch(40)


