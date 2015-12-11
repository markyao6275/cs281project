import sys
import matplotlib.pyplot as plt

mid_separated = True
filename = 'largeweights.txt'

datapoints = []
curr_datapoint = None
curr_iter = 0

class DataPoint:
    def __init__(self, iter_num, train_err, test_err, alpha, loss, L):
        self.iter_num = iter_num
        self.train_err = train_err
        self.test_err = test_err
        self.alpha = alpha
        self.loss = loss
        self.L = L

with open(filename, 'r') as f:
    for line in f:
        if line == '\n':
            continue

        if mid_separated:
            data_string = line[:-1].split('|')
            train_err, test_err, alpha = map(float, data_string)
            curr_datapoint = DataPoint(curr_iter, train_err, test_err, alpha, None, None)
            curr_iter += 1
            mid_separated = False
        else:
            data_string = line[:-1].split()
            L, loss = map(float, data_string)
            curr_datapoint.L = L
            curr_datapoint.loss = loss
            datapoints.append(curr_datapoint)
            curr_datapoint = None
            mid_separated = True

train_errs = [data.train_err for data in datapoints]
test_errs = [data.test_err for data in datapoints]
Ls = [data.L for data in datapoints]
losses = [data.loss for data in datapoints]
iterations = [data.iter_num for data in datapoints]
plt.subplot(131)
plt.plot(iterations, train_errs, color='blue', label='Training Error')
plt.plot(iterations, test_errs, color='red', label='Test Error')
plt.legend(loc='upper right')
plt.title('Training and Test Errors vs. Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Error')

plt.subplot(132)
plt.plot(iterations, Ls)
plt.title('L vs. Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('L')

plt.subplot(133)
plt.plot(iterations, losses)
plt.title('Loss vs. Iteration Number')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.savefig('largeweights_plot')






