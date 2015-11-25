import matplotlib.pyplot as plt
import sys

"""
Usage:

python parse_plot_data.py <data file> <title for training/test error chart> <title for alpha chart>

"""


TRAINING_ERR_INDEX, TEST_ERR_INDEX, ALPHA_INDEX = 0, 1, 2

if len(sys.argv) < 2:
    raise ValueError("Please specify a txt file to print!")

data_filename = sys.argv[1]
iteration_nums, training_errors, test_errors, alphas = [], [], [], []

with open(data_filename, 'r') as input_file:
    next(input_file) # skips the first line because it's a header

    curr_iteration = 0
    for line in input_file:
        data_as_string = line.split('|')
        data = map(float, data_as_string)
        iteration_nums.append(curr_iteration)
        training_errors.append(data[TRAINING_ERR_INDEX])
        test_errors.append(data[TEST_ERR_INDEX])
        alphas.append(data[ALPHA_INDEX])
        curr_iteration += 1

data_filename_without_filetype = data_filename.rsplit('.txt', 1)[0]

train_test_err_graph_title = sys.argv[2] if len(sys.argv) > 2 else 'Training and Test Errors'
alpha_graph_title = sys.argv[3] if len(sys.argv) > 3 else 'Values of Alpha'

train_test_err_graph_filename = data_filename_without_filetype
alpha_graph_filename = train_test_err_graph_filename + "_alpha"

plt.plot(iteration_nums, training_errors, color='blue', label='Training Errors')
plt.plot(iteration_nums, test_errors, color='red', label='Test Errors')
plt.legend(loc='upper right')
plt.title(train_test_err_graph_title)
plt.xlabel('Iteration Number')
plt.ylabel('Proportion Misclassified')
plt.savefig(train_test_err_graph_filename)

plt.clf()
plt.plot(iteration_nums, alphas)
plt.title(alpha_graph_title)
plt.xlabel('Iteration Number')
plt.ylabel('Alpha')
plt.savefig(alpha_graph_filename)

