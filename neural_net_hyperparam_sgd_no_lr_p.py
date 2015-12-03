from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import quick_grad_check
from scipy import optimize
import sys

import matplotlib.pyplot as plt

filename = "default.txt" if len(sys.argv) < 2 else sys.argv[1]

def logsumexp(X, axis=1):
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=True))

def get_wine_data():
    print("Loading training data...")
    wine_data_file = open('./wines_data/wine.data', 'r')
    num_class_1, num_class_2, num_class_3 = 59, 71, 48
    wines_data = []

    for line in wine_data_file:
        entries = line.split(',')
        wine_type = int(entries[0]) # 1, 2, or 3
        wine_type_one_hot = [1., 0., 0.] if wine_type == 1 else [0., 1., 0.] if wine_type == 2 else [0., 0., 1.]
        wine_features = map(float, entries[1:-1])

        wine_data = wine_features
        wine_data.extend(wine_type_one_hot)
        wines_data.append(wine_data)

    wines_data = np.array(wines_data)
    np.random.shuffle(wines_data)
    features, labels = wines_data[:, :-3], wines_data[:, -3:]

    num_data_pts = len(wines_data)
    train_set_size = 0.8 * num_data_pts
    train_data, train_labels = features[:train_set_size], labels[:train_set_size]
    test_data, test_labels = features[train_set_size:], labels[train_set_size:]

    assert np.sum(wines_data[:, -3]) == num_class_1
    assert np.sum(wines_data[:, -2]) == num_class_2
    assert np.sum(wines_data[:, -1]) == num_class_3
    return num_data_pts, train_data, train_labels, test_data, test_labels


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x)) if x >= 0 else np.exp(x - np.logaddexp(x, 0))

def make_nn_funs(layer_sizes, L2_reg):
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    N = sum((m+1)*n for m, n in shapes)

    def unpack_layers(W_vect):
        for m, n in shapes:
            yield W_vect[:m*n].reshape((m,n)), W_vect[m*n:m*n+n]
            W_vect = W_vect[(m+1)*n:]

    def predictions(W_vect, inputs, alpha):
        outputs = 0
        for W, b in unpack_layers(W_vect):
            prev_outputs = outputs
            outputs = np.dot(np.array(inputs), W) + b
            inputs = np.tanh(outputs)
        return sigmoid(alpha) * outputs + (1 - sigmoid(alpha)) * prev_outputs

    def loss(W_vect, X, T, alpha):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        preds = predictions(W_vect, X, alpha)
        normalised_log_probs = preds - logsumexp(preds)
        log_lik = np.sum(normalised_log_probs * T)
        return -1.0 * (log_prior + log_lik)

    def frac_err(W_vect, alpha, X, T):
        percent_wrong = np.mean(np.argmax(T, axis=1) != np.argmax(predictions(W_vect, X, alpha), axis=1))
        return percent_wrong

    return N, predictions, loss, frac_err


def load_mnist():
    print("Loading training data...")
    import imp, urllib
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    source, _ = urllib.urlretrieve(
        'https://raw.githubusercontent.com/HIPS/Kayak/master/examples/data.py')
    data = imp.load_source('data', source).mnist()
    train_images, train_labels, test_images, test_labels = data
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def make_batches(N_data, batch_size):
    return [slice(i, min(i+batch_size, N_data))
            for i in range(0, N_data, batch_size)]


def run_nn(params, learning_rate, momentum, input_size, output_size):
    N = params[0]

    integer_part = int(np.floor(N))
    alpha = N - integer_part
    # Network parameters
    layer_sizes = [input_size]
    layer_sizes.extend([output_size for i in range(0, integer_part - 1)])
    L2_reg = 1.0

    # Training parameters
    param_scale = 0.1
    num_epochs = 100

    # Load and process MNIST data (borrowing from Kayak)
    #N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    N_data, train_images, train_labels, test_images, test_labels = get_wine_data()
    batch_size = len(train_images)
    #train_images, test_images = np.array(zip(x1_train, x2_train)), np.array(zip(x1_test, x2_test))
    #train_labels, test_labels = y_train_labels, y_test_labels

    # Make neural net functions
    N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg)
    loss_grad_W = grad(loss_fun, 0)
    #loss_grad_alpha = grad(loss_fun, 3)

    # Initialize weights
    rs = npr.RandomState(11)
    W = rs.randn(N_weights) * param_scale
    #alpha = 1000000.

    # Check the gradients numerically, just to be safe
    # quick_grad_check(loss_fun, W, (train_images, train_labels))

    #print("    Epoch      |    Train err  |   Test err  ")
    print("    Train err  |   Test err  |   Alpha   |   Loss   ")
    f_out = open(filename, 'w')
    f_out.write("    Train err  |   Test err  |   Alpha   \n")
    f_out.close()

    def print_perf(epoch, W, alpha, learning_rate, momentum):
        f_out = open(filename, 'a')
        test_perf  = frac_err(W, alpha, test_images, test_labels)
        train_perf = frac_err(W, alpha, train_images, train_labels)
        loss = loss_fun(W, train_images, train_labels, alpha)
        print("{0:15}|{1:15}|{2:15}|{3:15}|{4:15}".format(epoch, train_perf, test_perf, alpha, loss))
        f_out.write("{0:15}|{1:15}|{2:15}|{3:15}|{4:15}\n".format(epoch, train_perf, test_perf, alpha, loss))
        f_out.close()

    # Train with sgd
    batch_idxs = make_batches(train_images.shape[0], batch_size)
    cur_dir_W = np.zeros(N_weights)
    cur_dir_alpha = 0

    for epoch in range(num_epochs):
        print_perf(epoch, W, alpha, learning_rate, momentum)
        for idxs in batch_idxs:
            grad_W = loss_grad_W(W, train_images[idxs], train_labels[idxs], alpha)
            cur_dir_W = momentum * cur_dir_W + (1.0 - momentum) * grad_W
            W = W - learning_rate * cur_dir_W

            """
            grad_alpha = loss_grad_alpha(W, train_images[idxs], train_labels[idxs], alpha)
            cur_dir_alpha = momentum * cur_dir_alpha + (1.0 - momentum) * grad_alpha
            alpha -= learning_rate * cur_dir_alpha
            """

    final_test_err = loss_fun(W, train_images, train_labels, alpha)
    print(N, final_test_err)
    return final_test_err

if __name__ == '__main__':
    run_nn_grad = grad(run_nn, 0)
    N = 3.4 
    learning_rate = 1e-5
    momentum = 0.1
    optimize.minimize(run_nn, (N), jac=run_nn_grad, method='L-BFGS-B', \
        args=(learning_rate, momentum, 12, 3), options={'disp': True})
