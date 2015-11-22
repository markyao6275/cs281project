from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import quick_grad_check

import matplotlib.pyplot as plt

x1_train = [66.6364165533359, 26.601006809981087, 27.027000079098997, 95.5417300759668, 96.94725331436625, 21.25854714270121, 13.145738090434833, 34.86809873211274, 39.394081252749615, 99.6737924340724, 26.08128440760661, 49.979999554561594, 45.22723049366354, 70.60652157756172, 52.90756511089998, 88.50202158134847, 20.203692512974847, 23.300119292351496, 51.713877646037176, 8.031043877784283, 27.76512680045059, 41.601003946919334, 84.99397284642889, 45.333593953559095, 16.206795428247112, 69.39755331953113, 60.0072153864466, 87.50665974928684, 72.05538053850947, 22.112685963724765, 97.58612712756137, 64.20890786909817, 14.198348819045304, 96.25737979931034, 90.51716365727883, 92.68109890424292, 84.84948371309909, 43.25624900420664, 80.67512036607111, 72.6979873992053, 83.83677619711818, 94.33213869485954, 44.34986586330236, 67.4374454520864, 11.459085248901191, 91.50770750950645, 32.12196249642576, 78.4427190938891, 39.465323668923766, 62.953570500587205, 86.8029763826086, 9.155052032118217, 53.162965076417436, 26.843860869717528, 5.2705203392161115, 56.9086585644142, 1.6055386661040116, 22.367330221132143, 50.40471306006412, 2.0170367804830724, 35.62509382316651, 35.91682454329772, 27.57834627340664, 32.8406244803684, 78.16606668366614, 91.91098948081131, 18.275043524914146, 2.1678593137649194, 20.529007187624416, 23.94387822836591, 44.485197770424456, 80.10639471391121, 83.20764865352155, 6.588576786816292, 40.08614106514763, 27.684186059073802, 45.337991730083175, 82.83040225273072, 37.26342073011077, 99.79187346170964, 68.02533889559558, 73.03252881468252, 55.772927738632575, 71.55117096215014, 73.84058935298047, 17.892732771694742, 58.15253817802436, 2.3254582305687777, 39.29638683281734, 18.29520907389559, 95.63521382988763, 76.01652103643286, 29.112922136028807, 39.77739271037141, 78.9973717751475, 4.273799940067247, 42.78826416707597, 74.64523100598073, 23.6138363102823, 59.358415964896]
x2_train = [97.84669655212498, 48.76517850862271, 91.35387335031999, 90.42385319134759, 5.7931895881954905, 1.5047380803805854, 52.42466000726799, 5.958736219858585, 97.22723928225764, 93.65410527932396, 45.72208286351874, 54.35377165377881, 34.35966585558027, 81.5923488902045, 69.98619739376745, 62.884336964008604, 1.1657122234731698, 20.88497333038134, 33.15776182027451, 70.97077529114159, 22.618241523521586, 45.997849506014546, 6.601362236164954, 75.82683198855055, 34.298262316392524, 9.18036427318455, 44.1551311399469, 89.87470881238733, 7.797931157866778, 34.791473154632804, 59.028722625655504, 6.313523718086356, 18.159661912266756, 12.062020951197805, 1.3158760146393722, 52.2722823997697, 41.65434544156573, 62.290764598145785, 64.65088389271338, 88.94946302793039, 23.74523052184322, 75.42693701539676, 87.65989712387477, 7.149942850470003, 42.49262035580932, 2.968500676519392, 96.98370433156231, 46.89733694585356, 99.82758200014477, 47.45425876098177, 33.427502005008655, 13.320372405554593, 73.449997738712, 88.04483628745793, 46.34620292732441, 60.079108208085664, 17.889914577958677, 59.50215225146813, 71.71251284983342, 85.70094940715447, 87.69372922982987, 97.85578944701273, 58.05658742163783, 84.31323932952296, 80.75684041453633, 42.11615547909246, 47.307529034102004, 20.935296990766407, 18.72867435551481, 72.76102544571799, 95.9499986214951, 90.39736885148741, 93.29044629122504, 72.59174011417502, 53.91326256817881, 71.21520169759576, 40.865511998566205, 10.244719390278089, 97.29517358772111, 34.036939773320796, 98.24412258232161, 49.11514772120682, 96.35216004210166, 78.34049827260445, 80.07828451595267, 37.11461230033043, 16.316753039671518, 81.78966008248288, 4.716181248541796, 96.45269806087539, 94.7087284810188, 68.41491830182558, 42.27602024420418, 33.381563556953886, 60.46491322775821, 84.6540153612215, 55.10130582761186, 20.43303757929279, 95.27418700448746, 83.487973434652]
y_train = [263.2113012179828, 124.7539338699263, 209.7907256115055, 277.15755927118875, 109.0917562948565, 24.786363111022343, 118.20738037646663, 47.117315531596496, 234.78390617680466, 287.847963324212, 117.56602231005007, 159.35680025912305, 114.50394303679668, 234.30889846829209, 193.84513616613881, 214.69999115048896, 23.451501236019503, 65.15698894441478, 118.59010618005013, 150.0474445816199, 73.21942958434876, 133.61026008388376, 98.95839683562231, 197.84707554343507, 85.56485194327206, 88.29728483311251, 148.67737588347822, 267.66133699732194, 87.80994743448181, 92.26702003006261, 216.09742146853827, 77.07727490520757, 50.78240394777728, 120.84682742959028, 93.90947112587655, 197.5735599076025, 168.31089902709104, 168.52478044172963, 210.07225762864684, 250.66999655725144, 132.26578954002798, 245.36322443867704, 220.1021212748855, 82.11486999172882, 96.65610067581902, 97.52400770393662, 226.2434791648725, 172.96530641237052, 239.87986476636715, 157.8689536507441, 153.9126256248157, 36.254378495373004, 200.54078120296447, 203.7029902791599, 98.45647676810387, 177.43352891639748, 38.02666586998019, 142.23091639307748, 194.15575824335724, 174.02692906281868, 211.1785391942814, 232.48069144060523, 143.83464823748386, 202.2164990099384, 239.77710235641632, 176.1812920301733, 113.41617819338235, 44.995848451786316, 58.489429020354365, 169.65790947560063, 237.12442261225527, 261.86708832445623, 270.4237831195461, 152.20423974440993, 147.9420408485127, 170.3627701350062, 127.7651987119891, 103.43894855322391, 232.31902487674262, 168.0013417814573, 265.4812660432088, 171.33971889141804, 249.32223682023178, 229.09946243356063, 234.81602855493747, 92.59733771297837, 91.25787863177325, 166.51518682432408, 48.785130602325204, 211.5696119209247, 285.22867041871507, 213.5211990518296, 113.66667260015966, 107.06343756793352, 200.90661719236823, 174.56252985160668, 153.17575896165644, 116.11208115611736, 214.3731974289979, 227.04123305121323]
y_train_median = np.median(y_train)
y_train_labels = np.array([[1, 0] if y > y_train_median else [0, 1] for y in y_train])

# n = len(x1_train)

# red_x1 = [x1_train[i] for i in xrange(n) if y_train[i] > y_train_median]
# red_x2 = [x2_train[i] for i in xrange(n) if y_train[i] > y_train_median]
# blue_x1 = [x1_train[i] for i in xrange(n) if y_train[i] <= y_train_median]
# blue_x2 = [x2_train[i] for i in xrange(n) if y_train[i] <= y_train_median]

# plt.scatter(red_x1, red_x2, color='r')
# plt.scatter(blue_x1, blue_x2, color='b')
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.title("Test Data")
# plt.show()

x1_test = [9.190350405812708, 53.13151721867801, 86.21931195923936, 52.461721188432776, 67.51861271482369, 75.21438245002142, 44.96657317184416, 32.56545031964826, 10.806618674280877, 9.38575688882275, 2.624599527028171, 4.2089690309571, 17.220759254693064, 28.70297956029496, 77.71132405000102, 20.71520456772966, 88.16181250072064, 36.34989857177305, 15.754445282261875, 62.56347343947759, 86.66395918199873, 90.51578598165209, 80.83801686530873, 94.60708966490932, 39.138133143550476, 84.9523157901793, 7.63214989935157, 93.65279384613665, 73.9713194765022, 71.66025061252346, 61.60068120001353, 82.54539317835568, 61.335993814581, 43.70981333238284, 22.071363351177055, 50.73307437058471, 42.17527617815253, 17.688521668413472, 61.9405479706146, 50.39984791869132, 84.02299568498105, 58.75984603793826, 6.420354556254471, 34.26181180077492, 60.729439789202964, 16.962463353325795, 27.70454969733135, 35.55955322256214, 85.28226953755971, 72.56523228611927]
x2_test = [12.804446516144342, 88.26281448518169, 50.03450674994787, 83.26885465061518, 5.518889042517516, 11.218839790511337, 64.79384151260828, 78.31688984545255, 60.0555677539111, 88.50627750111887, 51.9942966038906, 44.239683799964055, 68.34949677732364, 70.60267999478911, 69.42887717031142, 76.46527433997623, 17.394904170773618, 88.5581448578912, 24.443790331383543, 32.98514958464517, 85.70670171165722, 35.921333222908395, 50.28245819985236, 94.20727499457749, 35.400984920297795, 0.7495309637527092, 77.66288798285845, 93.11940067926263, 94.44944569843167, 0.21394091422501926, 22.660353966357405, 23.697979935012402, 93.64132124077614, 56.77878647742368, 55.7746619223772, 37.633421477264726, 49.2115783426248, 33.24264869900211, 86.16889223551254, 23.89091664637072, 2.716859594801335, 48.267450818670454, 56.264414499525785, 91.45820318522219, 40.443194636712896, 56.02656699144345, 85.22458165095115, 31.706657685148222, 42.3777521401275, 19.149065064058135]
y_test = [34.96333118513873, 230.16232372893245, 186.36219721193171, 219.25281422497747, 78.7901250442087, 98.15457979272267, 175.26932582846064, 189.3444413633278, 131.139995553925, 186.4803183290494, 107.29740112385734, 92.83380248276534, 154.40050338429373, 170.33329454599004, 217.3792732793444, 173.86210845976424, 123.21276491880252, 214.00214135927078, 65.01403140833044, 128.93174222541668, 258.27743407218856, 162.36474155385903, 182.09320469893115, 283.5433339272224, 110.30165666252365, 86.90947725162279, 163.50290321907775, 280.3849350293204, 263.8434372083714, 72.84101685684804, 107.02115664101954, 130.02155609318797, 249.51032618719807, 157.50138082744317, 134.12007805234455, 126.89038763572846, 140.78886679834815, 84.90954353328897, 234.41977756803252, 99.13122152130966, 90.38805349201213, 155.5288531223009, 119.88077924567759, 217.87831590054213, 142.42370847541295, 129.46162324741795, 198.30109990293153, 99.90505582291202, 170.65088729227438, 111.1599263911343]
y_test_labels = np.array([[1, 0] if y > y_train_median else [0, 1] for y in y_test])

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

    def predictions(W_vect, inputs):
        outputs = 0
        for W, b in unpack_layers(W_vect):
            #print("W:", W, " Inputs[0]:", inputs[0])
            #prev_outputs = outputs
            outputs = np.dot(np.array(inputs), W) + b
            #inputs = outputs
            #print(outputs.shape)
            inputs = np.tanh(outputs)
            #inputs = np.array([np.array([sigmoid(i) for i in x]) for x in outputs])
        #print("Inputs:", inputs)
        return outputs

    def loss(W_vect, X, T):
        #print ("W:", W_vect)
        #print ("log_lik_terms", predictions(W_vect, X, alpha))
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        preds = predictions(W_vect, X)
        #print("W_vect:", W_vect)
        #print("preds: ", preds)
        #print("T:", T)
        normalised_log_probs = preds - logsumexp(preds)
        #print("normalised_log_probs: ", normalised_log_probs)
        log_lik = np.sum(normalised_log_probs * T)
        return log_prior + log_lik

    def frac_err(W_vect, X, T):
        #print ("Prediction:", predictions(W_vect, X, alpha))
        #print ("Prediction:", np.argmax(predictions(W_vect, X, alpha), axis=1), "Answer:", np.argmax(T, axis=1))
        percent_wrong = np.mean(np.argmax(T, axis=1) != np.argmax(predictions(W_vect, X), axis=1))
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


if __name__ == '__main__':
    # Network parameters
    layer_sizes = [12, 30, 3]
    L2_reg = 0.0#1.0

    # Training parameters
    param_scale = 0.1
    learning_rate = 1e-5
    momentum = 0.1
    #batch_size = len(train_images)
    num_epochs = 15000

    # Load and process MNIST data (borrowing from Kayak)
    #N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    N_data, train_images, train_labels, test_images, test_labels = get_wine_data()
    batch_size = len(train_images)
    #train_images, test_images = np.array(zip(x1_train, x2_train)), np.array(zip(x1_test, x2_test))
    #train_labels, test_labels = y_train_labels, y_test_labels

    # Make neural net functions
    N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg)
    loss_grad_W = grad(loss_fun, 0)

    # Initialize weights
    rs = npr.RandomState()
    W = rs.randn(N_weights) * param_scale

    # Check the gradients numerically, just to be safe
    # quick_grad_check(loss_fun, W, (train_images, train_labels))

    print("    Epoch      |    Train err  |   Test err  ")

    def print_perf(epoch, W):
        test_perf  = frac_err(W, test_images, test_labels)
        train_perf = frac_err(W, train_images, train_labels)
        print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

    # Train with sgd
    batch_idxs = make_batches(train_images.shape[0], batch_size)
    cur_dir_W = np.zeros(N_weights)

    for epoch in range(num_epochs):
        print_perf(epoch, W)
        for idxs in batch_idxs:
            grad_W = loss_grad_W(W, train_images[idxs], train_labels[idxs])
            cur_dir_W = momentum * cur_dir_W + (1.0 - momentum) * grad_W
            W += learning_rate * cur_dir_W
