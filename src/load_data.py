import autograd.numpy as np
import autograd.numpy.random as npr


def classification_data(task=0):
    data = np.load("./data/fmnist/binary_split_fmnist_"+str(task)+".npy", allow_pickle=True).tolist()
    x_train, y_train_ = data['train']['x'], data['train']['y']
    x_test, y_test_ = data['test']['x'], data['test']['y']
    x_train, x_test = x_train.reshape(-1,784), x_test.reshape(-1,784)
    mu = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train-mu)/std
    x_test = (x_test-mu)/std
    y_train = np.zeros((len(y_train_), len(np.unique(y_train_))))
    y_train[np.arange(len(y_train_)),y_train_]=1
    y_test = np.zeros((len(y_test_), len(np.unique(y_test_))))
    y_test[np.arange(len(y_test_)),y_test_]=1
    return x_train, y_train, x_test, y_test
    
"""
def classification_data(seed=0):

    #Load 2D data. 2 Classes. Class labels generated from a 2-2-1 network.
    #:param seed: random number seed
    #:return:

    npr.seed(seed)
    data = np.load("./data/covtype.npy", allow_pickle=True).tolist()
    x, y_ = data['X'], data['y']
    y = np.zeros((len(y_), len(np.unique(y_))))
    y[np.arange(len(y_)),y_]=1
    #data = np.load('./data/2D_toy_data_linear.npz')
    #x, y = data['x'], data['y']
    ids = np.arange(x.shape[0])
    npr.shuffle(ids)
    # 75/25 split
    num_train = int(np.round(0.8*x.shape[0]))
    x_train = x[ids[:num_train]]
    y_train = y[ids[:num_train]]
    x_test = x[ids[num_train:]]
    y_test =y[ids[num_train:]]
    mu = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train-mu)/std
    x_test = (x_test-mu)/std
    train_stats = dict()
    train_stats['mu'] = mu
    train_stats['sigma'] = std
    return x_train, y_train, x_test, y_test, train_stats


def regression_data(seed, data_count=500):

    #Generate data from a noisy sine wave.
    #:param seed: random number seed
    #:param data_count: number of data points.
    #:return:

    np.random.seed(seed)
    noise_var = 0.1

    x = np.linspace(-4, 4, data_count)
    y = 1*np.sin(x) + np.sqrt(noise_var)*npr.randn(data_count)

    train_count = int (0.2 * data_count)
    idx = npr.permutation(range(data_count))
    x_train = x[idx[:train_count], np.newaxis ]
    x_test = x[ idx[train_count:], np.newaxis ]
    y_train = y[ idx[:train_count] ]
    y_test = y[ idx[train_count:] ]

    mu = np.mean(x_train, 0)
    std = np.std(x_train, 0)
    x_train = (x_train - mu) / std
    x_test = (x_test - mu) / std
    mu = np.mean(y_train, 0)
    std = np.std(y_train, 0)
    y_train = (y_train - mu) / std
    train_stats = dict()
    train_stats['mu'] = mu
    train_stats['sigma'] = std

    return x_train, y_train, x_test, y_test, train_stats
"""