import numpy as np

IntType = np.int64
FloatType = np.float64

class Dataset(object):
    """Dataset Dataset object of features and labels
        The label of the data is optinal

        Arguments:
            X {np.ndarray} -- features

        Keyword Arguments:
            y {np.ndarray} -- labels (default: {None})
            batch_size {int} -- size of mini-batch (default: {16})
            shuffle {bool} -- if True, shuffle the data (default: {False})
    """
    def __init__(self, X, y=None, batch_size=16, shuffle=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.size = X.shape[0] // batch_size
        if self.X.shape[0] % batch_size:
            self.size += 1
        if shuffle:
            index = np.random.permutation(X.shape[0])
            self.X = self.X[index]
            if self.y is not None:
                self.y = self.y[index]

    def __getitem__(self, i):
        if i == self.size - 1:
            features = self.X[i * self.batch_size:]
            if self.y is not None:
                labels = self.y[i * self.batch_size:]
        else:
            features = self.X[i * self.batch_size:(i + 1) * self.batch_size]
            if self.y is not None:
                labels = self.y[i * self.batch_size:(i + 1) * self.batch_size]

        if self.y is not None:
            return (features, labels)
        return features

    def __len__(self):
        return self.size


class Dataloader(object):
    """Dataloader A batch generator that iterates the whole dataset.
        The label of the data is optinal

        Arguments:
            X {np.ndarray} -- dataset

        Keyword Arguments:
            y {np.ndarray} -- labels (default: {None})
            batch_size {int} -- The size of each mini batch (default: {16})
            shuffle {bool} -- If True, shuffle the dataset everytime before iteration
    """
    def __init__(self, X, y=None, batch_size=16, shuffle=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.dataset = Dataset(self.X,
                               y=self.y,
                               batch_size=self.batch_size,
                               shuffle=self.shuffle)
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.dataset):
            batch = self.dataset[self.n]
            self.n += 1
            return batch
        else:
            raise StopIteration


def load_MNIST(path='./dataset/', name='train'):
    """load_MNIST Load MNIST data from csv file

    Keyword Arguments:
        path {str} -- The root directory of dataset (default: {'./dataset/'})
        name {str} -- train, val or test. (default: {'train'})

    Raises:
        ValueError: The name is not train, val or test.

    Returns:
        np.ndarray or tuple -- The dataset or a tuple of dataset and labels
    """

    name = name.lower()
    if name not in ['train', 'val', 'test']:
        raise ValueError("Only support train, val and test dataset")
    else:
        X = np.loadtxt('dataset/' + name + '_X.csv', delimiter=',')
        if name == 'test':
            return X
        else:
            Y = np.loadtxt('dataset/' + name + '_y.csv', delimiter=',')
            return X, Y.astype(IntType)


def one_hot_encoding(labels, num_class=None):
    """one_hot_encoding Create One-Hot Encoding for labels

    Arguments:
        labels {np.ndarray or list} -- The original labels

    Keyword Arguments:
        num_class {int} -- Number of classses. If None, automatically 
            compute the number of calsses in the given labels (default: {None})

    Returns:
        np.ndarray -- One-hot encoded version of labels
    """

    if num_class is None:
        num_class = np.max(labels) + 1
    one_hot_labels = np.zeros((len(labels), num_class))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels.astype(IntType)


def save_csv(x, filename="submission.csv"):
    """save_csv Save the input into csv file
    
    Arguments:
        x {np.ndarray} -- input array
    
    Keyword Arguments:
        filename {str} -- The file name (default: {"submission.csv"})
    
    Raises:
        ValueError: Input data structure is not np.ndarray
    """
    if isinstance(x, np.ndarray):
        x = x.flatten()
        np.savetxt(filename, x, delimiter=',')
    else:
        raise ValueError("The input is not an np.ndarray")
