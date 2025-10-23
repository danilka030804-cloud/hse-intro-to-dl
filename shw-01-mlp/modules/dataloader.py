import numpy as np

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0], "X and y must have same number of samples"
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # index of current batch
        self.indices = np.arange(X.shape[0])  # индексируем данные

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return int(np.ceil(self.X.shape[0] / self.batch_size))

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        self.batch_id = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        if self.batch_id >= len(self):
            raise StopIteration

        start = self.batch_id * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start:end]

        x_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        self.batch_id += 1
        return x_batch, y_batch
