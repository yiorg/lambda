import numpy as np
import pandas as pd


class Generator:
    def __init__(self, file_path, source):
        self.file_path = file_path
        self.source = source
        self.data = None
        self.file_to_array()

    def file_to_array(self):
        """
        Loads data into numpy array from file_path depending on the source, normalizes data
        using mean and standard deviation
        """
        if self.source == 'MESA':
            # get relevant data and normalize(MESA cols are act counts and psg sleep/wake)
            data = pd.read_csv(self.file_path, header=None, usecols=[2, 4], skiprows=1, delimiter=',').values
            mean = data[:, 0].mean(axis=0)
            data[:, 0] -= mean
            std = data[:, 0].std(axis=0)
            data[:, 0] /= std
        elif self.source == 'geneactiv':
            # TODO: geneactiv has no target data currently, need to add target data somehow
            # get relevant data and normalize
            data = pd.read_csv(self.file_path, header=None, usecols=[1, 2, 3], skiprows=100, delimiter=',').values
            mean = data.mean(axis=0)
            data -= mean
            std = data.std(axis=0)
            data /= std
        self.data = data

    def flow(self, look_back, delay, min_index, max_index, batch_size=128, step=1, shuffle=False):
        """
        Data generator for use with LSTM, yields samples and targets for each batch iteration

        :param look_back: How far to look back from current index, equivalent to window size (int)
        :param delay: how far forward(or backward) from current index to target (int)
        :param min_index: minimum index to be accessed in the data (default 0) (int)
        :param max_index: maximum index to be accessed in the data (default calculated if None) (int)
        :param batch_size: number of samples to be drawn (int)
        :param step: how many steps to take between drawing samples (int)
        :param shuffle: shuffles samples in batch randomly (boolean)
        """
        if max_index is None:
            if delay > 0:
                max_index = len(self.data) - delay - 1  # subtract delay(if positive) so we don't overshoot the index
            else:
                max_index = len(self.data) - 1
        idx = min_index + look_back
        while True:
            if shuffle:
                rows = np.random.randint(min_index + look_back, max_index, size=batch_size)
            else:
                if idx + batch_size >= max_index:
                    idx = min_index + look_back
                rows = np.arange(idx, min(idx + batch_size, max_index))
                idx += len(rows)
            samples = np.zeros((len(rows), look_back // step, self.data.shape[-1]-1))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - look_back, rows[j], step)
                samples[j] = self.data[:, 0:-1][indices]
                targets[j] = self.data[rows[j] + delay][1]
            yield samples, targets
