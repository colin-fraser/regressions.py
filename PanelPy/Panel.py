import pandas as pd
import numpy as np


class Panel:
    """

    """

    def __init__(self, data, i, t, keep_index=True):
        """

        :type data: pd.DataFrame
        :param data: a DataFrame
        :param i: the individual index
        :param t: the time index
        :param keep_index: should the indices stay in the data?
        :return: a new Panel
        """

        self.data = data.copy()
        self.data.index = [data[i], data[t]]
        self.i = i
        self.j = t

        if not keep_index:
            self.data.drop([i, t], axis=1, inplace=True)

        self.n, self.T = len(self.data.index.levels[0]), len(self.data.index.levels[1])
        self.N = self.data.shape[0]

    @property
    def balanced(self):
        return self.N == self.n * self.i

    def within(self, *args):
        if not args:
            return self.data.groupby(level=self.i).mean()
        return self.data[list(args)].groupby(level=self.i).mean()



df = pd.read_csv('../Data/grunfeld.csv')
p = Panel(df, 'FIRM', 'YEAR')