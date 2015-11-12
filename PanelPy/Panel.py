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
        self.t = t

        if not keep_index:
            self.data.drop([i, t], axis=1, inplace=True)

        self.n, self.T = len(self.data.index.levels[0]), len(self.data.index.levels[1])
        self.N = self.data.shape[0]

    @property
    def balanced(self):
        return self.N == self.n * self.i

    def within(self, *args):
        """
        Performs the 'within transformation' on the variables chosen in *args. If no variables are selected,
        the transformation is performed on all variables
        :param args:
        :return:
        """
        within = lambda x: x - x.mean()
        if not args:
            return self.data.groupby(level=self.i).transform(within)
        return self.data[list(args)].groupby(level=self.i).transform(within)

    @property
    def panel_summary(self):
        stats = []
        balance = 'Balanced' if self.balanced else 'Unbalanced'
        balance += ' panel.'
        stats.append(balance)
        stats.append('Group variable: %s' % self.i)
        stats.append('Time variable: %s' % self.t)
        stats.append('n = %i' % self.n)
        stats.append('T = %i' % self.T)
        stats.append('N = %i' % self.N)
        return '\n'.join(stats)

    def __str__(self):
        return '\n\n'.join([self.panel_summary, self.data.__str__()])



df = pd.read_csv('../Data/grunfeld.csv')
p = Panel(df, 'FIRM', 'YEAR')