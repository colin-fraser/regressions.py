import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy.highlevel import dmatrices


class Panel:
    """
    A panel has a data frame and an index, with methods for doing panel things.
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
    def values(self):
        return self.data.values

    @property
    def balanced(self):
        return self.N == self.n * self.T

    def within(self, *args, values_only=False):
        """
        Performs the 'within transformation' on the variables chosen in *args. If no variables are selected,
        the transformation is performed on all variables
        :param args: the columns to be transformed
        :param values_only: If this is true, only the matrix of values will be returned
        :return:
        """
        within = lambda x: x - x.mean()
        if not args:
            return self.data.groupby(level=self.i).transform(within)
        return self.data[list(args)].groupby(level=self.i).transform(within)

    def between(self, *args, values_only=False):
        """
        Performs the 'between transformation' on the variables chosen in *args. If no variables are selected,
        the transformation is performed on all variables
        :param args: columns to be transformed
        :param values_only: If this is true, only the matrix of values will be returned
        :return:
        """

        if not args:
            transformed = self.data.groupby(levels=self.t).transform(lambda x: x - x.mean())
        else:
            transformed = self.data[list(args)].groupby(level=self.t).transform(lambda x: x - x.mean())

        return transformed.values if values_only else transformed

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

    def __repr__(self):
        return self.panel_summary + '\n\n' + self.data.__str__()

    def __str__(self):
        return '\n\n'.join([self.panel_summary, self.data.__str__()])

    def grand_means(self, *args):
        if not args:
            return np.mean(self.data)
        else:
            return np.mean(self.data[list(args)])

    @classmethod
    def from_csv(cls, file, i, t, keep_index=True):
        """

        :param file: the file to import
        :param i: the individual index column
        :param t: the time index column
        :return: data.frame
        """

        df = pd.read_csv(file)
        return Panel(df, i, t, keep_index)

    def xtreg(self, formula=None, dep=None, indep=None, type='fe', robust_se=None, cluster=None):

        cov_kwds = None
        if formula:
            if dep or indep:
                raise ValueError('Cannot specify both formula and dep/indep lists')
            Y, X = dmatrices(formula, self.data, return_type='dataframe')
        cov_type = robust_se or 'nonrobust'

        if cov_type == 'cluster':
            cov_kwds = cluster or {'groups': self.data.index.labels[0]}

        if type == 'fe':
            Y = fixed_effects_transform(Y, self.i)
            X = fixed_effects_transform(X, self.i)

        return sm.OLS(Y, X).fit(cov_type=cov_type, cov_kwds=cov_kwds)


class RDataFrame(pd.DataFrame):
    """
    A RDataFrame is a Pandas DataFrame with regression methods attached
    """

    def __init__(self, data, index, columns, dtype, copy):
        """

        :return:
        """
        super().__init__(self, data, index, columns, dtype, copy)

    @property
    def _constructor(self):
        return RDataFrame

    def regress(self, formula, *args, **kwargs):
        return regress(self, formula, *args, **kwargs)


def fixed_effects_transform(df, idx):
    return df.groupby(level=idx).transform(lambda x: x - x.mean()) + df.mean()

robust = 'robust'
cluster = 'cluster'


class Regression:

    def __init__(self, formula, data, *args, type='pooled', **kwargs):

        self.formula = formula
        self.data = data
        self.type = type
        self.cluster = False
        self.args = args
        self.__dict__.update(kwargs)
        self.Y, self.X = dmatrices(formula, data)
        self.fit = self._fit()
        self.coefficients = dict(zip(self.X.design_info.term_names, self.fit.params))
        self.se = dict(zip(self.X.design_info.term_names, self.fit.bse))

    def _fit(self):
        if self.type == 'pooled':
            if robust in self.args:
                return sm.OLS(self.Y, self.X).fit(cov_type='HC1')
            if self.cluster:
                cluster_variable = self.data.ix[:, self.cluster]
                return sm.OLS(self.Y, self.X).fit(cov_type='cluster', cov_kwds={'groups': cluster_variable})
            else:
                return sm.OLS(self.Y, self.X).fit(cov_type='nonrobust')

    @property
    def summary(self):
        return self.fit.summary()

    def __repr__(self):
        return self.fit.summary.__repr__()





def regress(data, formula, *args, **kwargs):
    """

    :param formula:
    :param data:
    :param args:
    :return:
    """

    return Regression(formula, data, *args, type='pooled', **kwargs)

