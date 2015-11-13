import pandas as pd
import numpy as np
import statsmodels.api as sm


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



class Regression:
    """
    A regression is attached to a panel
    """

    def __init__(self, formula, data, type, std_err, cluster):
        self.formula = formula
        self.data = data
        self.type = type
        self.std_err = std_err
        self.cluster = cluster


def xtreg(dep, indep, panel, type='fe', robust_se=None, cluster=None):

    cov_type = robust_se or 'nonrobust'
    cov_kwds = cluster or {'groups': panel.data.index.labels[0]}

    if type == 'fe':
        panel = fixed_effects_transform(dep, indep, panel)
        return sm.OLS(panel['Y'], panel['X']).fit(cov_type=cov_type, cov_kwds=cov_kwds)

def fixed_effects_transform(dep, indep, data):
    """
    Does stata-style fixed effects where the intercept is the average fixed effect size
    :param dep: the name of the dependent variable
    :param indep: the names of the independent variables
    :param data: the data
    :type data: Panel
    :return:
    """
    grand_means = data.grand_means(dep, *indep)
    Y = data.within(dep) + grand_means[dep]

    # Build X
    X = data.within(*indep)
    X += np.ones(X.shape) * grand_means[indep].data
    X = sm.add_constant(X, prepend=False)

    return {'Y': Y, 'X': X}



df = pd.read_csv('../Data/grunfeld.csv')
p = Panel.from_csv('../Data/grunfeld.csv', 'FIRM', 'YEAR')

