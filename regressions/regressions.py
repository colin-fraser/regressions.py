import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy.highlevel import dmatrices

robust = 'robust'
cluster = 'cluster'


class RDataFrame(pd.DataFrame):
    """
    A RDataFrame is a Pandas DataFrame with regression methods attached
    """

    panel_error = TypeError('Panel variables not set. Use RDataFrame.xtset(i, t).')
    index_error = ValueError('\'i\', \'n\', \'N\', \'t\' and \'T\' are reserved column'
                             ' names in RDataFrames')

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False,
                 i=None, t=None, keep_index=True):
        """

        :return:
        """
        super().__init__(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self._i = None
        self._t = None
        if i and t:
            self.xtset(i, t, keep_index)

    def xtset(self, i, t, keep_index=True):
        self._i = i
        self._t = t
        # noinspection PyAttributeOutsideInit,PyAttributeOutsideInit
        self.index = [self[i], self[t]]
        if not keep_index:
            self.drop(labels=[i, t], axis=1, inplace=True)

    @property
    def _constructor(self):
        return RDataFrame

    @property
    def is_panel(self):
        return self._i and self._t

    @property
    def n(self):
        if not self.is_panel:
            raise self.panel_error
        return len(self.index.levels[0])

    @property
    def T(self):
        if not self.is_panel:
            raise self.panel_error
        return len(self.index.levels[1])

    @property
    def i(self):
        if not self.is_panel:
            raise self.panel_error
        return self._i

    @property
    def t(self):
        if not self.is_panel:
            raise self.panel_error
        return self._t

    @property
    def N(self):
        return self.data.shape[0]

    def regress(self, formula, vce='nonrobust', cluster=None):
        """
        Creates a pooled Regression object for this data with the formula given. For example,

           dt.regress('Y~X')

        will produce the same output as

            regress y x

        in Stata.

        :param formula: the formula for the regression
        :param vce: The Variance Covariance Estimator
        :param cluster: The name of the column along which to cluster for standard errors
        :return: Regression object
        :rtype: Regression
        """
        # Translate "robust" to HC1, which is the stata-style robust VCE:
        vce = 'HC1' if vce == 'robust' else vce
        return Regression(formula, self, 'pooled', vce, cluster)

    @classmethod
    def from_csv(cls, path, i=None, t=None, keep_index=True, header=0, sep=','):

        data = pd.read_csv(path, sep=sep, header=header, index_col=None)
        return RDataFrame(data=data, i=i, t=t, keep_index=keep_index)

    def xtreg(self, formula, regression_type='fe', vce='nonrobust', cluster=None):
        if not self.is_panel:
            raise self.panel_error
        return Regression(formula=formula, data=self, regression_type=regression_type, vce=vce,
                          cluster=cluster)

    @property
    def panel_summary(self):
        if not self.is_panel:
            raise TypeError('Not a panel. Run RDataFrame.xtset(i, t) to set up panel variables.')
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


def fixed_effects_transform(df, idx):
    return df.groupby(level=idx).transform(lambda x: x - x.mean()) + df.mean()


class Regression:
    def __init__(self, formula, data, regression_type='pooled', vce='nonrobust', cluster=False):

        self.formula = formula
        self.data = data
        self.regression_type = regression_type
        self.vce = vce if not cluster else 'cluster'

        if regression_type == 'fe' and vce == 'robust' and not cluster:
            self.cluster = True
            self.vce = 'cluster'
        else:
            self.cluster = cluster

        self.Y, self.X = self._set_XY()
        self.fit = self._fit()
        self.coefficients = dict(zip(self.X.design_info.term_names, self.fit.params))
        self.se = dict(zip(self.X.design_info.term_names, self.fit.bse))

    def _fit(self):

        model = sm.OLS(self.Y, self.X)
        if self.regression_type == 'fe':
            model.df_resid -= (self.data.n - 1)  # Account for df loss from FE transform
        if True is self.cluster:
            if not self.data.is_panel:
                raise TypeError('Cannot infer cluster variable because panel variables '
                                'are not set. Run RDataTable.xtset(i, t) to convert data '
                                'to Panel format.')
            cov_kwds = {'groups': self.data[self.data.i]}
        elif self.cluster:
            cov_kwds = {'groups': self.data[self.cluster]}
        else:
            cov_kwds = None

        return model.fit(cov_type=self.vce, cov_kwds=cov_kwds)

    def _set_XY(self):
        if self.regression_type == 'pooled':
            return dmatrices(self.formula, self.data, return_type='dataframe')
        elif self.regression_type == 'fe':
            return dmatrices(self.formula, fixed_effects_transform(self.data, self.data.i),
                             return_type='dataframe')
        else:
            raise ValueError('Regression type %s not implemented.' % self.regression_type)

    @property
    def summary(self):
        return self.fit.summary()

    def __repr__(self):
        return self.fit.summary.__repr__()


def regress(data, formula, *args, vce='nonrobust', cluster=None):
    """

    :param cluster: The name of the column along which to cluster standard errors
    :param vce: Variance-Covariance Estimator.
    :param formula: The patsy formula for the regression
    :param data: The RDataFrame object
    :param args:
    :return:
    """
    vce = 'HC1' if vce == 'robust' else vce
    return Regression(formula, data, regression_type='pooled', vce=vce, cluster=cluster)
