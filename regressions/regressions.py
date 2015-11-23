import pandas as pd
import statsmodels.api as sm
from patsy.highlevel import dmatrices

panel_error = TypeError('Panel variables not set. Use RDataFrame.xtset(i, t).')
index_error = ValueError('\'i\', \'n\', \'N\', \'t\' and \'T\' are reserved column'
                         ' names in RDataFrames')


class PanelAttributes:
    def __init__(self, data):
        """

        :type data: RDataFrame
        :param data:
        :return:
        """
        self.i = data.i
        self.t = data.t
        self.n = len(data.index.levels[0])
        self.T = len(data.index.levels[1])
        self.N = data.shape[0]
        self.K = data.shape[1]

    def __repr__(self):
        attrs = ['i', 't', 'n', 'T', 'N', 'Width']
        values = [self.i, self.t, self.n, self.T, self.N, self.K]
        return '%s Panel.' % {True: 'Balanced', False: 'Unbalanced'}[self.balanced] + '\n' + \
               pd.DataFrame(dict(attribute=attrs, value=values)).to_string(index=False)

    @property
    def balanced(self):
        return self.N == self.n * self.T


class RDataFrame(pd.DataFrame):
    """
    A RDataFrame is a Pandas DataFrame with regression methods attached
    """

    def _constructor_expanddim(self):
        pass

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

    @property
    def _constructor(self):
        return RDataFrame

    def xtset(self, i, t, keep_index=True):
        self._i = i
        self._t = t
        # noinspection PyAttributeOutsideInit,PyAttributeOutsideInit
        self.index = [self[i], self[t]]
        if not keep_index:
            self.drop(labels=[i, t], axis=1, inplace=True)
        print(self.panel_attributes)

    @property
    def is_panel(self):
        return bool(self._i and self._t)

    @property
    def panel_attributes(self):
        self.check_panel()
        return PanelAttributes(self)

    @property
    def n(self):
        self.check_panel()
        return len(self.index.levels[0])

    # noinspection PyPep8Naming
    @property
    def T(self):
        self.check_panel()
        return len(self.index.levels[1])

    @property
    def i(self):
        self.check_panel()
        return self._i

    @property
    def t(self):
        self.check_panel()
        return self._t

    # noinspection PyPep8Naming
    @property
    def N(self):
        return self.data.shape[0]

    def regress(self, formula, vce='nonrobust', cluster=None, verbose=True):
        """
        Creates a pooled Regression object for this data with the formula given. For example,

           dt.regress('Y~X')

        will produce the same output as

            regress y x

        in Stata.

        :param verbose: Should a summary be printed on calling?
        :param formula: the formula for the regression
        :param vce: The Variance Covariance Estimator
        :param cluster: The name of the column along which to cluster for standard errors
        :return: Regression object
        :rtype: Regression
        """
        # Translate "robust" to HC1, which is the stata-style robust VCE:
        vce = 'HC1' if vce == 'robust' else vce
        result = Regression(formula, self, 'pooled', vce, cluster)
        if verbose:
            print(result.summary)
        return result

    # noinspection PyMethodOverriding
    @classmethod
    def from_csv(cls, path, i=None, t=None, keep_index=True, header=0, sep=','):

        data = pd.read_csv(path, sep=sep, header=header, index_col=None)
        return RDataFrame(data=data, i=i, t=t, keep_index=keep_index)

    def xtreg(self, formula, regression_type='fe', vce='nonrobust', cluster=None, verbose=True):
        self.check_panel()
        result = Regression(formula=formula, data=self, regression_type=regression_type, vce=vce,
                            cluster=cluster)
        if verbose:
            print(result.summary)
        return result

    @property
    def balanced(self):
        self.check_panel()
        return self.panel_attributes.balanced

    def check_panel(self):
        if not self.is_panel:
            raise panel_error

    @property
    def panel_summary(self):
        if not self.is_panel:
            raise panel_error
        stats = []
        balance = 'Balanced' if self.balanced else 'Unbalanced'
        balance += ' panel.'
        stats.append(balance)
        stats.append('Group variable: %s' % self.panel_attributes)
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

    # noinspection PyPep8Naming
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


def regress(data, formula, vce='nonrobust', cluster=None):
    """

    :param cluster: The name of the column along which to cluster standard errors
    :param vce: Variance-Covariance Estimator.
    :param formula: The patsy formula for the regression
    :param data: The RDataFrame object
    :return:
    """
    vce = 'HC1' if vce == 'robust' else vce
    return Regression(formula, data, regression_type='pooled', vce=vce, cluster=cluster)
