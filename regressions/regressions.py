import pandas as pd
import statsmodels.api as sm
from patsy.highlevel import dmatrices
import numpy as np
from collections import OrderedDict
from itertools import chain
from tabulate import tabulate

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

    @property
    def is_panel(self):
        return bool(self._i and self._t)

    @property
    def panel_attributes(self):
        self.check_panel()
        return PanelAttributes(self)

    @property
    def i(self):
        return self._i

    @property
    def t(self):
        return self._t

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
        return str(self.panel_attributes)


def fixed_effects_transform(df, idx):
    """
    Does the Stata secret sauce fixed-effects transform (not an official name), which demeans
    all the columns in df, and then adds in the "grand mean" of each. This allows for a constant 
    term within a fixed effects regression, as well as produces correct standard errors of the
    estimates. More detail here: 
    http://www.stata.com/support/faqs/statistics/intercept-in-fixed-effects-model/
    :type df: pd.DataFrame
    :param df: DataFrame (or RDataFrame or Series) to transform.
    :param idx: Index along which to perform the transformation
    :return: Transformed DataFrame (or similar)
    """
    df2 = df.groupby(level=idx).transform(lambda x: x - np.mean(x))
    return df2 + df.mean()


class Regression:
    def __init__(self, formula, data, regression_type='pooled', vce='nonrobust', cluster=False):
        """

        :type data: RDataFrame
        :param formula:
        :param data:
        :param regression_type:
        :param vce:
        :param cluster:
        :return:

        """

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
        self.coefficients = OrderedDict(zip(self.X.columns, self.fit.params))
        self.se = OrderedDict(zip(self.X.columns, self.fit.bse))
        self.p_values = OrderedDict(zip(self.X.columns, self.fit.pvalues))

    def _fit(self):

        model = sm.OLS(self.Y, self.X)
        if self.regression_type == 'fe':
            # Account for df loss from FE transform
            model.df_resid -= (self.data.panel_attributes.n - 1)
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
            idx = self.data.i
            Y, X = dmatrices(self.formula, self.data, return_type='dataframe')
            Y = fixed_effects_transform(Y, idx)
            X = fixed_effects_transform(X, idx)
            return Y, X
        else:
            raise ValueError('Regression type %s not implemented.' % self.regression_type)

    @property
    def summary(self):
        return self.fit.summary()

    def __repr__(self):
        return self.fit.summary.__repr__()


class RegressionTable:
    def __init__(self, regressions, coefficient_names=None, model_names=None):
        """
        :type regressions: list[Regression]
        :param regressions:
        :param coefficient_names: a dict mapping the coefficient names from regressions to
        the desired pretty-print names. Make it an OrderedDict to make coefficients show up in
        a desired order.
        :param model_names:
        :return:
        """

        self.regressions = regressions
        if model_names and len(model_names) != len(regressions):
            raise ValueError('model_names must be either None or of the same length as regressions')
        self.model_names = model_names or ['(%i)' % (k + 1) for k in range(len(regressions))]
        self.coefficient_names = coefficient_names
        self.extra_rows = []

    def table(self, tablefmt='pipe', digits=3):
        table_rows = self.make_rows(digits=digits) + self.extra_rows
        return tabulate(table_rows, headers='firstrow', tablefmt=tablefmt)

    def make_rows(self, digits=3):
        """
        Makes a list of lists to be rows for the table. The first entry in the list is the
        headers, and subsequent entries form the body of the table.

        :param digits: How many digits to round to in the output?
        :return: List of lists of strings
        """
        rows = [self.model_names]
        if not self.coefficient_names:
            coefs = list(set(chain(*[r.coefficients for r in self.regressions])))
        else:
            coefs = list(self.coefficient_names)

        for c in coefs:
            if self.coefficient_names:
                coef_row = [self.coefficient_names[c]]
            else:
                coef_row = [c]
            se_row = [' ']
            for r in self.regressions:
                if c not in r.coefficients:
                    coef_row.append(' ')
                    se_row.append(' ')
                else:
                    stars = ''
                    for p in [.05, .01, .001]:
                        if r.p_values[c] < p:
                            stars += '*'
                    coef_row.append(
                        '{coef:0.{digits}f}{stars}'.format(coef=r.coefficients[c], digits=digits,
                                                           stars=stars))
                    se_row.append('({coef:0.{digits}f})'.format(coef=r.se[c], digits=digits))
            rows.append(coef_row)
            rows.append(se_row)

        return rows


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
