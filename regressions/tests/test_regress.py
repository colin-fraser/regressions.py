from unittest import TestCase
import pandas as pd
from regressions import *

df = pd.read_csv('Data/grunfeld.csv')


class TestRegress(TestCase):

    def test_nonrobust_coefs(self):
        r = regress(df, 'I ~ F + C')
        stata_results = dict(
            coefs=dict(F=.1155622, C=.2306785, Intercept=-42.71437),
            se=dict(F=.0058357, C=.0254758, Intercept=9.511676))

        for k in ['F', 'C', 'Intercept']:
            self.assertAlmostEqual(r.coefficients[k], stata_results['coefs'][k], places=5)

    def test_nonrobust_se(self):
        r = regress(df, 'I ~ F + C')
        stata_results = dict(
            coefs=dict(F=.1155622, C=.2306785, Intercept=-42.71437),
            se=dict(F=.0058357, C=.0254758, Intercept=9.511676))

        for k in ['F', 'C', 'Intercept']:
            self.assertAlmostEqual(r.se[k], stata_results['se'][k], places=5)

    def test_robust_se(self):
        r = regress(df, 'I ~ F + C', robust)
        stata_results = dict(
            coefs=dict(F=.1155622, C=.2306785, Intercept=-42.71437),
            se=dict(F=.006811, C=.0488655, Intercept=11.5747))

        for k in ['F', 'C', 'Intercept']:
            self.assertAlmostEqual(r.se[k], stata_results['se'][k], places=5)

    def test_cluster_se(self):
        r = regress(df, 'I ~ F + C', cluster='FIRM')
        stata_results = dict(
            coefs=dict(F=.1155622, C=.2306785, Intercept=-42.71437),
            se=dict(F=.0158943, C=.0849671, Intercept=20.4252))

        for k in ['F', 'C', 'Intercept']:
            self.assertAlmostEqual(r.se[k], stata_results['se'][k], places=5)