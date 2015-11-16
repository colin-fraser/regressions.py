from unittest import TestCase
from regressions import RDataFrame
import pandas as pd

grunfeld = 'regressions/Data/grunfeld.csv'

rdf = RDataFrame.from_csv(grunfeld)


class TestRDataFrame(TestCase):
    def test_regress(self):
        r = rdf.regress('I ~ F + C')
        stata_results = dict(
            coefs=dict(F=.1155622, C=.2306785, Intercept=-42.71437),
            se=dict(F=.0058357, C=.0254758, Intercept=9.511676))

        for k in ['F', 'C', 'Intercept']:
            self.assertAlmostEqual(r.coefficients[k], stata_results['coefs'][k], places=5)

    def test_from_csv(self):
        df = pd.read_csv(grunfeld)
        pan = RDataFrame.from_csv(grunfeld, i='FIRM', t='YEAR')
        for d, p in zip(list(pan.index), zip(list(df['FIRM']), list(df['YEAR']))):
            self.assertEqual(d[0], p[0])
            self.assertEqual(d[1], p[1])

    def test_keep_index(self):
        pan = RDataFrame.from_csv(grunfeld, i='FIRM', t='YEAR', keep_index=False)
        self.assertNotIn('FIRM', pan.columns)
