from __future__ import print_function
from unittest import TestCase
from regressions import RDataFrame
import pandas as pd

f = 'I ~ F + C'
grunfeld = 'regressions/Data/grunfeld.csv'


class TestRDataFrame(TestCase):
    def test_xtreg_nonrobust_coef(self):
        rdf = self.setup_rdf()
        with self.assertRaises(TypeError):
            rdf.xtreg(f)

        rdf.xtset('FIRM', 'YEAR')
        r = rdf.xtreg(f)
        stata_results = {
            'F':         .1101238,
            'C':         .3100653,
            'Intercept': -58.74393
        }

        for k in ['F', 'C', 'Intercept']:
            self.assertAlmostEqual(r.coefficients[k], stata_results[k], 4)

    def test_xtreg_nonrobust_se(self):
        rdf = self.setup_rdf()
        with self.assertRaises(TypeError):
            rdf.xtreg(f)

        rdf.xtset('FIRM', 'YEAR')
        r = rdf.xtreg(f, regression_type='fe')
        stata_results = {
            'F':         .0118567,
            'C':         .0173545,
            'Intercept': 12.45369
        }

        for k in ['F', 'C', 'Intercept']:
            print(k, r.se[k])
            self.assertAlmostEqual(r.se[k], stata_results[k], 4)

    def test_xtreg_robust(self):
        rdf = self.setup_rdf()
        rdf.xtset('FIRM', 'YEAR')
        r = rdf.xtreg(f, 'fe', vce='robust')
        stata_results = {
            'F':         .0151945,
            'C':         .0527518,
            'Intercept': 27.60286
        }

        for k in ['F', 'C', 'Intercept']:
            self.assertAlmostEqual(r.se[k], stata_results[k], 4)

    def setup_rdf(self):
        rdf = RDataFrame.from_csv(grunfeld)
        return rdf

    def test_from_csv(self):
        df = pd.read_csv(grunfeld)
        pan = RDataFrame.from_csv(grunfeld, i='FIRM', t='YEAR')
        for d, p in zip(list(pan.index), zip(list(df['FIRM']), list(df['YEAR']))):
            self.assertEqual(d[0], p[0])
            self.assertEqual(d[1], p[1])

    def test_keep_index(self):
        pan = RDataFrame.from_csv(grunfeld, i='FIRM', t='YEAR', keep_index=False)
        self.assertNotIn('FIRM', pan.columns)

    def test_nonrobust_coefs(self):
        rdf = self.setup_rdf()
        r = rdf.regress(f)
        stata_results = dict(
            coefs=dict(F=.1155622, C=.2306785, Intercept=-42.71437),
            se=dict(F=.0058357, C=.0254758, Intercept=9.511676))

        for k in ['F', 'C', 'Intercept']:
            self.assertAlmostEqual(r.coefficients[k], stata_results['coefs'][k], places=5)

    def test_nonrobust_se(self):
        rdf = self.setup_rdf()
        r = rdf.regress(f)
        stata_results = dict(
            coefs=dict(F=.1155622, C=.2306785, Intercept=-42.71437),
            se=dict(F=.0058357, C=.0254758, Intercept=9.511676))

        for k in ['F', 'C', 'Intercept']:
            self.assertAlmostEqual(r.se[k], stata_results['se'][k], places=5)

    def test_robust_se(self):
        rdf = self.setup_rdf()
        r = rdf.regress(f, vce='robust')
        stata_results = dict(
            coefs=dict(F=.1155622, C=.2306785, Intercept=-42.71437),
            se=dict(F=.006811, C=.0488655, Intercept=11.5747))

        for k in ['F', 'C', 'Intercept']:
            self.assertAlmostEqual(r.se[k], stata_results['se'][k], places=5)

    def test_cluster_assignment(self):
        rdf = self.setup_rdf()
        r = rdf.regress(f, cluster='FIRM')
        self.assertEqual(r.vce, 'cluster')
        with self.assertRaises(TypeError):
            # noinspection PyUnusedLocal
            r = rdf.regress(f, cluster=True)
        rdf2 = rdf.copy(deep=True)
        rdf2.xtset('FIRM', 'YEAR')
        r = rdf2.regress(f, cluster='FIRM')
        self.assertEqual(r.vce, 'cluster')

    def test_cluster_se(self):
        rdf = self.setup_rdf()
        r = rdf.regress(f, cluster='FIRM')
        stata_results = dict(
            coefs=dict(F=.1155622, C=.2306785, Intercept=-42.71437),
            se=dict(F=.0158943, C=.0849671, Intercept=20.4252))

        for k in ['F', 'C', 'Intercept']:
            self.assertAlmostEqual(r.se[k], stata_results['se'][k], places=5)

    def test_properties(self):
        rdf = self.setup_rdf()
        rdf.xtset('FIRM', 'YEAR')
        p = rdf.panel_attributes
        self.assertEqual(p.i, 'FIRM')
        self.assertEqual(p.t, 'YEAR')
        self.assertEqual(p.n, 10)
        self.assertEqual(p.T, 20)
        self.assertEqual(p.N, 200)
        self.assertTrue(rdf.balanced)

