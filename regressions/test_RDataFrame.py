from unittest import TestCase
from regressions import RDataFrame

rdf = RDataFrame.from_csv('Data/grunfeld.csv')

class TestRDataFrame(TestCase):

  def test_regress(self):
    r = rdf.regress('I ~ F + C')
    stata_results = dict(
        coefs=dict(F=.1155622, C=.2306785, Intercept=-42.71437),
        se=dict(F=.0058357, C=.0254758, Intercept=9.511676))

    for k in ['F', 'C', 'Intercept']:
        self.assertAlmostEqual(r.coefficients[k], stata_results['coefs'][k], places=5)