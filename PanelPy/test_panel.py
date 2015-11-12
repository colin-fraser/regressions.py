import panel
import pandas as pd
from unittest import TestCase


__author__ = 'colin.fraser'

df = pd.read_csv('../Data/grunfeld.csv')

class TestPanel(TestCase):

  def test___init__(self):
    self.fail()