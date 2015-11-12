import Pandas as pd
import numpy as np


class Panel:
    """

    """

    def __init__(self, data, i, t):
        """

        :param data: a Data.Frame
        :param i: the individual index
        :param t: the time index
        :return: a new Panel
        """
        self.data = data
        self.i = i
        self.j = j
