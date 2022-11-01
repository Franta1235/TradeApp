from abc import ABC
from Volatility.AVolatility import AVolatility
import numpy as np


class RealVolatility(AVolatility, ABC):

    def estimate(self, _data):
        _data = _data[0::5]  # takes 5min candles
        log_returns = np.log(_data['open']) - np.log(_data.shift(1)['open'])
        return np.sqrt(np.var(log_returns) * len(log_returns))
