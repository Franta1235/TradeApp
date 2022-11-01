from abc import ABC
from Volatility.AVolatility import AVolatility
from Volatility.RealVolatility import RealVolatility


class DelayVolatility(AVolatility, ABC):

    def estimate(self, _data):
        return RealVolatility().estimate(_data)
