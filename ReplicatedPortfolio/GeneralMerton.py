import pandas as pd
import numpy as np
from AReplicatedPortfolio import AReplicatedPortfolio
from Window.GeneralMertonWindow import GeneralMertonWindow
from ReplicatedPortfolio.Window.AWindow import AWindow
from ReplicatedPortfolio.Window.GeneralMertonWindow import GeneralMertonWindow
from Volatility.RealVolatility import RealVolatility


class GeneralMerton(AReplicatedPortfolio):
    sigma_p = None
    sigma_q = None
    sigma_est = None
    sigma_real = None
    mu = None
    gamma = None
    windows = []

    def __init__(self, mu=0, gamma=1):
        super().__init__()
        self.mu = mu
        self.gamma = gamma

    def create_window(self, prices) -> GeneralMertonWindow:
        X = pd.DataFrame(data={'price': prices['open'], 't': [(date.timestamp() % 86400) / 86400 for date in prices['time_from']]})
        data = pd.DataFrame({'price': X['price'], 'uY': self.uY(X)['uY'], 'uYx': self.uYx(X)['uYx']})
        window = GeneralMertonWindow(data=data, sigma_est=self.sigma_est, sigma_real=self.sigma_real, sigma_p=self.sigma_p, sigma_q=self.sigma_q, mu=self.mu, gamma=self.gamma)
        return window

    def update(self, prices) -> None:
        self.sigma_est = RealVolatility().estimate(prices)  # TODO
        self.sigma_real = RealVolatility().estimate(prices)
        self.sigma_p = self.sigma_real
        self.sigma_q = self.sigma_p * 0.8
        self.mu = 0  # TODO
        self.gamma = 1  # TODO
        self.windows.append(self.create_window(prices))

    def uY(self, X) -> pd.DataFrame:
        isinstance(X, pd.DataFrame)
        W = np.log(X['price'] / X.iloc[0]['price'])

        uy0 = 1  # V(0)
        uy1 = np.sqrt(((self.gamma - 1) * (self.sigma_p ** 2) + self.sigma_q ** 2) / (((self.gamma - 1) + X['t'] / 1) * (self.sigma_p ** 2) + (1 - X['t'] / 1) * (self.sigma_q ** 2)))
        uy2 = np.exp((((self.sigma_p / self.sigma_q) ** 2 - 1) * ((W ** 2) / 2) + self.mu * W - 0.5 * self.mu * self.mu * (
                (self.sigma_q ** 2) / ((self.gamma - 1) * (self.sigma_p ** 2) + self.sigma_q ** 2)) * X['t']) / (
                             ((self.gamma - 1) + X['t'] / 1) * (self.sigma_p ** 2) + (1 - X['t'] / 1) * (self.sigma_q ** 2)))

        uy = uy0 * uy1 * uy2
        return pd.DataFrame(data={'t': X['t'], 'uY': uy})

    def uX(self, X) -> pd.DataFrame:
        pass

    def uYx(self, X) -> pd.DataFrame:
        isinstance(X, pd.DataFrame)
        W = np.log(X['price'] / X.iloc[0]['price'])

        h = ((self.gamma - 1) + X['t'] / 1) * (self.sigma_p ** 2) + (1 - X['t'] / 1) * (self.sigma_q ** 2)
        uy = self.uY(X)
        uyx = ((((self.sigma_p / self.sigma_q) ** 2 - 1) * W + self.mu) / h) * uy['uY']

        return pd.DataFrame(data={'t': X['t'], 'uYx': uyx / X.iloc[0]['price']})

    def uXx(self, X) -> pd.DataFrame:
        pass

    def plot(self) -> None:
        pass

    def trade_theoretical(self) -> float:
        window = self.windows[len(self.windows) - 1]
        isinstance(window, GeneralMertonWindow)
        return window.data.iloc[len(window.data) - 1]['uY']

    def trade(self) -> float:
        pass
