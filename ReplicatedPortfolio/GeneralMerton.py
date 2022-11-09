from typing import Tuple

import pandas as pd
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt
from pandas import DataFrame

from AReplicatedPortfolio import AReplicatedPortfolio
from ReplicatedPortfolio.Window.GeneralMertonWindow import GeneralMertonWindow
from Volatility.RealVolatility import RealVolatility
from Volatility.EGARCH import EGARCH
from scipy.optimize import minimize
from scipy import stats


class GeneralMerton(AReplicatedPortfolio):
    sigma_p = None
    sigma_q = None
    sigma_est = None
    sigma_real = None
    param = {'mu': None, 'gamma': None, 'q': None}
    X = None
    uy = None
    uyx = None
    windows = []

    def __init__(self, mu=0, gamma=1, q=0.8):
        super().__init__()
        self.param = {'mu': mu, 'gamma': gamma, 'q': q}

    def create_window(self, prices) -> GeneralMertonWindow:
        self.X = pd.DataFrame(data={'price': prices['open'], 't': [(date.timestamp() % 86400) / 86400 for date in prices['time_from']]})
        data = pd.DataFrame({'price': self.X['price'], 'uY': self.uY(self.X)['uY'], 'uYx': self.uYx(self.X)['uYx']})
        self.uy = data['uY']
        self.uyx = data['uYx']
        log_return = math.log(prices.iloc[len(prices) - 1]['open'] / prices.iloc[0]['open'])
        date = prices.iloc[0]['time_from']
        window = GeneralMertonWindow(date=date, log_return=log_return, sigma_est=self.sigma_est, sigma_real=self.sigma_real, sigma_p=self.sigma_p, sigma_q=self.sigma_q, mu=self.param['mu'], gamma=self.param['gamma'])
        return window

    def update(self, prices) -> None:
        assert isinstance(prices, pd.DataFrame)
        try:
            if len(self.windows) < 200:
                raise ValueError('Not enough windows.')

            self.update_params()
            self.sigma_est = EGARCH().estimate(self.windows)
            self.sigma_real = RealVolatility().estimate(prices)
            self.sigma_q = self.sigma_est
            self.sigma_p = self.sigma_q * self.param['q']
            self.windows.append(self.create_window(prices))
        except:
            self.sigma_real = RealVolatility().estimate(prices)
            log_return = math.log(prices.iloc[len(prices) - 1]['open'] / prices.iloc[0]['open'])
            window = GeneralMertonWindow(date=prices.iloc[0]['time_from'], log_return=log_return, sigma_est=None, sigma_real=self.sigma_real, sigma_p=None, sigma_q=None, mu=None, gamma=None)
            self.windows.append(window)

    def uY(self, X) -> pd.DataFrame:
        isinstance(X, pd.DataFrame)
        W = np.log(X['price'] / X.iloc[0]['price'])

        uy0 = 1  # V(0)
        uy1 = np.sqrt(((self.param['gamma'] - 1) * (self.sigma_p ** 2) + self.sigma_q ** 2) / (((self.param['gamma'] - 1) + X['t'] / 1) * (self.sigma_p ** 2) + (1 - X['t'] / 1) * (self.sigma_q ** 2)))
        uy2 = np.exp((((self.sigma_p / self.sigma_q) ** 2 - 1) * ((W ** 2) / 2) + self.param['mu'] * W - 0.5 * self.param['mu'] * self.param['mu'] * (
                (self.sigma_q ** 2) / ((self.param['gamma'] - 1) * (self.sigma_p ** 2) + self.sigma_q ** 2)) * X['t']) / (
                             ((self.param['gamma'] - 1) + X['t'] / 1) * (self.sigma_p ** 2) + (1 - X['t'] / 1) * (self.sigma_q ** 2)))

        uy = uy0 * uy1 * uy2
        return pd.DataFrame(data={'t': X['t'], 'uY': uy})

    def uX(self, X) -> pd.DataFrame:
        pass

    def uYx(self, X) -> pd.DataFrame:
        isinstance(X, pd.DataFrame)
        W = np.log(X['price'] / X.iloc[0]['price'])

        h = ((self.param['gamma'] - 1) + X['t'] / 1) * (self.sigma_p ** 2) + (1 - X['t'] / 1) * (self.sigma_q ** 2)
        uy = self.uY(X)
        uyx = ((((self.sigma_p / self.sigma_q) ** 2 - 1) * W + self.param['mu']) / h) * uy['uY']

        return pd.DataFrame(data={'t': X['t'], 'uYx': uyx / X.iloc[0]['price']})

    def uXx(self, X) -> pd.DataFrame:
        pass

    def plot(self) -> None:
        """
        Plots merton portfolio in time
        :return: None
        """
        data = self.trade()[0]

        f = plt.figure(figsize=(30, 15))
        axes1 = f.add_subplot(111)
        plt.plot(data['time'], data['price'])
        plt.xticks(data['time'][0::int(len(data['time']) / 10)])

        buy = data[data['result'] > 0]
        sell = data[data['result'] < 0]
        plt.plot(buy['time'], buy['price'], '^', color='green', markersize=15)
        plt.plot(sell['time'], sell['price'], 'v', color='red', markersize=15)

        axes2 = axes1.twinx()
        axes2.set_ylabel(r'Hedging position $\Delta^X$(t)')
        plt.plot(data['time'], data['price_contract'], 'b')

        plt.show()

        # f.savefig("Data/Mean_Reversion.pdf", bbox_inches='tight')
        # input("Press Enter to continue...")

    def trade_theoretical(self) -> float:
        return self.uy.iloc[len(self.uy) - 1]

    def trade(self) -> tuple[DataFrame, int]:
        fee = 0.0015
        asset1 = self.uy.iloc[0]  # x
        asset2 = 0  # y
        self.X = self.X.reset_index(drop=True)
        last_position = 0
        data = []
        trades_count = 0
        for index, row in self.X.iterrows():
            price = row['price']

            diff = self.uyx.iloc[index] - last_position
            order = 0
            if abs(diff) > 4 / self.X.iloc[0]['price']:
                trades_count += 1
                order = diff
                if diff > 0:
                    asset1 -= diff * price
                    asset2 += diff * (1 - fee)
                else:
                    asset1 -= diff * price * (1 - fee)
                    asset2 += diff

                last_position = self.uyx.iloc[index]

            P = asset1 + asset2 * price
            data.append([row['t'], row['price'], P, self.uyx.iloc[index], order])

        order = asset2
        price = self.X.iloc[len(self.X) - 1]['price']
        if order > 0:
            trades_count += 1
            asset1 -= order * price
            asset2 += order * (1 - fee)
        elif order < 0:
            trades_count += 1
            asset1 -= order * price * (1 - fee)
            asset2 += order

        P = asset1 + asset2 * price
        data.append([1, price, P, 0, order])

        data = pd.DataFrame(data, columns=['time', 'price', 'price_contract', 'position', 'result'])
        return data, trades_count

    @staticmethod
    def pay_off(mu, gamma, x, sigma_p, sigma_q) -> float:
        y1 = ((gamma - 1) * (sigma_p ** 2) + sigma_q ** 2) / (gamma * (sigma_p ** 2))
        y2 = ((sigma_p ** 2 - sigma_q ** 2) / (2 * gamma * (sigma_p ** 2) * (sigma_q ** 2))) * (x ** 2)
        y3 = (mu * x) / (gamma * sigma_p ** 2)
        y4 = 0.5 * ((mu ** 2) / (gamma * (sigma_p ** 2))) * ((sigma_q ** 2) / ((gamma - 1) * (sigma_p ** 2) + sigma_q ** 2))
        return math.sqrt(y1) * math.exp(y2 + y3 - y4)

    @staticmethod
    def objective_function(x, data):
        isinstance(data, pd.DataFrame)
        mu = 0  # x[0]
        gamma = x[1]
        q = x[2]

        pay_off = []
        for index, row in data.iterrows():
            p = GeneralMerton.pay_off(mu=mu, gamma=gamma, x=row['log_return'], sigma_p=q * row['sigma'], sigma_q=row['sigma'])
            pay_off += [p]

        mean = statistics.mean(pay_off)
        var = statistics.variance(pay_off)
        result = {'Mean': mean, 'Variance': var, 't-statistic': stats.ttest_1samp(pay_off, 1)[0], 'p-value': stats.ttest_1samp(pay_off, 1)[1], 'N': len(pay_off)}
        return -result['t-statistic']

    def update_params(self) -> None:
        return

        if len(self.windows) % 50 > 0:
            return
        w = []
        for window in self.windows:
            assert isinstance(window, GeneralMertonWindow)
            w.append([window.log_return, window.sigma_real])
        w = pd.DataFrame(w, columns=['log_return', 'sigma'])

        x0 = [0, 0.5, 0.5]
        b1 = (-1, 1)
        b2 = (0.01, 2)
        b3 = (0.01, 1)
        bnds = (b1, b2, b3)

        sol = minimize(GeneralMerton.objective_function, x0, args=(w,), method='SLSQP', bounds=bnds)
        param = sol['x']
        print(param)
        self.param = {'mu': param[0], 'gamma': param[1], 'q': param[1]}
