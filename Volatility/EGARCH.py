import math
import numpy as np
import pandas as pd
from abc import ABC
from Volatility.AVolatility import AVolatility
from statsmodels.formula.api import ols
from ReplicatedPortfolio.Window.AWindow import AWindow


class EGARCH(AVolatility, ABC):

    @staticmethod
    def prediction(windows, m=1, s=3, n=4):
        assert isinstance(windows, list)
        data = []

        for window in windows[(len(windows) - 200):len(windows)]:
            assert isinstance(window, AWindow)
            vol = window.sigma_real
            e = window.log_return
            #e = math.log(window.data['price'].iloc[len(window.data) - 1] / window.data['price'].iloc[0])
            data += [[vol, e]]

        data = pd.DataFrame(data, columns=['s', 'e'])

        # Create delay values
        for x in range(1, max(m, s, n) + 1):
            data[f's{x}'] = data['s'].shift(x)
            data[f'e{x}'] = data['e'].shift(x)

        # Drop NaN
        data = data.dropna()

        # Create regressors and model string
        data['target'] = np.log(data['s'] ** 2)
        model_string = "target ~ 1"
        for x in range(1, m + 1):
            data[f'alpha{x}'] = np.abs(data[f'e{x}'] / data[f's{x}'])
            model_string += f' + alpha{x}'

        for x in range(1, s + 1):
            data[f'beta{x}'] = np.log(data[f's{x}'] ** 2)
            model_string += f' + beta{x}'

        for x in range(1, n + 1):
            data[f'gamma{x}'] = data[f'e{x}'] / data[f's{x}']
            model_string += f' + gamma{x}'

        model = ols(model_string, data=data)
        results = model.fit()
        vol_predicted = math.sqrt(math.exp(results.predict(data.iloc[len(data) - 1])))
        return results, vol_predicted

    def estimate(self, windows, m=1, s=3, n=4):
        results, vol_predicted = EGARCH.prediction(windows, m, s, n)
        return vol_predicted

    @staticmethod
    def AIC(windows, m=1, s=3, n=4):
        results, vol_predicted = EGARCH.prediction(windows, m, s, n)
        return results.aic


"""
    @staticmethod
    def estimate_best_param():
        candles = get_candles("BTCUSDT")
        start_date = roundTimeUp(candles.iloc[0]['time_from'])
        data = candles[(start_date + datetime.timedelta(200) <= candles['time_from']) & (candles['time_from'] < start_date + datetime.timedelta(400))]

        aic_best = 9999
        best_param = []
        for m in range(5):
            for s in range(5):
                for n in range(5):
                    print([m, s, n])
                    aic = EGARCH.AIC(data, m, s, n)
                    if aic < aic_best:
                        aic_best = aic
                        best_param = [m, s, n]

        print(best_param)
        print(aic_best)
"""
