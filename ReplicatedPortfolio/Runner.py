import datetime

import numpy as np

import MySQL as db
import statistics
import matplotlib.pyplot as plt
from scipy import stats
from ReplicatedPortfolio.AReplicatedPortfolio import AReplicatedPortfolio
from Helper import roundTimeUp, roundTimeDown, progressBar, daterange
from ReplicatedPortfolio.GeneralMerton import GeneralMerton
from ReplicatedPortfolio.Window.GeneralMertonWindow import GeneralMertonWindow


def run(symbol, rp):
    isinstance(symbol, str)
    isinstance(rp, AReplicatedPortfolio)

    candles = db.get_candles(symbol)
    start_date = roundTimeUp(candles.iloc[0]['time_from'])
    end_date = roundTimeDown(candles.iloc[len(candles) - 1]['time_from'])

    results = []
    trades = []

    for date in progressBar(list(daterange(start_date, end_date)), prefix=f'{symbol}: {start_date} - {end_date}:', suffix='Complete', length=50):
        prices = candles[(date <= candles['time_from']) & (candles['time_from'] < date + datetime.timedelta(1))]
        if len(prices) >= 1000:
            try:
                rp.update(prices)
                result, trades_count = rp.trade()
                trades.append(trades_count)
                results.append(result.iloc[len(results) - 1]['price_contract'])
                # rp.plot()
            except:
                pass

    # db.delete_replicated_portfolio(symbol)
    # trading_pair_id = db.get_id(symbol)

    # for window in rp.windows:
    #    assert isinstance(window, GeneralMertonWindow)
    #    db.import_to_replicated_portfolio(_symbol=symbol, _trading_pair_id=trading_pair_id, _date=window.date, _log_return=window.log_return, _sigma=window.sigma_real)

    return {
        'Mean': statistics.mean(results),
        'Variance': statistics.variance(results),
        't-statistic': stats.ttest_1samp(results, 1)[0],
        'p-value': stats.ttest_1samp(results, 1)[1],
        'N': len(results),
        'trades_max': max(trades),
        'trades_min': min(trades),
        'trades_avg': statistics.mean(trades)
    }


"""
pairs = db.get_trading_pairs()

for pair in pairs:
    try:
        rp = GeneralMerton(mu=0, gamma=1, q=0.5)
        res = run(symbol=f"{pair}", rp=rp)
        print(f"Mean =          {res['Mean']}")
        print(f"Variance =      {res['Variance']}")
        print(f"t-statistic =   {res['t-statistic']}")
        print(f"p-value =       {res['p-value']}")
        print(f"N=              {res['N']}")
        print(f"trades_max=     {res['trades_max']}")
        print(f"trades_min=     {res['trades_min']}")
        print(f"trades_avg=     {res['trades_avg']}")
    except:
        pass
"""
"""
for g in np.linspace(1.2, 3, 5):
    for q in np.linspace(0.6, 0.9, 5):
        print(f"gamma=  {g}")
        print(f"q=      {q}")
        rp = GeneralMerton(mu=0, gamma=g, q=q)
        res = run(symbol=f"LTCBTC", rp=rp)
        print(f"Mean =          {res['Mean']}")
        print(f"Variance =      {res['Variance']}")
        print(f"t-statistic =   {res['t-statistic']}")
        print(f"p-value =       {res['p-value']}")
        print(f"N=              {res['N']}")
        print(f"trades_max=     {res['trades_max']}")
        print(f"trades_min=     {res['trades_min']}")
        print(f"trades_avg=     {res['trades_avg']}")

"""
rp = GeneralMerton(mu=0, gamma=3, q=0.9)
res = run(symbol=f"LTCBTC", rp=rp)
print(f"Mean =          {res['Mean']}")
print(f"Variance =      {res['Variance']}")
print(f"t-statistic =   {res['t-statistic']}")
print(f"p-value =       {res['p-value']}")
print(f"N=              {res['N']}")
print(f"trades_max=     {res['trades_max']}")
print(f"trades_min=     {res['trades_min']}")
print(f"trades_avg=     {res['trades_avg']}")
