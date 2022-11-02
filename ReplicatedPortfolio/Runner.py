import datetime
import MySQL as db
import statistics
from scipy import stats
from ReplicatedPortfolio.AReplicatedPortfolio import AReplicatedPortfolio
from Helper import roundTimeUp, roundTimeDown, progressBar, daterange
from ReplicatedPortfolio.GeneralMerton import GeneralMerton


def run(symbol, rp):
    isinstance(symbol, str)
    isinstance(rp, AReplicatedPortfolio)

    candles = db.get_candles(symbol)
    start_date = roundTimeUp(candles.iloc[0]['time_from'])
    end_date = roundTimeDown(candles.iloc[len(candles) - 1]['time_from'])

    results = []

    for date in progressBar(list(daterange(start_date, end_date)), prefix=f'{symbol}:', suffix='Complete', length=50):
        prices = candles[(date <= candles['time_from']) & (candles['time_from'] < date + datetime.timedelta(1))]
        if len(prices) == 1440:
            try:
                rp.update(prices)
                result = rp.trade()
                results.append(result)
            except:
                pass

    mean = statistics.mean(results)
    var = statistics.variance(results)

    return {'Mean': mean, 'Variance': var, 't-statistic': stats.ttest_1samp(results, 1)[0], 'p-value': stats.ttest_1samp(results, 1)[1], 'N': len(results)}


rp = GeneralMerton(mu=-0.00725371, gamma=0.82963071, q=0.38189155)
# rp = GeneralMerton(mu=0, gamma=1, q=0.8)
res = run(symbol='ETHBTC', rp=rp)
print(f"Mean =          {res['Mean']}")
print(f"Variance =      {res['Variance']}")
print(f"t-statistic =   {res['t-statistic']}")
print(f"p-value =       {res['p-value']}")
print(f"N=              {res['N']}")
