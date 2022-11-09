from scipy.optimize import minimize
import datetime
import MySQL as db
import statistics
from scipy import stats
from ReplicatedPortfolio.AReplicatedPortfolio import AReplicatedPortfolio
from Helper import roundTimeUp, roundTimeDown, progressBar, daterange
from ReplicatedPortfolio.GeneralMerton import GeneralMerton


def run(x):
    rp = GeneralMerton(mu=0, gamma=x[0], q=x[1])
    symbol = 'LTCBTC'
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
    print(stats.ttest_1samp(results, 1)[0])
    return -stats.ttest_1samp(results, 1)[0]

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


x0 = [1, 0.8]
b0 = (0.2, 5)
b1 = (0.1, 0.95)
bnds = (b0, b1)

sol = minimize(run, x0, method='SLSQP', bounds=bnds)

print(f"gamma=  {sol['x'][1]}")
print(f"q=      {sol['x'][2]}")
