import math
import statistics
from scipy.optimize import minimize
from scipy import stats
import MySQL as db
from ReplicatedPortfolio.GeneralMerton import GeneralMerton

"""
def pay_off_general_merton(mu, gamma, x, sigma_p, sigma_q) -> float:
    y1 = ((gamma - 1) * (sigma_p ** 2) + sigma_q ** 2) / (gamma * (sigma_p ** 2))
    y2 = ((sigma_p ** 2 - sigma_q ** 2) / (2 * gamma * (sigma_p ** 2) * (sigma_q ** 2))) * (x ** 2)
    y3 = (mu * x) / (gamma * sigma_p ** 2)
    y4 = 0.5 * ((mu ** 2) / (gamma * (sigma_p ** 2))) * ((sigma_q ** 2) / ((gamma - 1) * (sigma_p ** 2) + sigma_q ** 2))

    return math.sqrt(y1) * math.exp(y2 + y3 - y4)


def general_merton(_data, mu, gamma, q):
    pay_off = []
    for index, row in _data.iterrows():
        p = pay_off_general_merton(mu=mu, gamma=gamma, x=row['log_return'], sigma_p=q * row['sigma'], sigma_q=row['sigma'])
        pay_off += [p]

    mean = statistics.mean(pay_off)
    var = statistics.variance(pay_off)
    return {'Mean': mean, 'Variance': var, 't-statistic': stats.ttest_1samp(pay_off, 1)[0], 'p-value': stats.ttest_1samp(pay_off, 1)[1], 'N': len(pay_off)}


def objective(x, _data):
    mu = x[0]
    gamma = x[1]
    q = x[2]
    return -general_merton(_data, mu, gamma, q)['t-statistic']

"""

data = db.get_replicated_portfolio('ETHBTC')

x0 = [0, 0.5, 0.5]
b1 = (-1, 1)
b2 = (0.01, 2)
b3 = (0.01, 1)
bnds = (b1, b2, b3)

sol = minimize(GeneralMerton.objective_function, x0, args=(data,), method='SLSQP', bounds=bnds)
print(sol['x'])
