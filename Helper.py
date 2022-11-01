import datetime
import sys
import numpy as np


def daterange(start, end):
    for n in range(int((end - start).days)):
        yield start + datetime.timedelta(n)


def roundTimeUp(dt, round_to=86400):
    dt = dt.to_pydatetime()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to

    date = dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)

    if date < dt:
        return date + datetime.timedelta(1)
    else:
        return date


def roundTimeDown(dt, round_to=86400):
    dt = dt.to_pydatetime()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to

    date = dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)

    if date > dt:
        return date + datetime.timedelta(-1)
    else:
        return date


def progressBar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """

    total = len(iterable)

    # Progress Bar Printing Function
    def printProgressBar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()

    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()


def GeometricBrownMotion(X0, mu, sigma, T, N, K):
    dt = T / N
    paths = np.zeros((N + 1, K), np.float64)
    paths[0] = X0
    for t in range(1, N + 1):
        rand = np.random.standard_normal(K)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)

    return paths


def BrownMotion(T, N, K):
    dt = T / N
    paths = np.zeros((N + 1, K), np.float64)
    paths[0] = 0
    for t in range(1, N + 1):
        rand = np.random.standard_normal(K)
        paths[t] = paths[t - 1] + np.sqrt(dt) * rand

    return paths
