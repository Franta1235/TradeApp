import pandas as pd
from ReplicatedPortfolio.Window.AWindow import AWindow


class GeneralMertonWindow(AWindow):

    def __init__(self, data, sigma_est, sigma_real, sigma_p, sigma_q, mu, gamma):
        self.data = data
        self.sigma_est = sigma_est
        self.sigma_real = sigma_real
        self.sigma_p = sigma_p
        self.sigma_q = sigma_q
        self.mu = mu
        self.gamma = gamma
