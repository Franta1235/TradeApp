from ReplicatedPortfolio.Window.AWindow import AWindow


class GeneralMertonWindow(AWindow):

    def __init__(self, log_return, sigma_est, sigma_real, sigma_p, sigma_q, mu, gamma):
        self.log_return = log_return
        self.sigma_est = sigma_est
        self.sigma_real = sigma_real
        self.sigma_p = sigma_p
        self.sigma_q = sigma_q
        self.mu = mu
        self.gamma = gamma
