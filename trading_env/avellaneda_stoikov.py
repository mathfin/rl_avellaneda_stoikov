import numpy as np

class AvellanedaStoikovModel:
    def __init__(self, gamma, k, T):

        if gamma > 0:
            self.gamma = gamma if gamma < 100 else 100
        else:
            self.gamma = 0.01

        if k > 0:
            self.k = k if k < 100 else 100
        else:
            self.k = 0.01

        self.T = T

    def reservation_price(self, s, disp, q, t):
        return s - q * self.gamma * disp * (self.T - t) / self.T  # Для нормализации T

    def optimal_spread(self):
        return (1 / self.gamma) * np.log(1 + self.gamma / self.k)
