import numpy as np
from sklearn.model_selection import train_test_split

class DataGenerator:
    def __init__(self, n:int, m:int):
        if n.type != int or m.type != int:
            raise TypeError("n and m must be integers.")
        
        self.n = n
        self.m = m
        self.X = np.empty((self.n, self.m))
        self.Y = np.empty(self.n)
        self.gen = np.random.Generator(np.random.PCG64(0))

        #Escolha do modo de geração dos dados.
    def generate(self, type = "none", **kwargs):
        if type == "latent":
            return self.latent(**kwargs)
        elif type == "distribution":
            return self.distribution(**kwargs)
        elif type == "isotropic":
            return self.isotropic(**kwargs)

    #Geração de dados latentes.
    def latent(self, beta, scaling, noise_std):
        k = len(beta)
        aux = self.gen.standard_normal((self.m, k))
        q, r = np.linalg.qr(aux, mode='reduced')
        factor = (1 / scaling) * np.sqrt(self.m / k)
        w = factor * q
        
        z = self.gen.standard_normal((self.n, k))
        u = 1 / scaling * self.gen.standard_normal((self.n, self.m))
        e = self.gen.standard_normal(self.n)
        y = z @ beta + noise_std * e
        X = z @ w.T + u
        return X, y

    #Gerção de dados na qual X_i|Y ~ N(f1 * y, f2).
    def distribution(self, f1, f2):
        Y = np.random.normal(0, 1, self.n)
        mean = (f1 * Y)[:, None]
        self.X[:] = np.random.normal(mean, np.sqrt(f2), (self.n, self.m))
        self.Y[:] = Y
        return self.X, self.Y   

    #Geração de dados isotrópicos.
    def isotropic(self, beta=None, noise_std=1.0, beta_norm=2, scaling=1.0):
        X = np.random.normal(0, 1, (self.n, self.m)) / scaling
        beta = beta * beta_norm * (scaling / np.sqrt(self.m))
        self.X[:] = X
        error = np.random.normal(0, noise_std, self.n)
        self.Y[:] = X @ beta + noise_std * error
        return self.X, self.Y

    #Separação dos dados de treino e dados de teste.
    def train_data(self, proportion=0.5):
        idx = np.arange(self.n)
        self.train_idx, self.test_idx = train_test_split(idx, test_size=proportion, random_state=42)
        return (
            self.X[self.train_idx],
            self.X[self.test_idx],
            self.Y[self.train_idx],
            self.Y[self.test_idx]
        )