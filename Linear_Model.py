import numpy as np
import cvxpy as cp
from sklearn.linear_model import Lasso, Ridge

class LinearModel:
    def __init__(self, intercept=False):
        self.intercept = intercept

    def beta_separation(self, beta):
        if(self.intercept == True):
            beta0 = beta[0]
            beta1 = beta[1:]
        else:
            beta0 = 0
            beta1 = beta
        return beta0, beta1

    #Estimação de beta via OLS.
    def estimate_beta(self, X, Y):
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        return beta

    #Estimação de beta via LASSO
    def lasso_beta(self, X, Y, alpha = 1.0):
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
        lasso.fit(X, Y)
        return lasso.coef_

    #Estimação de beta via regressão de ridge.
    def ridge_beta(self, X, Y, alpha = 1.0):
        ridge = Ridge(alpha=alpha, fit_intercept=False, max_iter=10000)
        ridge.fit(X, Y)
        return ridge.coef_    

    #Estimação do risco empírico.
    def estimate_risk(self, X, Y, beta):
        Y_pred = X @ beta
        risk = np.mean((Y - Y_pred) ** 2)
        return risk

    #Estimação do risco adversário empírico.
    def estimate_adv_risk(self, X, Y, beta, delta, attack = None, p=np.inf):
        #Caso nenhum ataque seja provido, é utilizado o ataque de máxima pertubação.
        if attack is None:
            attack = self.optimal_attack(beta, X, Y, p=p)

        beta0, beta1 = self.beta_separation(beta)
        
        X_adv = X + delta * attack
        Y_pred_adv = beta0 + X_adv @ beta1
        adv_risk = np.mean((Y - Y_pred_adv) ** 2)
        return adv_risk

    #Cálculo do risco adversário teórico.
    def max_adversarial_risk(self, X, Y, beta, risk, delta, p=np.inf):
        beta0, beta1 = self.beta_separation(beta)

        if p == np.inf:
            q = 1
        elif p == 1:
            q = np.inf
        else:
            q = p / (p - 1)

        norm = np.linalg.norm(beta1, q)
        adv_risk = risk + delta**2 * norm ** 2 + 2 * delta * norm * np.mean(np.abs(beta1))
        return adv_risk 

    #Cálculo do ataque adversário de máxima pertubação.
    def optimal_attack(self, beta, X, Y, p=np.inf):
        beta0, beta1 = self.beta_separation(beta)

        error = X @ beta1 + beta0 - Y
        error_sign = np.sign(error)

        if p == np.inf:
            attack = np.sign(beta1)[None, :]
        elif p == 1:
            attack = np.zeros((len(error_sign), len(beta1)))
            max_idx = np.where(np.abs(beta1) == np.max(np.abs(beta1)))[0]
            attack[:, max_idx] = 1 / len(max_idx)
        else:
            q = p / (p - 1)
            attack = np.sign(beta1) * np.abs(beta1) ** (q / p)
            attack = attack[None, :]

        attack = attack / np.linalg.norm(attack, ord=p, axis=1, keepdims=True)
        attack = attack * error_sign[:, None]

        return attack

    #Setup para a estimação de beta via minimização do risco adversário dos dados de treino.
    def setup_min_advrisk_beta(self, X, Y, p=np.inf):
        if p == np.inf:
            q = 1
        elif p == 1:
            q = np.inf
        else:
            q = p / (p - 1)

        n, m = X.shape

        self.X = cp.Parameter((n, m))
        self.Y = cp.Parameter(n)
        self.delta = cp.Parameter(nonneg=True)

        self.beta_0 = cp.Variable()
        self.beta = cp.Variable(m)

        self.X.value = X
        self.Y.value = Y

        if q == 1:
            beta_norm = cp.norm1(self.beta)
        elif q == 2:
            beta_norm = cp.norm2(self.beta)
        elif q == np.inf:
            beta_norm = cp.norm_inf(self.beta)
        else:
            beta_norm = cp.pnorm(self.beta, p=q)

        abs_error = cp.abs(self.Y - self.beta_0 - self.X @ self.beta)
        adversarial_residual = (abs_error + self.delta * beta_norm) ** 2
        objective = cp.Minimize(cp.sum(adversarial_residual) / n)
        self.problem = cp.Problem(objective)

    #Estimação do beta via minimização do risco adversário dos dados de treino.
    def solve_min_advrisk_beta(self, delta, warm_start=False):
        self.delta.value = delta
        try:
            self.problem.solve(solver=cp.CLARABEL, warm_start=warm_start)

            if self.problem.status in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE):
                return np.r_[self.beta_0.value,self.beta.value]

            return np.zeros(len(self.beta.value) + 1)

        except cp.error.SolverError:
            return np.zeros(len(self.beta.value) + 1)