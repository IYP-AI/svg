import numpy as np
import random
import scipy


def head(p):
    return not np.isnan(p) and p > random.random()


def tail(p):
    return not head(p)


def H(q):
    return scipy.stats.entropy(np.abs(q) / np.abs(q).sum())


class Model:
    def p(self, x):
        print("Not Implemented: p(x)")

    def sample(self, x):
        print("Not Implemented: sample")

    def logp(self, x, eps=0.0):
        return np.log(max(eps, self.p(x)))


class MCMCSampler:
    def __init__(self, model, model_pi, n_batch=1):
        self.model = model
        self.model_pi = model_pi
        self.n_batch = n_batch

    def log_alpha(self, x, x_last):
        curr = self.model_pi.logp(x)
        last = self.model_pi.logp(x_last)
        backward = self.model.logp_transition(x, x_last)
        forward = self.model.logp_transition(x_last, x)

        alpha = curr - last + backward - forward
        print(
            f"alpha=e^{alpha:.3f}, p(x)=e^{curr:.3f}, p(x0)=e^{last:.3f}, "
            + f"p(x0<-x)=e^{backward:.3f}, p(x0->x)=e^{forward:.3f}"
        )

        return min(0, alpha) if not np.isnan(alpha) else alpha

    def alpha(self, x, x_last):
        return np.exp(self.log_alpha(x, x_last))

    def before_sample(self, xi_last):
        pass

    def sample(self, T=100, t_0=0, xi_0=None):
        xi_last = xi_0
        Omega = []
        for t in range(T):
            if t == 0:
                continue

            print(f"Step {t}")
            print(xi_last)
            self.before_sample(xi_last)

            xi = self.model.sample(xi_last)

            self.after_sample(xi_last)
            alpha = math.exp(self.log_alpha(xi, xi_last))
            if alpha != np.nan and alpha > random.random():
                xi_curr = xi
                print("ACCEPT")
            else:
                xi_curr = xi_last
                print("REJECT")

            if t > t_0:
                Omega.append(xi_curr)

            if t in [1, 3, 5, 10, 30, 50, 100]:
                print(xi)
            xi_last = xi_curr

        return Omega
