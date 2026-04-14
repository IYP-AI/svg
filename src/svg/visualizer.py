import visdom
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import time

from collections import namedtuple
from probabilistic import H
from .utils import *

LogField = namedtuple("LogField", ("data", "plot", "x_axis", "divide_by"))


class Visualizer:
    # Initialize Visdom environment
    def __init__(self, sampler, plot_env=None, port=3000):

        if plot_env is None or plot_env == "":
            envname = f"{sampler.model_pi.task}_v1-{random.choice(range(10000))}"
        else:
            envname = plot_env

        log = dict()
        log["epoch"] = LogField(list(), False, None, None)
        log["accuracy"] = LogField(list(), True, "epoch", None)
        log["entropy"] = LogField(list(), True, "epoch", None)
        log["density"] = LogField(list(), True, "epoch", None)
        log["n_samples"] = LogField(list(), True, "epoch", None)
        log["dimension"] = LogField(list(), True, "epoch", None)
        log["log_c"] = LogField(list(), True, "epoch", None)
        log["time"] = LogField(list(), True, "epoch", None)
        log["mean_accuracy"] = LogField(list(), True, "epoch", None)

        self.time0 = time.time()

        self.log = log
        self.sampler = sampler
        self.envname = envname

        print(f"Env: {envname}")
        self.vis = visdom.Visdom(port=port, env=envname)

    def dump(self, Omega_parallel, t, figscale=5):
        print(f"Env: {self.envname}")

        for k, v in self.log.items():
            v.data.append(t if k == "epoch" else [])

        # Creating the figure and subplots
        n = len(Omega_parallel)
        plt.style.use("ggplot")
        fig, ax = plt.subplots(1, n, figsize=(figscale * n, figscale))
        if n == 1:
            ax = [ax]

        self.log["time"].data[-1].append((time.time() - self.time0) / (t + 1))

        for i in range(n):
            Omega = Omega_parallel[i]
            # acc   = self.sampler.before_sample(Omega, t)
            acc = self.sampler.model_pi.test(Omega, t)

            f = self.sampler.model_pi.train(Omega, produce=True)
            dimension = len(f._coef)  # number of support vectors
            n_samples = sum([len(samples) for samples in f.Omega])

            self.log["accuracy"].data[-1].append(acc)
            self.log["entropy"].data[-1].append(H(f._coef))
            self.log["dimension"].data[-1].append(dimension)
            self.log["n_samples"].data[-1].append(n_samples)
            self.log["density"].data[-1].append(1.0 * dimension / n_samples)
            self.log["log_c"].data[-1].append(np.log10(f.C))

            if isinstance(f._svs[0], str):
                svs = [omit(x) for x in f._svs]
            else:
                svs = [omit(x1) + "/" + omit(x2) for x1, x2 in f._svs]

            if len(svs) == len(f._coef):
                ax[i].set_title(f"LM #{i + 1} ({acc:.3f})")
                ax[i].barh(svs, f._coef)
            else:
                print("Error: dimension missmatch.", svs, coef)

        # Adjusting the layout and adding labels
        plt.tight_layout()

        score = sum(self.log["accuracy"].data[-1]) / len(self.log["accuracy"].data[-1])

        self.log["mean_accuracy"].data[-1] = score

        self.vis.matplot(plt, win="words")

        for k, v in self.log.items():
            if v.plot:
                self.vis.line(
                    X=self.log["epoch"].data,
                    Y=v.data,
                    win=k,
                    opts=dict(xlabel="epoch", ylabel=k),
                )

        fig, ax = plt.subplots(1, 1, figsize=(figscale, figscale))

        def draw_error_band(a, x, y, err, **kwargs):
            # Calculate normals via centered finite differences (except the first point
            # which uses a forward difference and the last point which uses a backward
            # difference).
            dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
            dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
            l = np.hypot(dx, dy)
            nx = dy / l
            ny = -dx / l

            # end points of errors
            xp = x + nx * err
            yp = y + ny * err
            xn = x - nx * err
            yn = y - ny * err

            vertices = np.block([[xp, xn[::-1]], [yp, yn[::-1]]]).T
            codes = np.full(len(vertices), Path.LINETO)
            codes[0] = codes[len(xp)] = Path.MOVETO
            path = Path(vertices, codes)
            a.add_patch(PathPatch(path, **kwargs))

        ax.set_title("Learning Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        std = np.std(np.array([np.array(a) for a in self.log["accuracy"].data]), axis=1)

        ax.plot(self.log["epoch"].data, self.log["mean_accuracy"].data)
        x = self.log["epoch"].data
        y = self.log["mean_accuracy"].data
        yerr = std
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color="C0")

        # draw_error_band(ax, self.log["epoch"].data, self.log["mean_accuracy"].data, err=std,
        #            facecolor="C0", edgecolor="none", alpha=.3)
        self.vis.matplot(plt, win="plt_accuracy")

        stats = {k: (v.data) for k, v in self.log.items()}
        model_dump = {"stats": stats, "Omega_parallel": Omega_parallel}

        print("Saving...")
        with open(f"var/models/{self.envname}.pkl", "wb") as f:
            pickle.dump(model_dump, f)

        return score
