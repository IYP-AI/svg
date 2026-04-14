import copy

from probabilistic import MCMCSampler, head
from time import time
from .visualizer import Visualizer
from .utils import *


class SVGSampler(MCMCSampler):
    """
    Metropolis-Hastings sampling sequence for Support Vector Generation (SVG).
    This class implements the core algorithm from the paper "Support Vector Generation: 
    Kernelizing Large Language Models for Efficient Zero-Shot NLP" (NeurIPS'25).
    It generates and samples natural-language sentences as explicit support vectors.
    """

    def __init__(self, model, model_pi, model_seed=None, greedy=False):
        super().__init__(model, model_pi)
        self.model_seed = model_seed
        self.greedy = greedy
        self.noise = ["", " ", "\n"]

    def before_sample(self, Omega, t, tau=10, dump_confusion=False, each=False):
        return self.model_pi.test(
            Omega, t, tau=tau, dump_confusion=dump_confusion, each=each
        )

    def _reset(self, Omega_parallel):
        self._Omega_base = Omega_parallel
        self._base = [
            self.model_pi.train(Omega, produce=True, grid=True)
            for Omega in Omega_parallel
        ]

    def sample(
        self,
        T=100,
        t_0=0,
        xi_0=None,
        counter_labeling=False,
        n_batch=10,
        static_y=False,
        plot_env=None,
        pick_last=True,
        use_coef=True,
        with_initial_model=False,
        format=None,
        length_prob=False,
        save_every=100,
        n_parallel=3,
        p_replace=0.1,
        n_seeds=10,
        raw_seeds=False,
        n_shot=0,
    ):

        t0 = time()
        visualizer = Visualizer(self, plot_env)
        dump_timing = [i + 1 for i in range(10)] + [i * i for i in range(1000)]

        if xi_0 is not None:
            seeds = [
                [[x] if isinstance(x, str) else x.copy() for x in xi_0]
                for i in range(n_parallel)
            ]
        else:
            if raw_seeds:
                seeds = self.model_seed.raw(n_parallel)
            else:
                if n_shot > 0:
                    oracle = self.model_seed.original(n_parallel, n_seeds=n_shot)
                else:
                    oracle = None

                if n_seeds > 0:
                    seeds = self.model_seed.cached(
                        n_parallel, n_seeds=n_seeds, base=oracle
                    )
                else:
                    if n_shot > 0:
                        seeds = oracle
                    else:
                        raise SVGException("Either n_shot or n_seeds must be > 0.")

        def Omega0():
            return copy.deepcopy(seeds)

        Omega_parallel = Omega0()
        Omega_base = Omega0()

        acc = 0

        def propose(f, last, reference):
            for i in range(10):
                x_next = self.model.sample(last.x, length_prob=length_prob)
                if x_next != last.x and x_next not in self.noise:
                    break

            y_next = last.y if static_y else None
            return self.model_pi.propose(f.Omega, reference, x=x_next, y=y_next)

        def next_given(f, y, reference):
            last = f.pick(y, pick_last)

            classifier, next = propose(f, last, reference)
            if next.x == last.x or next.x in self.noise:
                return f

            print(
                f"({last.x}, {last.y}) -> ({next.x}, {next.y} [{next.confidence:.3f}])"
            )

            next_a, last_a = classifier.compare_with(last.index) if use_coef else (1, 1)

            last_a = f.a(last.index)

            alpha = self.alpha((next.x, next.confidence, next_a), (last.x, 1, last_a))

            if head(alpha):
                # update( f, last, next )
                print(f"ACCEPT (y={y})")
                return classifier
            else:
                print(f"REJECT (y={y})")
                return f

        def next(f, reference):
            # Evaluation
            self.model.reset(one_of(format))
            n_samples = sum([len(X) for X in f.Omega])
            n_labels = len(f.Omega)
            y = n_samples % n_labels
            return next_given(f, y, reference)

        self._reset(Omega_parallel)
        models = self._base

        try:
            score0 = visualizer.dump(self._base, 0)
        except:
            print("Error while visualizing.")

        print(f"Done initialization (t=0). {time() - t0:.2f} [s]")

        for t in range(1, T + 1):
            print(f"-- Epoch {t} --")

            # TODO:-multi threading
            t0 = time()
            for i in range(n_parallel):
                reference = (
                    self._base[i]
                    if with_initial_model
                    else self._base[(i + t) % n_parallel]
                )
                models[i] = next(models[i], reference)
            print(
                f"Epoch done (t={t}). {time() - t0:.2f} [s], {(time() - t0) / n_parallel * 1000:.2f} [ms/thread]"
            )

            if t in dump_timing:
                t0 = time()
                score = visualizer.dump(models, t)
                if score < score0 and self.greedy:
                    return self._base
                print(f"Visualization done (t={t}). {time() - t0:.2f} [s]")

            if t_0 > 0 and t % t_0 == 0 and not with_initial_model:
                t0 = time()
                self._reset(models)
                print(f"Training of reference done (t={t}). {time() - t0:.2f} [s]")

            print("")

        return models
