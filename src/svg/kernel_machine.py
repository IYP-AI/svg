import time
import numpy as np

from probabilistic import Model
from text_classifier import TextClassifier
from gpt3 import extract  # FIXME "extract" shuoldn't be here
from .utils import _


class SVGKernelMachine(Model):
    """
    Kernel machine optimization component of SVG.
    This component manages the Support Vector Machine (SVM) training and evaluation,
    using the natural-language sentences drawn by the SVGSampler as support vectors.
    """

    def __init__(
        self,
        C,
        task,
        direct_predcition=False,
        with_coef=True,
        y_prob=False,
        inbalanced=False,
        kernel="rbf",
        decision_function_shape="ovr",
    ):

        self.classifier = TextClassifier(
            prob=y_prob,
            class_weight="balanced" if not inbalanced else None,
            kernel=kernel,
            decision_function_shape=decision_function_shape,
        )
        self.direct_predcition = direct_predcition
        self.with_coef = with_coef
        self.y_prob = y_prob
        self.task = task
        try:
            self.tester = extract(f"v1-{self.task}-validation")
        except FileNotFoundError:
            print(f"[{self.task}] Validation cache not found. Auto-generating from HuggingFace dataset...")
            import make_dataset
            from gpt3 import compress
            dataset = make_dataset.sst(key="validation", name=self.task)
            compress(dataset, f"v1-{self.task}-validation")
            self.tester = extract(f"v1-{self.task}-validation")

        self.C = C if isinstance(C, list) else [C]

    def p(self, x):
        if self.with_coef:
            return x[1] * np.abs(x[2])
        elif self.direct_predcition:
            return 1.0 if self.classifier.predict(x[0]) == x[1] else 0.0
        else:
            return self.classifier.test(x[0])

    def filter(self, xi):
        self.classifier.train(xi)
        return xi[
            xi.apply(lambda row: self.classifier.is_support_vector(row.x), axis=1)
        ]

    def train(self, Omega, C=None, produce=False, grid=False, reference=None):
        if produce and isinstance(Omega, TextClassifier):
            if grid == False:
                return Omega
            else:
                Omega = Omega.Omega

        classifier = TextClassifier(prob=self.y_prob) if produce else self.classifier
        if C is None:
            if grid:
                C = self.C
            elif reference is not None:
                C = reference.C
            else:
                C = self.C[0]  # Default

        t0 = time.time()
        dframe = _(Omega, trans=True)
        classifier.train(dframe, C=C, gamma="scale")
        elapsed = time.time() - t0
        eps = elapsed / len(dframe)
        print(f"Trained SVC: {elapsed:.2f} [s], {eps * 1000:.2f} [ms/vec]")

        classifier.Omega = Omega
        if produce:
            return classifier

    def concat(self, D, sample):
        dicts = [{x: 1 for x in X} for X in D]
        dicts[sample.y][sample.x] = 1

        Omega_proposed = [list(d.keys()) for d in dicts]
        return Omega_proposed

    def predict(self, x, y=None):
        return self.classifier.predict_ex(x, y)

    def propose(self, Omega, reference, x, y):
        proposed = reference.predict_ex(x, y=y)

        tmp = self.concat(Omega, proposed)
        classifier = self.train(tmp, produce=True, reference=reference)

        return classifier, proposed

    def test(self, Omega, t, tau=10, each=False, dump_confusion=False):
        acc = []
        X = self.tester

        C = [Omega.C] if isinstance(Omega, TextClassifier) else self.C

        # Grid
        for c in C:
            f = self.train(Omega, C=c, produce=True)
            if self.task == "cola":
                acc.append(f.test_matt(X))
            elif self.task == "sst2":
                acc.append(f.test(X, encoded=True))
            else:
                acc.append(f.test(X, encoded=True, task=self.task))

            if dump_confusion:
                validation = sst()
                xs = [validation[validation.y == y] for y in range(2)]
                confusion = [xs[f.prediction[y] != y].x for y in range(2)]

                print("# Confusion")
                print(confusion[0])
                print(confusion[1])

        print(f"Performance: {np.max(acc):.3f}")

        if len(acc) == 1:
            return acc[0]
        elif each:
            return acc
        else:
            return np.max(acc)
