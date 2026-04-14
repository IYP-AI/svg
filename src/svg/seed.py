import pickle
from make_dataset import sst

from probabilistic import Model
from os.path import exists
from task import glue
from .completion import SVGCompletion
from .utils import *
from .parser import Parser


def enum(labels):
    return [f"{i}: {labels[i]}" for i in range(len(labels))]


BATCH_SIZE = 10


class Seed(Model):
    def __init__(self, task, temperature=1.0, length=1024):

        self.task = glue[task]
        print(self.task)

        self.double = len(self.task.samples) >= 2
        self.parser = Parser(self.double)
        self.labels = self.task.labels and enum(self.task.labels)
        self.query = self._query(self.task.samples)
        self.n_classes = len(self.labels)
        self.Y = range(self.n_classes)

        # 2025-08-20 Ohsawa: text-davinci-003 is no longer supporeted as of 2024-01-04.
        # See also: https://platform.openai.com/docs/deprecations#instructgpt-models
        self.model = SVGCompletion(
            "gpt-3.5-turbo-instruct",
            temperature,
            length,
            p_synth=0,
            stop=None,
            format=self._format(),
        )

    def tasks():
        return glue.keys()

    def _query(self, samples):

        if not self.double:
            return (
                f"The possible ten examples of the {samples[0]} whose label is"
                + ' "{}" are:'
            )
        else:
            return (
                f"The possible ten examples of the pair of {samples[0]}-{samples[1]} whose label is"
                + ' "{}" are:'
            )

    def _format(self):

        if self.labels is None:
            return '"{}"'

        else:
            str = self.task.description + "\n"
            str += "\n".join(self.labels)
            str += "\n"
            str += self.query + "\n"

            return str

    def sample(self):
        return [self._cond(y) for y in enum(self.labels)]

    def _cond(self, y, retry=3):

        for i in range(retry):
            try:
                parsed = self._parse(self.model.sample(str(y)))
                if len(parsed) > 0:
                    return parsed
            except SVGException:
                print(f"Retry. #{i}")

        raise SVGException(f"Failed to obtain the seeds. # of retries: {retry}.")

    def _parse(self, str):
        print(str)
        items = self.parser.parse(str)
        if items is None:
            raise SVGException(f"Error while parsing the result: {str}")
        return items

    def _cached(self, n=3, override=False, dir="var/data", n_seeds=10):
        n_batch = n_seeds / 10

        fn = f"{dir}/v1-{self.task.id}-seeds{n}.pkl"

        if not override and exists(fn):
            print("Found: " + fn)
            with open(fn, "rb") as file:
                data = pickle.load(file)

        else:
            data = [self.sample() for i in range(n)]
            with open(fn, "wb") as file:
                pickle.dump(data, file)

        return data

    def raw(self, n=3):
        return [[[x] for x in self.task.labels] for i in range(n)]

    def original(self, n=3, n_seeds=16):
        data = sst(key="train", name="sst2")

        m = len(self.task.labels)

        seeds = []
        for i in range(n):
            seed = []
            for y in range(m):
                snippet = data[data.y == y].sample(
                    n=int(n_seeds / m), replace=False, random_state=i
                )
                seed.append(list(snippet.x))
            seeds.append(seed)
        return seeds

    def cached(
        self, n=3, override=False, dir="var/data", n_seeds=BATCH_SIZE, base=None
    ):
        n_batch = int(n_seeds / BATCH_SIZE)
        if n_batch == 1:
            D_parallel = self._cached(n=n, override=override, dir=dir)
            if base is not None:
                for i in range(len(D_parallel)):
                    D = D_parallel[i]
                    for y in range(len(D)):
                        D[y] = D[y] + base[i][y]
            return D_parallel

        def concat(l, basei):
            ret = basei or [[] for i in self.Y]
            print(l)
            for item in l:
                for y in self.Y:
                    for x in item[y]:
                        ret[y].append(x)
            return ret

        cached = self._cached(n * n_batch, override=override, dir=dir)
        data = [
            concat(cached[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], base and base[i])
            for i in range(n)
        ]

        return data
