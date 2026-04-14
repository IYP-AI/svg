from svg.completion import *


def seed(y):
    format = """The Corpus of Linguistic Acceptability consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a sequence of words annotated with whether it is a grammatical English sentence.

0: unacceptable
1: acceptable

The possible ten examples of label "{}" are:
"""

    comp = SVGCompletion("text-davinci-003", 1, 256, 0.0, format=format, stop=None)
    str = comp.sample(f"{y}")
    items = str.split("\n")

    import re

    pattern = r"[0-9]+[.:] ['\"]*(.*)['\"]*$"
    items = [re.findall(pattern, str) for str in items]
    items = [lst[0] for lst in items if len(lst) >= 1]
    return items


cola_seed = [seed(i) for i in range(2)]

from svg import SVG

task = "mnli"
description = ("Negative", "Positive")

import random

random.seed(0)
fmt = [
    '"{}" ' + prep + ' "'
    for prep in [
        "and",
        "or",
        "but",
        "to",
        "for",
        "of",
        "as",
        "means",
        "is opposite of",
        "is after",
        "basically implies",
    ]
] + ['Synonym of "{}" is ']

outcome = [["negative"], ["positive"]]

ret = SVG(task=task, temperature=0, p_synth=0.8).sampler.sample(
    xi_0=description, format=fmt, T=10, plot_env="test", n_parallel=1, use_coef=False
)
print(ret)

Omega = ret[0]
for y in range(2):
    for sample in outcome[y]:
        if sample not in Omega[y]:
            print(f"Test failed! {y}-{sample} not in ", Omega[y])
        else:
            print(f"Test succeed. {y}-{sample} in ", Omega[y])
