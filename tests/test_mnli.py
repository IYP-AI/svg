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
