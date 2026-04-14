from .kernel_machine import *
from .completion import *
from .double_completion import *
from .sampler import *
from .seed import *


class SVG:
    def __init__(
        self,
        classifier=None,
        generator=None,
        sampler=None,
        seed=None,
        task="sst2",
        n_batch=3,
        length=100,
        format=None,
        greedy=False,
        # 2025-08-20 Ohsawa: text-curie-001 is no longer supporeted as of 2024-01-04.
        # See also: https://platform.openai.com/docs/deprecations#instructgpt-models
        model="gpt-3.5-turbo-instruct",
        temperature=0.3,
        C=1e6,
        with_coef=True,
        psvm=False,
        attacker=None,
        p_synth=0.8,
        stop=[".", '"', "\n"],
        inbalanced=False,
        kernel="rbf",
        decision_function_shape="ovr",
    ):

        self.seed = seed or Seed(task)

        Generator = DoubleCompletion if self.seed.double else SVGCompletion

        self.generator = generator or Generator(
            length=length,
            model=model,
            temperature=temperature,
            attacker=attacker,
            p_synth=p_synth,
            stop=stop,
        )

        self.classifier = classifier or SVGKernelMachine(
            direct_predcition=True,
            with_coef=with_coef,
            y_prob=psvm,
            C=C,
            task=task,
            inbalanced=inbalanced,
            kernel=kernel,
            decision_function_shape=decision_function_shape,
        )

        self.sampler = sampler or SVGSampler(
            self.generator,
            self.classifier,
            model_seed=self.seed,
            greedy=greedy,
        )
