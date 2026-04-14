from probabilistic import Model
import random
import openai
from gpt3 import openai_client
import numpy as np
from text_classifier import len_tokens


class SVGCompletion(Model):
    def __init__(
        self,
        model,
        temperature,
        length,
        p_synth,
        format=None,
        attacker=None,
        max_trials=3,
        max_length=1000,
        stop=None,
    ):
        self.model = model
        self.temperature = temperature
        self.length = length
        self.format = format
        self.max_trials = max_trials
        self.max_length = max_length
        self.p_synth = p_synth
        self.attacker = SVGAttack(attacker)
        self.stop = stop
        self.reset(self.format)

    # Randomizing the state
    def reset(self, format):
        if isinstance(self.model, list):
            self.current_model = random.choice(self.model)
        else:
            self.current_model = self.model

        if isinstance(self.temperature, type(lambda: None)):
            self.current_temp = self.temperature()
        else:
            self.current_temp = self.temperature

        self.do_attack = self.p_synth > random.random()
        self.format = format

    def logp(self, x):
        if len_tokens(x) <= 3:
            return 0.0

        return sum(
            self._try(
                lambda: openai_client.completions.create(
                    model=self.current_model,
                    max_tokens=0,
                    prompt=x,
                    echo=True,
                    logprobs=0,
                    temperature=self.current_temp,
                )
            )
            .choices[0]
            .logprobs.token_logprobs[1:]
        )

    # Pr(x'|x) = Pr(x, x') / Pr(x)
    def logp_transition(self, x_from, x_to, backward=False, format=None):
        if not isinstance(x_from, str):
            x_from = x_from[0]
        if not isinstance(x_to, str):
            x_to = x_to[0]

        if self.do_attack:
            return self.attacker.logp_transition(x_from, x_to)

        format = format or self.format
        if isinstance(format, list) or format is not None:
            prompt = self._prompt(x_from, format=format)
            return self.logp(prompt + x_to) - self.logp(
                x_from
            )  # FIXME: x_from or prompt
        else:
            if backward:
                return self.logp(x_to + x_from) - self.logp(x_from)
            else:
                return self.logp(x_from + x_to) - self.logp(x_from)

    def sample(
        self,
        x_from,
        length=None,
        temperature=None,
        format=None,
        p_synth=0,
        length_prob=False,
    ):

        if self.do_attack:
            return self.attacker.sample(x_from)

        prompt = self._prompt(x_from, format=format or self.format)
        temperature = self._beta(temperature or self.current_temp)
        length = self._len(length or self.length, length_prob)

        return (
            self._try(
                lambda: openai_client.completions.create(
                    model=self.current_model,
                    max_tokens=length,
                    prompt=prompt,
                    logprobs=0,
                    temperature=temperature,
                    stop=self.stop,
                )
            )
            .choices[0]
            .text
        )

    def _prompt(self, x, format=None):
        format = format or self.format
        if isinstance(self.format, str):
            return self.format.format(x)
        elif format is not None:
            return random.choice(self.format).format(x)
        else:
            return x

    def _len(self, length, length_prob):
        if length_prob:
            return min(self.max_length, np.random.poisson(length or self.length))
        else:
            return length

    def _beta(self, temperature):
        if isinstance(temperature, type(lambda: None)):
            return self.current_temp
        else:
            return temperature

    def _try(self, func):
        for i in range(self.max_trials):
            try:
                return func()
            except openai.OpenAIError:
                pass


class SVGAttack(Model):
    def __init__(self, attacker=None):
        self.attacker = attacker

    # Pr(x'|x) = Pr(x, x') / Pr(x)
    def logp_transition(self, x_from, x_to):
        return 0.0

    def sample(self, x):
        if self.attacker is not None:
            return self.attacker(x)

        u = random.random()
        if 0.7 > u:
            # if x.isupper():
            if 0.5 > random.random():
                return x.lower()
            else:
                return x.upper()
        elif 1.0 > u:
            return x[0].upper() + x[1:].lower() if len(x) > 2 else x
        else:
            return x[: random.choice(range(len(x) - 1)) + 1]
