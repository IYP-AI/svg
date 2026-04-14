from .completion import SVGCompletion
from .utils import *


class DoubleCompletion(SVGCompletion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def logp(self, x, *args, **kwargs):
        return super(DoubleCompletion, self).logp(x, *args, **kwargs)

    def logp_transition(self, x_from, x_to, *args, **kwargs):
        if not isinstance(x_from[0], str):
            x_from = x_from[0]
        if not isinstance(x_to[0], str):
            x_to = x_to[0]

        return super(DoubleCompletion, self).logp_transition(
            x_from[1], x_to[0], *args, **kwargs
        )

    def sample(self, x_from, *args, **kwargs):
        if len(x_from) < 2:
            raise SVGException("Argument is not a sentence-pair.")
        else:
            _, sentence_from = x_from

        sentence_to = super(DoubleCompletion, self).sample(
            sentence_from, *args, **kwargs
        )
        return (sentence_from, sentence_to)
