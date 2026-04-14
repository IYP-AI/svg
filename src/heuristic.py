from textattack.transformations import *
from textattack.constraints.pre_transformation import *
from textattack.augmentation import Augmenter


def attack_naive(x):
    u = random.random()
    if 0.7 > u:
        if x.isupper():
            return x.lower()
        else:
            return x.upper()
    elif 1.0 > u:
        return x[0].upper() + x[1:].lower()
    else:
        return x[: random.choice(range(len(x) - 1)) + 1]


transformation = CompositeTransformation(
    [WordSwapRandomCharacterDeletion(), WordSwapQWERTY()]
)
constraints = [RepeatModification(), StopwordModification()]
augmenter = Augmenter(
    transformation=transformation,
    constraints=constraints,
    pct_words_to_swap=0.1,
    transformations_per_example=1,
)


def attack(s, p_attack=0.0):
    if len(s) <= 2:
        return s

    # Augment!
    if p_attack > random.random():
        l = augmenter.augment(s)
        if len(l) >= 1:
            s2 = l[0]
        else:
            s2 = s
    else:
        s2 = attack_naive(s)
    print(f"Attacked: {s} \n\n  -> {s2}")
    return s2
