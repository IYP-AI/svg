import pandas as pd
import random


class SVGException(Exception):
    pass


def _(data, trans=False, double="auto"):
    if len(data[0]) == 0:
        raise Exception("Empty data is provided: ", data)

    if double == "auto":
        double = not isinstance(data[0][0], str)

    if double and len(data[0][0]) != 2:
        raise SVGException("Error: shape of the data doesn't match. ", data[0][0])

    if trans:
        items = []
        for y in range(len(data)):
            for x in data[y]:
                if double:
                    items.append([x[0], x[1], y])
                else:
                    items.append([x, y])
        data = items

    return pd.DataFrame(data, columns=["x", "y"] if not double else ["x1", "x2", "y"])


def one_of(x):
    return random.choice(x) if isinstance(x, list) else x


def omit(x, maxlen=20, elipses="...", tail=5):
    if len(x) >= maxlen:
        return x[: maxlen - tail] + elipses + x[-tail:]
    else:
        return x
