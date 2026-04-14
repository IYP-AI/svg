import numpy as np
from gpt3 import get_embedding


def _Xy(data):
    if len(data[0]) <= 2:
        w = np.array(data)
        return np.array(w[:, 0]), np.array(w[:, 1])

    else:
        X1 = np.array([x1 for x1, x2, y in data])
        X2 = np.array([x2 for x1, x2, y in data])
        y = np.array([y for x1, x2, y in data])
        X = X2 - X1
        return X, y


def embed(docs, reduce=False):
    if isinstance(docs[0], str):
        return np.array([get_embedding(doc) for doc in docs])
    else:
        X = np.array([[get_embedding(d1), get_embedding(d2)] for d1, d2 in docs])
        if reduce:
            return np.subtract.reduce(X, axis=1)
        else:
            return X


if __name__ == "__main__":
    X = embed([("A", "B"), ("C", "D")])
    print(np.subtract.reduce(X, axis=1).shape)
