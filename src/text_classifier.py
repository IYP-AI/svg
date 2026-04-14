import tiktoken
import numpy as np
import math
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix
from text import embed, _Xy
from collections import namedtuple
from task import glue

SupportVector = namedtuple("SupportVector", ("index", "x", "y", "dual_coef"))
Prediction = namedtuple("Prediction", ("x", "y", "confidence"))


def len_tokens(str):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(str))


def adjust_dict_size(dictionary, size):
    sorted_dict = {
        k: v
        for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)
    }
    adjusted_dict = dict(list(sorted_dict.items())[:size])
    return adjusted_dict


def ranged(theta):
    if not isinstance(theta, list):
        return [theta]
    return theta


class TextClassifier:
    def __init__(
        self,
        prob=False,
        kernel="rbf",
        class_weight="balanced",
        decision_function_shape="ovr",
    ):
        self.prob = prob
        self.kernel = kernel
        self.class_weight = class_weight
        self.cache_tester = None
        self.decision_function_shape = decision_function_shape

    def tokenize(self, docs, dim=None, df_threshold=0, vocab=None):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        df = {}
        bags = []
        for doc in docs:
            tokens = [
                encoding.decode_single_token_bytes(token)
                for token in encoding.encode(doc)
            ]
            bag = {}
            for token in tokens:
                bag[token] = bag[token] + 1 if token in bag else 1
                if bag[token] == 1:
                    df[token] = df[token] + 1 if token in df else 1
            bags.append(bag)

        df_mini = {}
        for token in df.keys():
            if df[token] > df_threshold:
                df_mini[token] = df[token]

        if dim is not None and len(df_mini.keys()) > dim:
            df_mini = adjust_dict_size(df_mini, dim)

        return bags, df_mini

    def vectorize(
        self, tf, df, tfidf=False, scale=True, sparse=True, with_dict_vectorizer=True
    ):
        # DF to indices
        index_of = {}
        index = 0
        for token in df.keys():
            index_of[token] = index
            index += 1

        features = []
        i = 0
        print(f"#DF={len(df.keys())}")
        for bag in tf:
            if i in [1, 10, 100, 1000, 10000] or i % 10000 == 0:
                print(f"i={i}, #TF={len(bag)}")

            if tfidf:
                features.append(
                    [
                        (
                            bag[token] * 1.0 / math.log(df[token] + 1)
                            if token in bag
                            else 0
                        )
                        for token in df.keys()
                    ]
                )
            else:
                if with_dict_vectorizer:
                    features.append(
                        {token: bag[token] for token in df.keys() if token in bag}
                    )
                else:
                    features.append(
                        [(bag[token] if token in bag else 0) for token in df.keys()]
                    )
            i += 1

        print("  making matrix...")
        if sparse:
            if with_dict_vectorizer:
                vectorizer = DictVectorizer(sparse=True)
                feature_matrix = vectorizer.fit_transform(features)
                X = csr_matrix(feature_matrix)
            else:
                X = csr_matrix(features)
        else:
            X = np.array(features)

        print("  preparing a vector...")

        if scale:
            return 1.0 * X / (X.sum() + 1e-5) * len(tf)
        else:
            return X

    def table_to_features(self, data, with_openai_embeddings=True):
        if "x" in data:
            docs = list(data.x)
        else:
            docs = list(zip(list(data.x1), list(data.x2)))

        y = np.array(data.y)

        if with_openai_embeddings:
            return embed(docs, reduce=True), y, None, None

        else:
            print("  tokenizing...")
            tf, df = self.tokenize(docs)
            print("  vectorizing...")
            X = self.vectorize(tf, df)
            return X, y, tf, df

    def least_members_of(self, y_vect):
        cnt = {}
        for y in y_vect:
            if y not in cnt:
                cnt[y] = 0
            cnt[y] += 1
        return min(list(cnt.values()))

    def grid(self, X, y, C, gamma):

        C = np.array(ranged(C or np.logspace(-2, 10, 13))) / len(X) * 20
        gamma = ranged(gamma or np.logspace(-9, 3, 13))

        if len(C) == 1 and len(gamma) == 1:
            return C[0], gamma[0]
        if self.least_members_of(y) == 1:
            print(
                "Warning: least members of y is 1 so the grid search is not required."
            )
            return C[0], gamma[0]

        param_grid = dict(gamma=gamma, C=C)
        max_split = min(5, self.least_members_of(y))
        cv = StratifiedShuffleSplit(
            n_splits=max_split, test_size=1 / max_split, random_state=42
        )
        grid = GridSearchCV(
            SVC(
                kernel=self.kernel,
                probability=self.prob,
                class_weight=self.class_weight,
                decision_function_shape=self.decision_function_shape,
            ),
            param_grid=param_grid,
            cv=cv,
        )
        grid.fit(X, y)

        return grid.best_params_["C"], grid.best_params_["gamma"]

    def train_by(self, X, y, C=100, gamma="scale"):

        C, gamma = self.grid(X, y, C, gamma)
        self.C = C
        self.gamma = gamma

        model = SVC(
            gamma=gamma,
            C=C,
            kernel=self.kernel,
            cache_size=1000,
            probability=self.prob,
            class_weight=self.class_weight,
            decision_function_shape=self.decision_function_shape,
        )

        model.fit(X, y)

        return model

    def i2x(self, i):
        if "x" in self.data:
            return self.data.x[i]
        else:
            return (self.data.x1[i], self.data.x2[i])

    def train(self, data, C=100, gamma="scale"):
        X, y, tf, self.df = self.table_to_features(data)

        self.data = data
        self.model = self.train_by(X, y, C=C, gamma=gamma)
        self._coef = np.array(list(self.model.dual_coef_[0]))

        self._svs = [self.i2x(i) for i in self.model.support_]

        return self.model, X, y, self.df

    def test(self, data, Y=2, encoded=False, task=None, cached=True):

        if task is not None:
            if cached and self.cache_tester is not None:
                X, y = self.cache_tester
            else:
                X, y = _Xy(data)
                self.cache_tester = X, y

            metric = load_metric("glue", task)
            metric.add_batch(predictions=self.model.predict(X), references=y)
            score = metric.compute()

            return score[glue[task].metric]

        else:
            tru = 0
            self.prediction = [None] * 2
            for y in range(Y):
                if encoded:
                    X = np.array([x for x, y_cand in data if y_cand == y])
                else:
                    X, _1, _2, _3 = self.table_to_features(data[data.y == y])

                prediction = self.model.predict(X)
                tru += (prediction == y).sum()

                self.prediction[y] = prediction
            n = len(data)
            return 1.0 * tru / n

    def test_matt(self, data):
        X = np.array([x for x, y in data])
        Y = np.array([y for x, y in data])

        from sklearn.metrics import matthews_corrcoef

        return matthews_corrcoef(self.model.predict(X), Y)

    def predict(self, x, prob=False, y=1):
        phi = embed([x], reduce=True)
        if prob:
            return self.model.predict_proba(phi)[0][y]
        else:
            return self.model.predict(phi)[0]

    def predict_ex(self, x, y=None):
        if y is not None:
            confidence = self.predict(x, y=y, prob=self.prob)
        elif self.prob:
            y = self.predict(x)
            p = self.predict(x, y=1, prob=True)
            confidence = [1 - p, p][y]
        else:
            y = self.predict(x)
            p = y
            confidence = [1 - p, p][y]

        return Prediction(x, y, confidence)

    def kernel(self, x1, x2):
        return np.dot(x1, x2) / math.sqrt(np.dot(x1, x1) * np.dot(x2, x2))

    def is_support_vector(self, str, delta=0.9):
        tf, _ = self.tokenize([str])
        x = self.vectorize(tf, self.df)[0]
        for sv in self.model.support_vectors_:
            if self.kernel(x, sv) > delta:
                return True
        return False

    def pick_i(self, y, pick_last):
        # FIXME: multi-class
        signed_alpha = -self._coef if y == 0 else self._coef
        dist = np.maximum(0, signed_alpha)
        dim = len(dist)
        if pick_last:
            return max([i for i in range(dim) if dist[i] > 0])
        else:
            return np.random.choice(range(dim), p=dist / dist.sum())

    def pick(self, y, pick_last):
        i = self.pick_i(y, pick_last)
        return SupportVector(self.model.support_[i], self._svs[i], y, self._coef[i])

    def a(self, i=None):
        if i is None:
            return self._coef[-1]

        if i not in self.model.support_:
            return 0
        else:
            index = list(self.model.support_).index(i)
            return self._coef[index]

    def compare_with(self, i):
        return self.a(), self.a(i)
