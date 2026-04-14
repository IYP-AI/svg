import pandas as pd
import os
"""
Large Language Model (LLM) Interface.
This module accesses a frozen language model (e.g., GPT-3) to evaluate or generate
texts used during the Support Vector Generation sampling process, without requiring
fine-tuning or local GPU resources.
"""

import math
import random
import openai
from openai import OpenAI

import getpass
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("\n" + "="*60)
    print("🌟 Welcome to Support Vector Generation (SVG) 🌟")
    print("="*60)
    print("An OpenAI API Key is required for language model inference.")
    api_key = getpass.getpass("🔑 Enter your OPENAI_API_KEY (sk-...): ").strip()
    if not api_key:
        raise ValueError("\n🚨 Error: OPENAI_API_KEY cannot be empty. Please provide a valid key to continue.")

org_id = os.environ.get("OPENAI_ORGANIZATION")
if not org_id:
    org_input = input("🏢 Enter OPENAI_ORGANIZATION ID (press Enter to skip): ").strip()
    if org_input:
        org_id = org_input

openai_client = OpenAI(
    api_key=api_key,
    organization=org_id
)

# Each label must be made of one token. See: https://platform.openai.com/tokenizer
LABELS = ("not", "hot")

OPENAI_COLUMNS = ["prompt", "completion"]
DEFAULT_MODEL = "babbage-002"
MID_FILENAME = "tmp.jsonl"

N = 0
Y = 1


######## METHODS ################ METHODS ################ METHODS ########


def set_keys(organization, api_key):
    global openai_client
    openai_client = OpenAI(api_key=api_key, organization=organization)


# array: a two-dimensional list
# [ [ doc_neg1, doc_neg2, ... ]
#   [ doc_pos1, doc_pos2, ...] ]
def dataframe(array):
    return pd.DataFrame(
        [(_(doc), LABELS[Y]) for doc in array[Y]]
        + [(_(doc), LABELS[N]) for doc in array[N]],
        columns=OPENAI_COLUMNS,
    )


def finetune(df, model=DEFAULT_MODEL):
    resp = upload(df)
    return openai_client.fine_tuning.jobs.create(training_file=resp.id, model=model)


def retrieve(ft_id):
    return openai_client.fine_tuning.jobs.retrieve(ft_id).fine_tuned_model


def predict(row, ft_model):
    return _predict(row.prompt, ft_model), _y(row.completion)


def test(df, ft_model, _nsamples=None):
    print("Connecting to the server... ", ft_model, df.prompt[0])

    cand = range(len(df.prompt))

    # Shrink the sample size for debug
    if _nsamples is not None:
        cand = random.sample(cand, _nsamples)
    else:
        _nsamples = len(cand)

    print("Est. Cost: ${:.1f}".format(est(df) / len(df.prompt) * _nsamples))

    # Predict from each sample
    y_true = []
    y_pred = []
    for i in cand:
        yp, yt = predict(df.iloc[i], ft_model)
        y_true.append(yt)
        y_pred.append(yp)
        print(i, yt, yp)

    return y_true, y_pred


def upload(df):
    df.to_json(MID_FILENAME, orient="records", lines=True)
    return openai_client.files.create(
        file=open(MID_FILENAME, "rb"), purpose="fine-tune"
    )


def est(df, train=False):
    # Suppose we're using Ada finetuned.
    price_per_1k_tokens = 0.0004 if train else 0.0016

    # Suppose each row is made of 2048 bytes.
    n_tokens = 2048

    return price_per_1k_tokens * len(df) * n_tokens / 1000


######## SUB ROUTINES ################ SUB ROUTINES ################ SUB ROUTINES ########


def _(str):
    return str[0:OPENAI_STRLEN] + OPENAI_DELIMITER


def _y(label):
    return Y if label == LABELS[Y] else N


from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_not_exception_type,
)

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = "cl100k_base"

_cache_embed = {}

from pymemcache.client.base import Client
import json
import pickle
import hashlib


def json_serializer(key, value):
    if type(value) == str:
        return value, 1
    return json.dumps(value), 2


def json_deserializer(key, value, flags):
    if flags == 1:
        return value
    if flags == 2:
        return json.loads(value)
    raise Exception("Unknown serialization format")


memcache_host = os.environ.get("MEMCACHED_HOST", "localhost")
memcache_port = int(os.environ.get("MEMCACHED_PORT", "11211"))
client = Client(
    (memcache_host, memcache_port),
    serializer=json_serializer,
    deserializer=json_deserializer,
)


# let's make sure to not retry on an invalid request, because that is what we want to demonstrate
@retry(
    wait=wait_random_exponential(min=1, max=20),
    stop=stop_after_attempt(6),
    retry=retry_if_not_exception_type(openai.BadRequestError),
)
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):

    key = hashlib.md5(text_or_tokens.encode("utf-8")).hexdigest()[:6]
    if text_or_tokens in _cache_embed.keys():
        return _cache_embed[key]

    else:
        from_mem = None  # client.get( key )
        if from_mem is not None:
            return from_mem
        else:
            resp = openai_client.embeddings.create(input=text_or_tokens, model=model)
            x = resp.data[0].embedding
            _cache_embed[key] = x
            # client.set(key, x)
            return x




def _fn(id, split=None):
    data_dir = os.environ.get("DATA_DIR", "/app/var/data")
    os.makedirs(data_dir, exist_ok=True)
    if split is None:
        return os.path.join(data_dir, f"{id}.pkl")
    else:
        return os.path.join(data_dir, f"{id}-{split}.pkl")


def compress(data, id, limit=None, split_every=10000):
    print(data)
    print(f"compressing {id}...")
    rows = []
    i = 0

    if "x1" in data:
        # Double sentence
        for x1, x2, y in data[["x1", "x2", "y"]].values:
            if i in [1, 10, 100, 1000, 10000] or i % 10000 == 0:
                print(f"i={i}")
            rows.append((get_embedding(x1), get_embedding(x2), y))
            i += 1

            if i % split_every == 0:
                fn = _fn(id, i)
                print(f"saving files into {fn}...")
                with open(fn, "wb") as file:
                    pickle.dump(rows, file)
                rows = []

            if limit is not None and i >= limit:
                break
    else:
        # Single sentence
        for x, y in data[["x", "y"]].values:
            if i in [1, 10, 100, 1000, 10000] or i % 10000 == 0:
                print(f"i={i}")
            rows.append((get_embedding(x), y))
            i += 1
            if limit is not None and i >= limit:
                break

    fn = _fn(id)
    print(f"saving files into {fn}...")
    with open(fn, "wb") as file:
        pickle.dump(rows, file)

    return rows


def extract(id):
    with open(_fn(id), "rb") as file:
        return pickle.load(file)


def score(res, richness=0.05, eps=1e-12):
    """Reformat to the probability p( hot | prompt ) from the GPT completion."""

    # Likelihood: p( class | prompt )
    res_dict = res.model_dump() if hasattr(res, "model_dump") else res
    p = res_dict["choices"][0]["logprobs"]["top_logprobs"][0]

    n_label, h_label = LABELS

    # Prior from Laplace smoothing
    hp = eps * richness
    np = eps * (1 - richness)

    # (Unnormalized) posterior
    if h_label in p.keys():
        hp += math.exp(p[h_label])
    if n_label in p.keys():
        np += math.exp(p[n_label])

    # Normalization
    Z = hp + np
    score = hp / Z
    return score


def translate(english, model="babbage-002", max_tokens=256):
    input = f"Please translate the following into Japanese:\n{english}\n"
    resp = openai_client.completions.create(
        model=model, max_tokens=max_tokens, prompt=input, temperature=0.1
    )
    return resp.choices[0].text


def _predict(input, ft_model, maximum_trials=3):
    for i in range(maximum_trials):
        try:
            resp = openai_client.completions.create(
                model=ft_model,
                prompt=input,
                max_tokens=1,  # Each class label should be made of one token
                temperature=0,  # Does not affect to the result because we consider prob directly
                logprobs=len(LABELS),
            )
            return score(resp)
        except openai.APIError as e:
            print(e)
            print("at {}/{}-th trial".format(i, maximum_trials))
    return None
