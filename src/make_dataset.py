from gpt3 import compress
import pandas as pd
from datasets import load_dataset

# for name in ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax']:

# Not yet due to errors: "qqp", 'stsb', 'mnli', "ax"
# Validation and test only: 'mnli_mismatched', 'mnli_matched'
# Memory error "qnli"


def sst(key="validation", name="sst2"):
    data = pd.DataFrame(load_dataset("glue", name, split=key))

    print(data)
    if "sentence1" in data:
        data["x1"] = data.sentence1
        data["x2"] = data.sentence2
    elif "question1" in data:
        data["x1"] = data.question1
        data["x2"] = data.question2
    elif "hypothesis" in data:
        data["x1"] = data.hypothesis
        data["x2"] = data.premise
    elif "question" in data:
        data["x1"] = data.question
        data["x2"] = data.sentence
    else:
        data["x"] = data.sentence
    data["y"] = data.label
    return data


# target = ['sst2', 'cola', 'mrpc', 'qqp', 'stsb', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte']
# target = ['stsb', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte']
# target = ['mnli_matched', 'mnli_mismatched']
target = ["sst2"]

if __name__ == "__main__":
    for name in target:
        # ["qqp", 'stsb', 'mnli_mismatched', 'mnli_matched', 'qnli']:
        # for k in ["train"]:
        for k in ["validation"]:
            id = f"v1-{name}-{k}"
            print(id)
            dataset = sst(key=k, name=name)
            print(dataset)
            compress(dataset, id)
