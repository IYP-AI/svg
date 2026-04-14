from sklearn import metrics
import uuid
import argparse
import gpt3
import pickle
import pandas as pd

parser = argparse.ArgumentParser(description="Testing a bidirectional classifier.")

parser.add_argument("model_list", type=str, help="model list to be tested")
parser.add_argument("dataset", type=str, help="dataset we use")
parser.add_argument("-n", "--ntrials", type=int, help="number of trials", default=3)
parser.add_argument("-t", "--trial", type=int, help="trial number", default=-1)
parser.add_argument("-i", "--id", type=str, help="ID", default=str(uuid.uuid4())[0:6])
parser.add_argument("-m", type=str, help="m", default=150)
parser.add_argument("--debug_nsamples", type=int, default=None)

args = parser.parse_args()
id = args.id
trial = args.trial
ntrials = args.ntrials
dsname = args.dataset
nsamples = args.debug_nsamples
m = None if args.m == "inf" else int(args.m)

trials = [trial] if trial > 0 else range(1, args.ntrials + 1)
print("Args: ", args)

### MAIN ROUTINE ###
with open(dsname, "rb") as f:
    ds = pickle.load(f)
with open(args.model_list, "rb") as f:
    models = pickle.load(f)

for i in range(ntrials):
    print(f"#{i + 1}> Testing")
    y_true, y_pred = gpt3.test(ds[i][1], models[i], _nsamples=nsamples)
    fpr, tpr, thres = metrics.roc_curve(y_true, y_pred)

    pd.DataFrame([y_true, y_pred], index=["true", "pred"]).T.to_csv(
        "predict-{}-{}.csv".format(i + 1, id)
    )
    pd.DataFrame([fpr, tpr, thres]).T.to_csv("ROC-{}-{}.csv".format(i + 1, id))

    print(f"#{i + 1}> [{id}] auc: {metrics.auc(fpr, tpr)}")

# print(f"The model list has been out to: '{filename}'")
