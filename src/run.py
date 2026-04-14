import argparse
import random
import numpy as np

from svg import SVG
from task import glue
from heuristic import attack

parser = argparse.ArgumentParser(description="Support Vector Generation (SVG)")

# ========== 1. General ==========

parser.add_argument(
    "task",
    type=str,
    choices=list(glue.keys()),
    help="The task of GLUE you want to solve. ",
)

parser.add_argument("--epochs", default=100, type=int, help="Number of epochs")
parser.add_argument(
    "--parallel", default=3, type=int, help="Number of chains in parallel."
)
parser.add_argument(
    "--n_shot", default=0, type=int, help="The number of train samples."
)

# ========== 2. Model selection and hyperparameters ==========
# The default parameters are tuned for SST-2

# 2.1 Language models

# 2025-08-20 Ohsawa: Modified the default GPT completion model from text-curie-001
# into gpt-3.5-turbo-instruct due to deprecatetion. See also:
# https://platform.openai.com/docs/deprecations#instructgpt-models
parser.add_argument(
    "--model",
    # default="gpt-3.5-turbo-instruct",
    default="babbage-002",
    type=str,
    help="Model of GPT",
)
parser.add_argument("--temperature", default=0.5, type=float, help="Mean temperature")
parser.add_argument(
    "--temperature_std", default=0.05, type=float, help="Std. of the temperature"
)
parser.add_argument("--length", default=20, type=int, help="Length of tokens")
parser.add_argument(
    "--length_prob",
    action="store_true",
    default=False,
    help="Use probabilistic model for the token length.",
)
parser.add_argument(
    "--p_synth",
    default=0.0,
    type=float,
    help="Probability of heuristic text augumentation.",
)
parser.add_argument(
    "--p_replace", default=0.0, type=float, help="Replacing prob for SVs."
)
parser.add_argument(
    "--decision_function_shape", default="ovr", type=str, choices=["ovo", "ovr"]
)

# 2.2 Kernel machines

parser.add_argument("--kernel", default="rbf", type=str, help="Eucledian kernels")

parser.add_argument(
    "--num_logc", default=13, type=int, help="The number of Cs in the logspace."
)
parser.add_argument("--max_logc", default=10, type=int)
parser.add_argument("--min_logc", default=2, type=int)

parser.add_argument(
    "--psvm", action="store_true", default=True, help="Turn on probabilistic SVMs."
)
parser.add_argument("--inbalanced", action="store_true", default=False)


# ========== 3. Variance Reduction ==========

parser.add_argument(
    "--n_seeds",
    default=10,
    type=float,
    help="The number of seeds to be drawn at the initial phase.",
)
parser.add_argument(
    "--burn_in", default=10, type=int, help="Length of the burn-in period for MCMC."
)
parser.add_argument(
    "--without_coef",
    action="store_true",
    default=False,
    help="Use not coefficient. Recommend to turn off.",
)
parser.add_argument(
    "--pick_last",
    action="store_true",
    default=False,
    help="Pick only the last drawn variable. This option makes the sampler pure MH. Recommend to turn off.",
)
parser.add_argument(
    "--without_initial_model",
    action="store_true",
    default=True,
    help="Evaluate support vector sequencially. This option makes the training unstable.",
)
parser.add_argument(
    "--static_y",
    action="store_true",
    default=True,
    help="Set y when generate an examples.",
)
parser.add_argument(
    "--raw",
    action="store_true",
    default=False,
    help="Initializing seeds without transfer learning.",
)
parser.add_argument("--neg", default=None, type=str, help="Negative seed")
parser.add_argument("--pos", default=None, type=str, help="Positive seed")

# ========== 4. Computational Resources ==========

parser.add_argument(
    "--greedy",
    action="store_true",
    default=False,
    help="Stop the script if the performance goes below the initial phase.",
)
parser.add_argument(
    "--refresh_seeds",
    action="store_true",
    default=False,
    help="The cached seeds are normally used so as to reduce the amount of costs. However, if the cache seems broken or you have update the seeding code, then you can turn on this option to refresh the seeds.",
)
parser.add_argument("--plot_env", type=str)

args = parser.parse_args()
print(args)

# ==================================================

# Works for SST2.
# The important thing is conservatively generate the logical entailment.
fmt, stop = ('"{}" as "', ['"', "\n", "."])
if args.task == "mrpc":
    fmt = ['"{}" doesn\'t mean "', fmt]

# alpha is selected by large C=100, then train with small C=1
anealing = lambda: args.temperature + (random.random() - 0.5) * 2 * args.temperature_std

# FIXME: CoLA somehow accepts just "negative"/"postive" seeds
xi_0 = None if args.neg or args.pos is None else [args.neg, args.pos]

SVG(
    length=args.length,
    greedy=args.greedy,
    model=args.model,
    task=args.task,
    temperature=anealing,
    psvm=args.psvm,
    attacker=attack,
    p_synth=args.p_synth,
    stop=stop,
    kernel=args.kernel,
    inbalanced=args.inbalanced,
    C=list(np.logspace(args.min_logc, args.max_logc, args.num_logc)),
    decision_function_shape=args.decision_function_shape,
).sampler.sample(
    xi_0=xi_0,
    format=fmt,
    use_coef=not args.without_coef,
    t_0=args.burn_in,
    raw_seeds=args.raw,
    T=args.epochs,
    n_batch=1,
    pick_last=args.pick_last,
    with_initial_model=not args.without_initial_model,
    length_prob=args.length_prob,
    static_y=args.static_y,
    n_parallel=args.parallel,
    plot_env=args.plot_env,
    p_replace=args.p_replace,
    n_seeds=args.n_seeds,
    n_shot=args.n_shot,
)
