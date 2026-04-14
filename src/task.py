# The model agnostic task description module.
# Each task description as well as label and class of sample has been scraped from
#    https://www.tensorflow.org/datasets/catalog/glue
# which is also based on the original paper.

from collections import namedtuple

Task = namedtuple("Task", ("id", "description", "labels", "samples", "metric"))

_original = [
    # 1. Single sentence
    Task(
        "cola",
        "The Corpus of Linguistic Acceptability consists of English acceptability judgments drawn from books and journal articles on linguistic theory. Each example is a sequence of words annotated with whether it is a grammatical English sentence.",
        ("unacceptable", "acceptable"),
        ("sentence",),
        "matthews_correlation",
    ),
    Task(
        "sst2",
        "The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. We use the two-way (positive/negative) class split, and use only sentence-level labels.",
        ("negative", "positive"),
        ("sentence",),
        "accuracy",
    ),
    # 2. Double sentence
    # 2.1 Paraphrase
    Task(
        "mrpc",
        "The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.",
        ("not equivalent", "equivalent"),
        ("sentence1", "sentence2"),
        "f1",
    ),
    Task(
        "qqp",
        "The Quora Question Pairs2 dataset is a collection of question pairs from the community question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent.",
        ("not duplicate", "duplicate"),
        ("question1", "question2"),
        "f1",
    ),
    # 2.2 Entailment, NLI
    Task(
        "mnli",
        "The Multi-Genre Natural Language Inference Corpus is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports. We use the standard test set, for which we obtained private labels from the authors, and evaluate on both the matched (in-domain) and mismatched (cross-domain) section. We also use and recommend the SNLI corpus as 550k examples of auxiliary training data.",
        ("entailment", "neutral", "contradiction"),
        ("premise", "hypothesis"),
        "accuracy",
    ),
    Task(
        "qnli",
        "The Stanford Question Answering Dataset is a question-answering dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator). We convert the task into sentence pair classification by forming a pair between each question and each sentence in the corresponding context, and filtering out pairs with low lexical overlap between the question and the context sentence. The task is to determine whether the context sentence contains the answer to the question. This modified version of the original task removes the requirement that the model select the exact answer, but also removes the simplifying assumptions that the answer is always present in the input and that lexical overlap is a reliable cue.",
        ("entailment", "not entailment"),
        ("question", "sentence"),
        "accuracy",
    ),
    Task(
        "rte",
        "The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual entailment challenges. We combine the data from RTE1 (Dagan et al., 2006), RTE2 (Bar Haim et al., 2006), RTE3 (Giampiccolo et al., 2007), and RTE5 (Bentivogli et al., 2009).4 Examples are constructed based on news and Wikipedia text. We convert all datasets to a two-class split, where for three-class datasets we collapse neutral and contradiction into not entailment, for consistency.",
        ("entailment", "not entailment"),
        ("hypothesis", "premise"),
        "accuracy",
    ),
]

glue = {task.id: task for task in _original}
glue["mnli_mismatched"] = glue["mnli"]
glue["mnli_matched"] = glue["mnli"]
