import re

patterns = [
    # Single-sentence
    [(r"[0-9]+[.:] ['\"]*(.*)['\"]*$", False)],
    # Double-sentence
    [
        (r"[A-Za-z]+:\s([^:\n]+?\.?)\s+[A-Za-z]+:\s([^:\n]+?\.)", True),
        (
            r"[0-9]+ *[.:)-] *[\"]*([A-Z][^-:]+[?.])[\"]*.*?[\"]*([A-Z][^-:]+[?.])[\" ;]*",
            False,
        ),
        (r"[0-9]+ *[.:)-] *[\"]*([A-Z][^-:]+)[\"].*?[\"]([A-Z][^\"]+)[\" ;]*", False),
        (r"[0-9]+ *[.:)-] ([A-Z][^-:]+) - ([A-Z][^\"]+)", False),
        # (r"['\"]*(.+[.]?)['\"]*.*?(.+[.]?)['\" ]*$", False),
        # (r"(.*)-(.*)", False),
    ],
]


class Parser:
    def __init__(self, double):
        self.pattern = patterns[int(double)]

    def parse(self, str, verbose=False):
        for pattern, multi_lines in self.pattern:
            if multi_lines:
                items = re.findall(pattern, str, re.DOTALL)
            else:
                result = [re.findall(pattern, line) for line in str.split("\n")]
                items = [lst[0] for lst in result if len(lst) >= 1]
            if len(items) > 0:
                print("====")
                print(items[0])
                if verbose:
                    return items, pattern, multi_lines
                else:
                    return items
        return None


if __name__ == "__main__":

    def _test(result, expect):
        if result != expect:
            print(f"Test failed! \n Given : {result} \n Expect: {expect}")
        else:
            print(f"Test succeed. {result}")

    questions = {
        "1. The Prime Minister is expected to visit China in the near future. The Prime Minister is anticipated to make a trip to China shortly.": (
            "The Prime Minister is expected to visit China in the near future.",
            "The Prime Minister is anticipated to make a trip to China shortly.",
        ),
        '1) "He will not be attending" - "He will not show up";': (
            "He will not be attending",
            "He will not show up",
        ),
        "1. The cat sat on the mat - The dog was sleeping on the couch.": (
            "The cat sat on the mat",
            "The dog was sleeping on the couch.",
        ),
        "1-Sentence 1: The boy walked down the street. Sentence 2: The street was walked by the boy.": (
            "The boy walked down the street.",
            "The street was walked by the boy.",
        ),
        '1 - "The cat couldn\'t find her kitten" - "The kitten couldn\'t find her cat"': (
            "The cat couldn't find her kitten",
            "The kitten couldn't find her cat",
        ),
    }

    for input, output in questions.items():
        print(input)
        result, pattern, multi_lines = Parser(True).parse(input, verbose=True)
        print(pattern, multi_lines)
        _test(result[0][0], output[0])
        _test(result[0][1], output[1])
